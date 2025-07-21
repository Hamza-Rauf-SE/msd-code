# pip install unsloth
# pip install --no-deps bitsandbytes
# pip install "trl<0.9.0" accelerate
# pip install scikit-learn matplotlib seaborn
# 1. Imports
import os
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
import gc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling, LogitsProcessor, LogitsProcessorList
from typing import List, Union, Any, Dict
from tqdm import tqdm

# ===================================================================
# 2. Configuration
# ===================================================================
model_name = "unsloth/gemma-3-4b-pt"
NUM_CLASSES = 2
max_seq_length = 2048

# ===================================================================
# 3. Load Model and Processor/Tokenizer
# ===================================================================
print("Loading model and tokenizer...")
model, processor = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)
text_tokenizer = processor.tokenizer
if text_tokenizer is None:
    raise RuntimeError("Text tokenizer could not be loaded from the processor.")
# Add a padding token if it doesn't exist (Gemma models often don't have one)
if text_tokenizer.pad_token is None:
    text_tokenizer.pad_token = text_tokenizer.eos_token

# ===================================================================
# 4. Trim the Classification Head
# ===================================================================
print("Trimming language model head for classification...")
number_token_ids = [text_tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(NUM_CLASSES)]
print(f"Token IDs for our labels {list(range(NUM_CLASSES))}: {number_token_ids}")

# IMPORTANT: We do not trim the head before training. We will do this after merging adapters for inference.
# This makes the training setup simpler and more stable.
reverse_map = {token_id: i for i, token_id in enumerate(number_token_ids)}

# ===================================================================
# 5. Configure PEFT / LoRA
# ===================================================================
print("Configuring LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"], # Targeting lm_head is good practice
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=True,
)
print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# ===================================================================
# 6. Load and Prepare Your Dataset
# ===================================================================
print("Loading and preparing the multilingual spam dataset...")
# Load the dataset from Hugging Face Hub
full_dataset = load_dataset("hamza-khan/spam-multilingual-2k-per-lang", split="train")
df = full_dataset.to_pandas()

# Create training and validation splits
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)
print(f"Train size: {len(train_df)}")
print(f"Validation size: {len(val_df)}")

# Define the prompt template
prompt = "Classify the following email into 'ham' (class 0) or 'spam' (class 1).\n\nEmail:\n{}\n\nClassification:\nThe correct answer is: class {}"

def formatting_prompts_func(examples):
    texts = [prompt.format(text, str(label)) + text_tokenizer.eos_token for text, label in zip(examples['text'], examples['label'])]
    return {"formatted_text": texts}

# Apply formatting
train_dataset = Dataset.from_pandas(train_df).map(formatting_prompts_func, batched=True)

# ===================================================================
# 7. Custom Data Collator
# ===================================================================
class DataCollatorForLastTokenLM(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.ignore_index = -100

    def torch_call(self, examples: List[Union[Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)
        for i in range(len(examples)):
            labels = batch["labels"][i]
            # Find the last non-ignored token
            if (labels != self.ignore_index).any():
                last_token_idx = (labels != self.ignore_index).nonzero()[-1].item()
                last_token_id = labels[last_token_idx].item()
                # Ignore all tokens except the last one
                batch["labels"][i, :last_token_idx] = self.ignore_index
                # We don't need reverse_map for training as we train on the full lm_head with LoRA
            else:
                # Handle cases with no valid labels
                batch["labels"][i] = self.ignore_index
        return batch

collator = DataCollatorForLastTokenLM(tokenizer=text_tokenizer)

# ===================================================================
# 8. Set up SFTTrainer and Train!
# ===================================================================
print("Starting training...")
trainer = SFTTrainer(
    model=model,
    tokenizer=text_tokenizer,
    train_dataset=train_dataset,
    data_collator=collator,
    dataset_text_field="formatted_text",
    max_seq_length=max_seq_length,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=3, # Adjusted for a larger dataset
        learning_rate=1e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="./outputs_gemma",
        report_to="tensorboard",
        group_by_length=True,
    ),
)
trainer.train()
print("Training complete.")

# ===================================================================
# 9. INFERENCE PREPARATION
# ===================================================================
print("\n--- Preparing for Inference ---")
# Clean up memory before inference
del collator, train_dataset, df, full_dataset
del trainer
gc.collect()
torch.cuda.empty_cache()
print("Cleaned up training-related memory.")

# Merge LoRA adapters into the base model for faster inference
print("Merging LoRA adapters...")
model = model.merge_and_unload()
print("Adapters merged successfully.")

FastLanguageModel.for_inference(model)

# ===================================================================
# 10. Run Full Evaluation on Validation Set
# ===================================================================
print("Starting evaluation on the validation set...")

# Use a logits processor to constrain the model's output to only "0" or "1"
class ConstrainToClassTokens(LogitsProcessor):
    def __init__(self, allowed_token_ids: list[int]):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -float("Inf"))
        mask[:, self.allowed_token_ids] = 0
        return scores + mask

logits_processor = LogitsProcessorList([ConstrainToClassTokens(number_token_ids)])

# Inference prompt template
inference_prompt = "Classify the following email into 'ham' (class 0) or 'spam' (class 1).\n\nEmail:\n{}\n\nClassification:\nThe correct answer is: class "

# Prepare for batch inference
all_preds = []
all_true = val_df['label'].tolist()
batch_size = 8
device = "cuda"

for i in tqdm(range(0, len(val_df), batch_size), desc="Evaluating"):
    batch_df = val_df.iloc[i:i+batch_size]
    prompts = [inference_prompt.format(text) for text in batch_df['text']]

    inputs = text_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length-1).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        pad_token_id=text_tokenizer.eos_token_id,
        logits_processor=logits_processor
    )
    
    # Decode only the generated token
    generated_tokens = outputs[:, -1]
    decoded_preds = text_tokenizer.batch_decode(generated_tokens)
    
    # Convert string predictions ("0" or "1") to integers
    int_preds = [int(p.strip()) for p in decoded_preds]
    all_preds.extend(int_preds)

# ===================================================================
# 11. Calculate and Display Metrics
# ===================================================================
accuracy = accuracy_score(all_true, all_preds)
precision = precision_score(all_true, all_preds, zero_division=0)
recall = recall_score(all_true, all_preds, zero_division=0)
f1 = f1_score(all_true, all_preds, zero_division=0)

print("\n--- QLoRA Fine-Tuned Gemma-3-4B Model Evaluation Metrics ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("-" * 30)

# --- Plotting ---
cm = confusion_matrix(all_true, all_preds)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (QLoRA Fine-Tuned)')
plt.show()


# Clean up memory
del model, trainer
gc.collect()
torch.cuda.empty_cache()