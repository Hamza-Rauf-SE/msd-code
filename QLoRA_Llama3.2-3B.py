# ===================================================================
# 1. Imports 
# ===================================================================
# pip install -q unsloth "trl<0.9.0" accelerate datasets
# pip install -q scikit-learn matplotlib seaborn pandas tqdm

# huggingface-cli login

from unsloth import FastLanguageModel
import os
import torch
import pandas as pd
from datasets import load_dataset, Dataset
import gc
from tqdm import tqdm

# Import the correct trainer and a more flexible data collator
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling, LogitsProcessor, LogitsProcessorList
from typing import List, Dict, Any, Union

# For evaluation
from sklearn.model_selection import train_test_split ### FIX: Import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# ===================================================================
# 2. Configuration
# ===================================================================
model_name = "unsloth/Llama-3.2-3B-bnb-4bit"
NUM_CLASSES = 2
max_seq_length = 2048 # Choose a reasonable length for emails

# ===================================================================
# 3. Load and Prepare the Real Dataset
# ===================================================================
print("Loading the spam dataset from Hugging Face...")
full_dataset = load_dataset("hamza-khan/spam-multilingual-2k-per-lang", split="train")
df = full_dataset.to_pandas()

# ### FIX: Use sklearn's train_test_split correctly
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['label'] # Stratify to keep label distribution similar in both sets
)

print(f"Dataset loaded and split.")
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print("\nSample from training data:")
print(train_df.head(1))

# ### FIX: Convert pandas DataFrames back to Hugging Face Dataset objects
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)


# ===================================================================
# 4. Load Llama-3.2 Model and Tokenizer
# ===================================================================
print(f"\nLoading model: {model_name} using Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)
# Unsloth handles pad_token setup automatically.

# ===================================================================
# 5. Configure PEFT / LoRA
# ===================================================================
print("Configuring LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
print("Trainable parameters:", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ===================================================================
# 6. Prepare Data, Prompts, and Custom Data Collator
# ===================================================================
prompt_template = "Classify the following email into 'ham' (class 0) or 'spam' (class 1).\n\nEmail:\n{}\n\nClassification:\nThe correct answer is: class {}"

def formatting_prompts_func(examples):
    texts = [prompt_template.format(text, str(label)) + tokenizer.eos_token for text, label in zip(examples['text'], examples['label'])]
    return {"formatted_text": texts}

# ### FIX: Apply mapping to the Dataset object, not the DataFrame
formatted_train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

class DataCollatorForLastTokenLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)
        for i in range(len(examples)):
            labels = batch["labels"][i]
            last_token_indices = (labels != -100).nonzero(as_tuple=True)[0]
            if len(last_token_indices) > 0:
                last_token_idx = last_token_indices[-1]
                labels[:last_token_idx] = -100
        return batch

data_collator = DataCollatorForLastTokenLM(tokenizer=tokenizer, mlm=False)


# ===================================================================
# 7. Set up SFTTrainer and Train
# ===================================================================
print("\nStarting training with SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_train_dataset,
    dataset_text_field="formatted_text",
    data_collator=data_collator,
    max_seq_length=max_seq_length,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=1, # 1 epoch is often enough for good results with large datasets
        learning_rate=1e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        output_dir="outputs",
        save_strategy="epoch",
        report_to="none",
    ),
)
trainer.train()
print("Training complete.")


# ===================================================================
# 8. Evaluation with Metrics and Confusion Matrix
# ===================================================================
print("\n--- Starting Evaluation on Validation Set ---")
model.eval()

number_token_ids = [tokenizer.encode(str(i), add_special_tokens=False)[-1] for i in range(NUM_CLASSES)]
print(f"Token IDs for classes 0 and 1: {number_token_ids}")

class ConstrainToClassTokens(LogitsProcessor):
    def __init__(self, allowed_token_ids: list[int]):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -float("Inf"))
        mask[:, self.allowed_token_ids] = 0
        return scores + mask

logits_processor = LogitsProcessorList([ConstrainToClassTokens(number_token_ids)])

inference_prompt = "Classify the following email into 'ham' (class 0) or 'spam' (class 1).\n\nEmail:\n{}\n\nClassification:\nThe correct answer is: class "

true_labels = []
predicted_labels = []

# ### FIX: Correctly iterate over a pandas DataFrame using iterrows()
for index, row in tqdm(val_df.iterrows(), total=val_df.shape[0], desc="Evaluating"):
    true_label = row['label']
    email_text = row['text']

    inputs = tokenizer([inference_prompt.format(email_text)], return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        pad_token_id=tokenizer.eos_token_id,
        logits_processor=logits_processor,
        do_sample=False
    )
    
    predicted_token = outputs[0, -1]
    decoded_prediction = tokenizer.decode(predicted_token)

    true_labels.append(true_label)
    try:
        predicted_labels.append(int(decoded_prediction.strip()))
    except ValueError:
        predicted_labels.append(-1)

# --- Calculate and Display Metrics ---
print("\n--- Evaluation Results ---")

accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='binary', zero_division=0)
recall = recall_score(true_labels, predicted_labels, average='binary', zero_division=0)
f1 = f1_score(true_labels, predicted_labels, average='binary', zero_division=0)

print("\n--- QLoRA Fine-Tuned Llama-3.2-3B Model Evaluation Metrics ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("-" * 60)

print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, target_names=['ham (0)', 'spam (1)'], zero_division=0))

print("\nConfusion Matrix:")
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Ham', 'Predicted Spam'],
            yticklabels=['Actual Ham', 'Actual Spam'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Spam Classification')
plt.show()


# ===================================================================
# 9. Clean up
# ===================================================================
print("\nCleaning up resources...")
del model, tokenizer, trainer, full_dataset, train_df, val_df, train_dataset, val_dataset
gc.collect()
torch.cuda.empty_cache()
print("Done.")