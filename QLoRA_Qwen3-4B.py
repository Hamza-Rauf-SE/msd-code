# pip install unsloth
# pip install --no-deps bitsandbytes
# pip install "trl<0.9.0" accelerate
# pip install scikit-learn matplotlib seaborn
import os
import torch
import pandas as pd
import gc
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer
from peft import LoftQConfig
from unsloth import FastLanguageModel
from unsloth import tokenizer_utils
from typing import Any, Dict, List, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from tqdm import tqdm
import torch.nn.functional as F

# --- 1. SETUP & PATCHING ---

def do_nothing(*args, **kwargs):
    pass
tokenizer_utils.fix_untrained_tokens = do_nothing
print("Applied Unsloth patch for lm_head resizing.")

# --- 2. DATA LOADING AND PREPARATION ---

print("Loading dataset: hamza-khan/spam-multilingual-2k-per-lang")
dataset = load_dataset("hamza-khan/spam-multilingual-2k-per-lang", split="train")
df = dataset.to_pandas()

prompt_template = """Here is an email text:
{}
Classify this email into one of the following:
class 0: Ham (Not Spam)
class 1: Spam
SOLUTION
The correct answer is: class {}"""

# We need the tokenizer to be loaded before we can format the prompts
# So this function will be applied later.
def formatting_prompts_func(dataset_row, tokenizer):
    text_ = dataset_row['text']
    label_ = dataset_row['label']
    return prompt_template.format(text_, f" {label_}") + tokenizer.eos_token

train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label']
)
print(f"Train size: {len(train_df)}")
print(f"Validation size: {len(val_df)}")
print("-" * 30)

# --- 3. MODEL INITIALIZATION ---

max_seq_length = 2048
dtype = None
load_in_4bit = True
model_name = "unsloth/Qwen3-4B-Base"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Now apply the formatting function
train_df['formatted_text'] = train_df.apply(lambda row: formatting_prompts_func(row, tokenizer), axis=1)
val_df['formatted_text'] = val_df.apply(lambda row: formatting_prompts_func(row, tokenizer), axis=1)

train_dataset = Dataset.from_pandas(train_df[['formatted_text']], preserve_index=False)
eval_dataset = Dataset.from_pandas(val_df[['formatted_text']], preserve_index=False)

# --- 4. LM_HEAD TRIMMING & PEFT CONFIGURATION ---

print("Trimming the language model head for classification...")
number_token_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(2)]
new_lm_head = torch.nn.Parameter(model.lm_head.weight[number_token_ids, :])
model.lm_head.weight = new_lm_head
reverse_map = {token_id: idx for idx, token_id in enumerate(number_token_ids)}
print(f"Trimmed lm_head. Target token IDs: {number_token_ids}. Reverse map: {reverse_map}")

# Configure PEFT (LoRA)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "lm_head", "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("-" * 30)

# --- 5. CUSTOM DATA COLLATOR & TRAINING ---

class DataCollatorForLastTokenLM(DataCollatorForLanguageModeling):
    def __init__(self, *args, mlm: bool = False, ignore_index: int = -100, **kwargs):
        super().__init__(*args, mlm=mlm, **kwargs)
        self.ignore_index = ignore_index

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)
        for i in range(len(examples)):
            labels = batch["labels"][i]
            last_token_idx = (labels != self.ignore_index).nonzero()[-1].item()
            label_token_id = labels[last_token_idx].item()
            batch["labels"][i, :last_token_idx] = self.ignore_index
            if label_token_id in reverse_map:
                batch["labels"][i, last_token_idx] = reverse_map[label_token_id]
            else:
                batch["labels"][i, last_token_idx] = self.ignore_index
        return batch

collator = DataCollatorForLastTokenLM(tokenizer=tokenizer, mlm=False)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        warmup_steps=10,
        num_train_epochs=2,
        learning_rate=1e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="./results",
        report_to="tensorboard",
        group_by_length=True,
    ),
    data_collator=collator,
    dataset_text_field="formatted_text",
)

print("Starting training...")
trainer_stats = trainer.train()
print("Training complete.")
print("-" * 30)

# --- 6. INFERENCE AND EVALUATION (Code remains the same) ---

FastLanguageModel.for_inference(model)
print("Model set for inference.")

def inference_prompt_template(text: str) -> str:
    return f"""Here is an email text:
{text}
Classify this email into one of the following:
class 0: Ham (Not Spam)
class 1: Spam
SOLUTION
The correct answer is: class """

val_inference_df = val_df[['text', 'label']].copy()
val_inference_df['token_length'] = val_inference_df['text'].apply(
    lambda x: len(tokenizer.encode(inference_prompt_template(x)))
)
val_sorted = val_inference_df.sort_values('token_length').reset_index(drop=True)

batch_size = 16
device = model.device
all_preds, all_true, all_probs = [], [], []

model.eval()
with torch.inference_mode():
    for start in tqdm(range(0, len(val_sorted), batch_size), desc="Evaluating"):
        batch_df = val_sorted.iloc[start:start+batch_size]
        prompts = [inference_prompt_template(t) for t in batch_df['text']]
        inputs = tokenizer(
            prompts, return_tensors='pt', padding=True, truncation=True, max_length=max_seq_length
        ).to(device)
        logits = model(**inputs).logits.cpu()
        last_token_indices = inputs['attention_mask'].sum(dim=1) - 1
        last_logits = logits[torch.arange(len(last_token_indices)), last_token_indices, :]
        
        # We get logits for our two classes directly since lm_head is trimmed
        class_logits = last_logits 
        
        class_probs = F.softmax(class_logits, dim=-1)
        preds = class_probs.argmax(dim=-1).numpy()
        probs_class_1 = class_probs[:, 1].numpy()
        true_labels = batch_df['label'].to_numpy()

        all_preds.extend(preds)
        all_true.extend(true_labels)
        all_probs.extend(probs_class_1)

# --- Calculate and Display Metrics ---

accuracy = accuracy_score(all_true, all_preds)
precision = precision_score(all_true, all_preds)
recall = recall_score(all_true, all_preds)
f1 = f1_score(all_true, all_preds)
auc_roc = roc_auc_score(all_true, all_probs)

print("\n--- QLoRA Fine-Tuned Model Evaluation Metrics ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")
print("-" * 30)

# --- Plotting ---
cm = confusion_matrix(all_true, all_preds)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (QLoRA Fine-Tuned)')
plt.show()

fpr, tpr, _ = roc_curve(all_true, all_probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (QLoRA Fine-Tuned)')
plt.legend(loc="lower right")
plt.show()

# Clean up memory
del model, tokenizer, trainer
gc.collect()
torch.cuda.empty_cache()