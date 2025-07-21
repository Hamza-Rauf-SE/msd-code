# ===================================================================
# 1. Imports 
# ===================================================================
# pip install -q unsloth bitsandbytes "trl<0.9.0" accelerate datasets
# pip install -q scikit-learn matplotlib seaborn pandas tqdm

# huggingface-cli login

from unsloth import FastLanguageModel
import os
import torch
import pandas as pd
from datasets import load_dataset, Dataset
import gc
from tqdm import tqdm

# Import necessary components for inference and evaluation
from transformers import LogitsProcessor, LogitsProcessorList
from typing import List, Dict, Any, Union

# For evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ===================================================================
# 2. Configuration
# ===================================================================
model_name = "unsloth/Llama-3.2-3B-bnb-4bit"
NUM_CLASSES = 2
max_seq_length = 2048

# ===================================================================
# 3. Load and Prepare the EXACT SAME Validation Dataset
# ===================================================================
print("Loading the spam dataset from Hugging Face...")
full_dataset = load_dataset("hamza-khan/spam-multilingual-2k-per-lang", split="train")
df = full_dataset.to_pandas()

# Recreate the exact same train/validation split to ensure a fair comparison
# Using the same random_state is critical here.
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42, # Must be the same as in the training script
    stratify=df['label']
)

print(f"Dataset loaded. Using the same validation set for a fair comparison.")
print(f"Validation samples to be evaluated: {len(val_df)}")


# ===================================================================
# 4. Load the BASE Llama-3.2 Model (NO quantization, Fine-tuning, NO 4-bit)
# ===================================================================
print(f"\nLoading BASE model: {model_name} in bfloat16 (load_in_4bit=False)...")
# Note: This will use significantly more VRAM than the 4-bit version.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16, # Use bfloat16 for speed and memory efficiency
    load_in_4bit=False,   # <<< THE KEY CHANGE: Loading the full model
)

# No PEFT/LoRA section is needed since we are evaluating the base model.
# No Trainer section is needed since we are not training.

# ===================================================================
# 5. Evaluation with Metrics and Confusion Matrix (Zero-Shot)
# ===================================================================
print("\n--- Starting Zero-Shot Evaluation on the Base Model ---")
model.eval()

# Get token IDs for "0" and "1" to constrain the output
number_token_ids = [tokenizer.encode(str(i), add_special_tokens=False)[-1] for i in range(NUM_CLASSES)]
print(f"Token IDs for classes 0 and 1: {number_token_ids}")

# LogitsProcessor to constrain the output to only "0" or "1"
class ConstrainToClassTokens(LogitsProcessor):
    def __init__(self, allowed_token_ids: list[int]):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -float("Inf"))
        mask[:, self.allowed_token_ids] = 0
        return scores + mask

logits_processor = LogitsProcessorList([ConstrainToClassTokens(number_token_ids)])

# This is the prompt format for INFERENCE
inference_prompt = "Classify the following email into 'ham' (class 0) or 'spam' (class 1).\n\nEmail:\n{}\n\nClassification:\nThe correct answer is: class "

true_labels = []
predicted_labels = []

# Loop through the validation dataframe with a progress bar
for index, row in tqdm(val_df.iterrows(), total=val_df.shape[0], desc="Evaluating Base Model"):
    true_label = row['label']
    email_text = row['text']

    # Prepare the input for the model
    inputs = tokenizer([inference_prompt.format(email_text)], return_tensors="pt").to("cuda")

    # Generate the prediction
    with torch.no_grad(): # Ensure no gradients are calculated during inference
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=logits_processor,
            do_sample=False
        )
    
    # Decode the single predicted token
    predicted_token = outputs[0, -1]
    decoded_prediction = tokenizer.decode(predicted_token)

    true_labels.append(true_label)
    try:
        # Convert the string prediction '0' or '1' to an integer
        predicted_labels.append(int(decoded_prediction.strip()))
    except ValueError:
        # If the model fails to output a number, record it as a failure
        predicted_labels.append(-1) 

# --- Calculate and Display Metrics ---
print("\n--- Base Model Evaluation Results (Zero-Shot) ---")

accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='binary', zero_division=0)
recall = recall_score(true_labels, predicted_labels, average='binary', zero_division=0)
f1 = f1_score(true_labels, predicted_labels, average='binary', zero_division=0)

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
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', # Using a different color for distinction
            xticklabels=['Predicted Ham', 'Predicted Spam'],
            yticklabels=['Actual Ham', 'Actual Spam'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Base Model (Zero-Shot)')
plt.show()

# ===================================================================
# 6. Clean up
# ===================================================================
print("\nCleaning up resources...")
del model, tokenizer, full_dataset, train_df, val_df
gc.collect()
torch.cuda.empty_cache()
print("Done.")