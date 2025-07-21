# pip install unsloth
# pip install "trl<0.9.0" accelerate
# pip install scikit-learn matplotlib seaborn
import os
import torch
import pandas as pd
import gc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from unsloth import FastLanguageModel
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc

# --- 1. DATA LOADING AND PREPARATION (SAME AS BEFORE FOR FAIR COMPARISON) ---

print("Loading dataset: hamza-khan/spam-multilingual-2k-per-lang")
dataset = load_dataset("hamza-khan/spam-multilingual-2k-per-lang", split="train")
df = dataset.to_pandas()

# We only need the validation set for this script.
# We recreate the *exact same split* using the same random_state.
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)
print(f"Loaded and split data. Using validation set of size: {len(val_df)}")
print("-" * 30)


# --- 2. 16-BIT MODEL INITIALIZATION (NO 4-BIT QUANTIZATION) ---

# Model parameters
max_seq_length = 2048
model_name = "unsloth/Qwen3-4B-Base"

# ** KEY CHANGE **: Load the model in 16-bit precision
print(f"Loading base model '{model_name}' in 16-bit precision...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=torch.float16,  # Explicitly use float16
    load_in_4bit=False,   # The crucial change
)

# Get token IDs for "0" and "1" which we'll need for inference
number_token_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(2)]
print(f"Target token IDs for classification: '0' -> {number_token_ids[0]}, '1' -> {number_token_ids[1]}")
print("-" * 30)

# --- 3. K-SHOT INFERENCE AND EVALUATION (NO FINE-TUNING) ---

# Set model to evaluation mode. No training is performed.
model.eval()
print("Model set for K-Shot (zero-shot) inference. No fine-tuning has been performed.")

# Inference prompt template (without the label part)
def inference_prompt_template(text: str) -> str:
    return f"""Here is an email text:
{text}
Classify this email into one of the following:
class 0: Ham (Not Spam)
class 1: Spam
SOLUTION
The correct answer is: class """

# Prepare validation data for inference
# Sort by length to speed up inference by reducing padding
val_inference_df = val_df[['text', 'label']].copy()
val_inference_df['token_length'] = val_inference_df['text'].apply(
    lambda x: len(tokenizer.encode(inference_prompt_template(x)))
)
val_sorted = val_inference_df.sort_values('token_length').reset_index(drop=True)

# Inference loop
batch_size = 16 # You might need to lower this if you run out of VRAM with the 16-bit model
device = model.device
all_preds, all_true, all_probs = [], [], []

with torch.inference_mode():
    for start in tqdm(range(0, len(val_sorted), batch_size), desc="Evaluating 16-bit Baseline"):
        batch_df = val_sorted.iloc[start:start+batch_size]
        prompts = [inference_prompt_template(t) for t in batch_df['text']]
        
        inputs = tokenizer(
            prompts, return_tensors='pt', padding=True, truncation=True, max_length=max_seq_length
        ).to(device)

        # Get logits from the model
        logits = model(**inputs).logits.cpu()
        
        # Find the logits of the very last token for each sequence
        last_token_indices = inputs['attention_mask'].sum(dim=1) - 1
        last_logits = logits[torch.arange(len(last_token_indices)), last_token_indices, :]
        
        # Extract logits only for our target tokens "0" and "1" from the full vocabulary
        class_logits = last_logits[:, number_token_ids]
        
        # Get probabilities and predictions
        class_probs = F.softmax(class_logits, dim=-1)
        preds = class_probs.argmax(dim=-1).numpy()
        probs_class_1 = class_probs[:, 1].numpy()

        true_labels = batch_df['label'].to_numpy()

        all_preds.extend(preds)
        all_true.extend(true_labels)
        all_probs.extend(probs_class_1)

# --- 4. CALCULATE AND DISPLAY RESULTS ---

# Calculate metrics
accuracy = accuracy_score(all_true, all_preds)
precision = precision_score(all_true, all_preds, zero_division=0)
recall = recall_score(all_true, all_preds, zero_division=0)
f1 = f1_score(all_true, all_preds, zero_division=0)
auc_roc = roc_auc_score(all_true, all_probs)

print("\n" + "="*50)
print("--- 16-bit Baseline K-Shot Inference Results ---")
print("="*50)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"AUC-ROC:   {auc_roc:.4f}")
print("-" * 50)

# Confusion Matrix
cm = confusion_matrix(all_true, all_preds)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greys', cbar=False, xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (16-bit Baseline)')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(all_true, all_probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='grey', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (16-bit Baseline)')
plt.legend(loc="lower right")
plt.show()

# Clean up memory
del model, tokenizer
gc.collect()
torch.cuda.empty_cache()