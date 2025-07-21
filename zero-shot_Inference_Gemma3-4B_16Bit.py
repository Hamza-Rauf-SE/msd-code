# pip install unsloth
# pip install "trl<0.9.0" accelerate
# pip install scikit-learn matplotlib seaborn
# 1.Imports
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
from transformers import LogitsProcessor, LogitsProcessorList
from typing import List
from tqdm import tqdm

# ===================================================================
# 2. Configuration
# ===================================================================
model_name = "unsloth/gemma-3-4b-pt"
NUM_CLASSES = 2
max_seq_length = 2048

# ===================================================================
# 3. Load 16-bit Model and Processor/Tokenizer
# ===================================================================
print("Loading BASE model and tokenizer in 16-BIT precision...")
# ** THE KEY CHANGE IS HERE **
model, processor = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=torch.float16,  # Explicitly use 16-bit
    load_in_4bit=False,   # Do NOT use 4-bit quantization
)
text_tokenizer = processor.tokenizer
if text_tokenizer is None:
    raise RuntimeError("Text tokenizer could not be loaded from the processor.")

if text_tokenizer.pad_token is None:
    text_tokenizer.pad_token = text_tokenizer.eos_token
print("Base model loaded successfully.")

# ===================================================================
# 4. Load and Prepare the SAME Validation Data
# ===================================================================
print("Loading and preparing the multilingual spam dataset...")
full_dataset = load_dataset("hamza-khan/spam-multilingual-2k-per-lang", split="train")
df = full_dataset.to_pandas()

# Create the EXACT SAME training and validation splits to ensure a fair comparison
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42, # Using the same random state is crucial
    stratify=df['label']
)
print(f"Using the exact same validation set of size: {len(val_df)}")
print("-" * 50)


# ===================================================================
# 5. Run Full Evaluation on Validation Set
# ===================================================================
print("Starting K-Shot (baseline) evaluation on the validation set...")

FastLanguageModel.for_inference(model)
model.eval()

# Get the token IDs for "0" and "1" to constrain the output
number_token_ids = [text_tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(NUM_CLASSES)]

# Use a logits processor to force the model to choose between "0" or "1"
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
# NOTE: The 16-bit model uses more VRAM. You may need to lower the batch size if you encounter memory errors.
batch_size = 4
device = "cuda"

with torch.inference_mode():
    for i in tqdm(range(0, len(val_df), batch_size), desc="Evaluating 16-bit Baseline"):
        batch_df = val_df.iloc[i:i+batch_size]
        prompts = [inference_prompt.format(text) for text in batch_df['text']]

        inputs = text_tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length-1
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            pad_token_id=text_tokenizer.eos_token_id,
            logits_processor=logits_processor
        )
        
        generated_tokens = outputs[:, -1]
        decoded_preds = text_tokenizer.batch_decode(generated_tokens)
        
        try:
            int_preds = [int(p.strip()) for p in decoded_preds]
            all_preds.extend(int_preds)
        except ValueError:
            # Handle cases where the model might output something unexpected despite the processor
            # This is a fallback, but the logits processor should prevent this.
            for p in decoded_preds:
                try:
                    all_preds.append(int(p.strip()))
                except:
                    all_preds.append(1 - all_true[len(all_preds)]) # Assign incorrect label on failure


# ===================================================================
# 6. Calculate and Display Metrics
# ===================================================================
# Ensure we have predictions for all true labels
if len(all_preds) != len(all_true):
     # This can happen if the last batch fails. Truncate for now.
    print(f"Warning: Mismatch in prediction length. Have {len(all_preds)} preds for {len(all_true)} labels.")
    all_true = all_true[:len(all_preds)]


accuracy = accuracy_score(all_true, all_preds)
precision = precision_score(all_true, all_preds, zero_division=0)
recall = recall_score(all_true, all_preds, zero_division=0)
f1 = f1_score(all_true, all_preds, zero_division=0)

print("\n" + "="*50)
print("--- Gemma-3 16-bit Baseline K-Shot Inference Results ---")
print("="*50)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("-" * 50)

# Confusion Matrix
cm = confusion_matrix(all_true, all_preds)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greys', cbar=False, xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Gemma-3-4B 16-bit Baseline)')
plt.show()

# Clean up memory
del model, processor, text_tokenizer, df, full_dataset
gc.collect()
torch.cuda.empty_cache()