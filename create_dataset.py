# pip install datasets transformers torch pandas openai tqdm huggingface_hub accelerate bitsandbytes

# huggingface-cli login

import os
import torch
import pandas as pd
import random
import json
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from openai import OpenAI
from huggingface_hub import HfApi

# --- 1. CONFIGURATION & SETUP ---

# Check for OpenAI API key
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError(
        "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
    )
client = OpenAI()

# **UPDATED**: Using the user-specified list of target languages.
TARGET_LANGUAGES = [
    "Hindi", "German", "French", "Spanish", "Chinese", "Arabic", 
    "Bengali", "Russian", "Portuguese", "Japanese", "Indonesian", 
    "Urdu", "Punjabi", "Javanese", "Turkish", "Korean", "Marathi", 
    "Ukrainian", "Swedish", "Norwegian"
]
# Adding "English" to make it 21 languages
TARGET_LANGUAGES.append("English")


# Parameters
SAMPLES_PER_LANG = 1000
SOURCE_DATASET = "bvk/ENRON-spam"
TRANSLATION_MODEL_ID = "CohereLabs/aya-101"
VERIFICATION_MODEL_ID = "gpt-4o-mini"
HF_REPO_ID = "UserName/spam-multilingual-2k-per-lang" # Your HF username / repo name

# --- 2. LOAD AND PREPARE SOURCE DATASET ---

print("Loading and preparing the source Enron dataset...")
source_ds = load_dataset(SOURCE_DATASET, split="train")

# Separate spam (label=1) and ham (label=0) emails
spam_emails = [item['email'] for item in source_ds if item['label'] == 1]
ham_emails = [item['email'] for item in source_ds if item['label'] == 0]

print(f"Found {len(spam_emails)} spam and {len(ham_emails)} ham emails in the source dataset.")

# --- 3. INITIALIZE TRANSLATION MODEL ---

print(f"Loading the translation model: {TRANSLATION_MODEL_ID}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cpu":
    print("\nWARNING: Running on CPU. Translation will be extremely slow.\n")

tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_ID)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(
    TRANSLATION_MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True,
)

# --- 4. TRANSLATION PIPELINE ---

final_translated_data = []
verification_source_data = []
spam_idx = 0
ham_idx = 0

print("\nStarting the translation/data generation process...")
# **UPDATED**: Looping directly over the list of language names.
for lang_name in tqdm(TARGET_LANGUAGES, desc="Processing Languages"):
    print(f"\n--- Processing: {lang_name} ---")

    # Special handling for English: no translation needed, just copy from source
    if lang_name == "English":
        for _ in tqdm(range(SAMPLES_PER_LANG), desc="Copying English Spam"):
            source_text = spam_emails[spam_idx % len(spam_emails)]
            final_translated_data.append({"label": 1, "text": source_text, "language": "English"})
            verification_source_data.append({"source_text": source_text, "translated_text": source_text, "label": 1, "language": "English"})
            spam_idx += 1
        for _ in tqdm(range(SAMPLES_PER_LANG), desc="Copying English Ham"):
            source_text = ham_emails[ham_idx % len(ham_emails)]
            final_translated_data.append({"label": 0, "text": source_text, "language": "English"})
            verification_source_data.append({"source_text": source_text, "translated_text": source_text, "label": 0, "language": "English"})
            ham_idx += 1
        continue # Move to the next language

    # Translate SPAM messages for other languages
    for _ in tqdm(range(SAMPLES_PER_LANG), desc=f"Spam to {lang_name}"):
        source_text = spam_emails[spam_idx % len(spam_emails)]
        prompt = f"Translate to {lang_name}: {source_text}"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        try:
            outputs = translation_model.generate(**inputs, max_new_tokens=512)
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            final_translated_data.append({"label": 1, "text": translated_text, "language": lang_name})
            verification_source_data.append({"source_text": source_text, "translated_text": translated_text, "label": 1, "language": lang_name})
        except Exception as e:
            print(f"Error translating spam for {lang_name}: {e}")
        spam_idx += 1

    # Translate HAM messages for other languages
    for _ in tqdm(range(SAMPLES_PER_LANG), desc=f"Ham to {lang_name}"):
        source_text = ham_emails[ham_idx % len(ham_emails)]
        prompt = f"Translate to {lang_name}: {source_text}"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        try:
            outputs = translation_model.generate(**inputs, max_new_tokens=512)
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            final_translated_data.append({"label": 0, "text": translated_text, "language": lang_name})
            verification_source_data.append({"source_text": source_text, "translated_text": translated_text, "label": 0, "language": lang_name})
        except Exception as e:
            print(f"Error translating ham for {lang_name}: {e}")
        ham_idx += 1

# --- 5. VERIFICATION PIPELINE ---

print("\nStarting the verification process with GPT-4o-mini...")
verification_results = {}

def verify_translation(source_text, translated_text, target_lang):
    """Uses GPT-4o-mini to verify the quality of a translation."""
    # If the language is English, the translation is perfect by definition
    if target_lang == "English":
        return True

    prompt = f"""
    You are a professional translator and linguistic analyst. You will be given:
    1. `<SOURCE_LANG>` text (English):
    "{source_text}"
    2. Its translation into `<TARGET_LANG>` ({target_lang}):
    "{translated_text}"
    Your task is to deeply evaluate whether the translation is accurate in terms of:
    *Semantic equivalence*: Does the translated text preserve the exact meaning?
    *Pragmatic register*: Is the formality, tone, and politeness level appropriate?
    *Cultural and idiomatic fidelity*: Are idioms, proverbs, and culture-specific references correctly rendered or suitably adapted?
    *Context of spam vs. ham**: Given this is an email text, does the translation maintain any persuasive or deceptive cues characteristic of spam, or the clarity and authenticity of a legitimate (ham) message?
    *Instructions*
    1. Examine each of the criteria above.
    2. Decide whether the translation is correct overall.
    3. *Output only* either *Yes* (if it meets all criteria) or *No* (if any significant mistranslation or register slip is present).
    4. Do *not* include any additional explanation or commentary. Only the single word.
    Your response:
    """
    try:
        response = client.chat.completions.create(
            model=VERIFICATION_MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        answer = response.choices[0].message.content.strip().lower()
        return "yes" in answer
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return False

verification_df = pd.DataFrame(verification_source_data)

for lang_name in tqdm(TARGET_LANGUAGES, desc="Verifying Languages"):
    lang_results = {}
    lang_data = verification_df[verification_df['language'] == lang_name]
    
    spam_samples = lang_data[lang_data['label'] == 1].sample(n=30, random_state=42)
    ham_samples = lang_data[lang_data['label'] == 0].sample(n=30, random_state=42)
    
    spam_good_count = sum(1 for _, row in tqdm(spam_samples.iterrows(), total=30, desc=f"Verifying {lang_name} Spam") if verify_translation(row['source_text'], row['translated_text'], lang_name))
    ham_good_count = sum(1 for _, row in tqdm(ham_samples.iterrows(), total=30, desc=f"Verifying {lang_name} Ham") if verify_translation(row['source_text'], row['translated_text'], lang_name))

    lang_results['spam_quality'] = "good" if (spam_good_count / 30) > 0.5 else "bad"
    lang_results['ham_quality'] = "good" if (ham_good_count / 30) > 0.5 else "bad"
    lang_results['spam_pass_rate'] = f"{spam_good_count}/30"
    lang_results['ham_pass_rate'] = f"{ham_good_count}/30"
    
    verification_results[lang_name] = lang_results

print("\n--- Translation Verification Results ---")
print(json.dumps(verification_results, indent=2))
print("----------------------------------------\n")

# --- 6. FINALIZE AND UPLOAD DATASET ---

print("Finalizing dataset and uploading to Hugging Face Hub...")

final_df = pd.DataFrame(final_translated_data)
final_dataset = Dataset.from_pandas(final_df)

print("\nDataset Information:")
print(final_dataset)
print("\nColumns in the final dataset:", final_dataset.column_names)
print("\nExample entry:")
print(random.choice(final_dataset))

try:
    final_dataset.push_to_hub(HF_REPO_ID, private=False)
    print(f"\nSuccessfully uploaded dataset to: https://huggingface.co/datasets/{HF_REPO_ID}")
except Exception as e:
    print(f"\nFailed to upload dataset. Error: {e}")
    print("Saving dataset locally to 'spam_multilingual_dataset.csv' instead.")
    final_df.to_csv("spam_multilingual_dataset.csv", index=False)