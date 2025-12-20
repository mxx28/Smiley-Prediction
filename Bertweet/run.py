import os
import torch
import csv
import numpy as np
from tqdm import tqdm
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- CONFIGURATION ---
BASE_PATH = "../"
HF_REPO_ID = "RangerX/Bertweet_large"

TEST_FILE = f"{BASE_PATH}/twitter-datasets/test_data.txt"
OUTPUT_FILE = f"{BASE_PATH}/submission_large.csv"
CACHE_DIR = f"{BASE_PATH}/tokenized_test_data"
MAX_LEN = 128
BATCH_SIZE = 64

def parse_test_file(file_path):
    """Reads the test file and separates IDs from Text."""
    print(f"Parsing {file_path}...")
    ids, texts = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(',', 1)
            if len(parts) < 2:
                parts = line.split(' ', 1)
            tweet_id = parts[0].strip()
            tweet_text = parts[1].strip() if len(parts) > 1 else ""
            ids.append(tweet_id)
            texts.append(tweet_text)
    return {"id": ids, "text": texts}

def prepare_test_data(tokenizer):
    """Loads test data, tokenizes it, and caches it to disk."""
    if os.path.exists(CACHE_DIR):
        print(f"Loading cached test data from {CACHE_DIR}...")
        return load_from_disk(CACHE_DIR)

    data_dict = parse_test_file(TEST_FILE)
    dataset = Dataset.from_dict(data_dict)

    print("Tokenizing test data...")
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LEN)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    print(f"Saving tokenized test data to {CACHE_DIR}...")
    tokenized_dataset.save_to_disk(CACHE_DIR)
    return tokenized_dataset

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Tokenizer from Hugging Face
    print(f"Loading tokenizer from {HF_REPO_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID)
    test_dataset = prepare_test_data(tokenizer)

    # 2. Load Model from Hugging Face
    print(f"Loading model from {HF_REPO_ID}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        HF_REPO_ID,
        num_labels=2,
        use_safetensors=True
    )
    model.to(device)
    model.eval()

    # 3. Inference Loop
    print("Running inference...")
    predictions = []
    total_len = len(test_dataset)

    for i in tqdm(range(0, total_len, BATCH_SIZE)):
        batch = test_dataset[i : i + BATCH_SIZE]
        input_ids = torch.tensor(batch['input_ids']).to(device)
        attention_mask = torch.tensor(batch['attention_mask']).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

        # Map 0 -> -1 (Negative), 1 -> 1 (Positive)
        mapped_preds = [1 if p == 1 else -1 for p in preds]
        predictions.extend(mapped_preds)

    # 4. Save Submission
    print(f"Writing {len(predictions)} predictions to {OUTPUT_FILE}...")
    ids = test_dataset['id']
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Prediction"])
        for pid, pred in zip(ids, predictions):
            writer.writerow([pid, pred])

    print("Done! You are ready to submit.")

if __name__ == "__main__":
    main()