import csv
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import re, html

from utils import read_test

MODEL_DIR = "model_out"
TEST_PATH = "twitter-datasets/test_data.txt"
OUT_PATH = "submission.csv"
BATCH_SIZE = 64


tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


ids, texts = read_test(TEST_PATH)
results = []

for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Predicting"):
    batch_texts = texts[i:i+BATCH_SIZE]
    inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True,
                       padding=True, max_length=128).to(device)

    # with torch.no_grad():
    #     logits = model(**inputs).logits
    #     preds = torch.argmax(logits, dim=-1)

    # batch_labels = [1 if p.item() == 1 else -1 for p in preds]

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)  # → [neg, neu, pos]

    # pos prob > neg prob → 1, otherwise -1
        preds = (probs[:, 2] > probs[:, 0]).long()

    batch_labels = [1 if p.item() == 1 else -1 for p in preds]


    batch_ids = ids[i:i+BATCH_SIZE]
    results.extend(zip(batch_ids, batch_labels))

with open(OUT_PATH, "w", newline="", encoding="utf8") as f:
    writer = csv.writer(f)
    writer.writerow(["Id", "Prediction"])
    for id_, lab in results:
        writer.writerow([id_, lab])

print("Saved to", OUT_PATH)
