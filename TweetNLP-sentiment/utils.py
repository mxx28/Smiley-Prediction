
import pandas as pd
import re, html

# Clean tweet function
def clean_tweet(text):
    text = html.unescape(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Load the cleaned data
def load_file(path, label):
    rows = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            cleaned = clean_tweet(line.strip())
            if cleaned:
                rows.append({"text": cleaned, "label": label})
    return rows


def read_test(path):
    ids, texts = [], []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            id_str, text = line.split(",", 1)
            ids.append(int(id_str))
            texts.append(clean_tweet(text)) # Clean the tweet text
    return ids, texts