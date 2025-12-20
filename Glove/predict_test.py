#!/usr/bin/env python3
import numpy as np
import pickle
import joblib
import csv   # needed for create_csv_submission
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

VOCAB_PATH = os.path.join(SCRIPT_DIR, "vocab.pkl")
EMB_PATH   = os.path.join(SCRIPT_DIR, "embeddings.npy")
MODEL_PATH = os.path.join(SCRIPT_DIR, "mlp_model.joblib")

DATA_DIR = os.path.join(PROJECT_ROOT, "twitter-datasets")
TEST_PATH  = os.path.join(DATA_DIR, "test_data.txt")
OUT_PATH   = os.path.join(SCRIPT_DIR, "submission_glove.csv")

STOPWORDS = {
    "<user>", "<url>", "rt",
    ",", ".", ";", ":", "'",
}


def load_vocab_and_embeddings():
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    emb = np.load(EMB_PATH)
    return vocab, emb


def tweet_to_vec(text, vocab, emb):
    tokens = text.strip().split()

    tokens = [t for t in tokens if t not in STOPWORDS]  # remove stopwords

    idxs = [vocab[t] for t in tokens if t in vocab]
    if not idxs:
        return np.zeros(2 * emb.shape[1], dtype=np.float32)

    vecs = emb[idxs]              # (n_tokens, D)

    # concatenate the mean and max vectors as the feature vector for each tweet
    mean_vec = vecs.mean(axis=0)
    max_vec  = vecs.max(axis=0)
    return np.concatenate([mean_vec, max_vec], axis=0)

def create_csv_submission(ids, y_pred, name):
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def main():
    print("Loading vocab, embeddings, and model...")
    vocab, emb = load_vocab_and_embeddings()
    clf = joblib.load(MODEL_PATH)

    # Build test features
    X_list = []
    ids = []

    with open(TEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            # test data format is "id,text", need to split by the first comma
            parts = line.strip().split(',', 1)  # split by the first comma
            if len(parts) == 2:
                tweet_id = int(parts[0])
                text = parts[1]
            else:
                # if there is no comma, use the line number as the ID (for backward compatibility)
                tweet_id = len(ids) + 1
                text = parts[0]
            ids.append(tweet_id)
            X_list.append(tweet_to_vec(text, vocab, emb))

    X_test = np.vstack(X_list).astype(np.float32)
    print("Test shape:", X_test.shape)

    # Predict: assume classifier outputs 0/1, convert 0 -> -1
    y_pred01 = clf.predict(X_test)
    y_pred = np.where(y_pred01 == 0, -1, 1)

    # Use your official helper to write CSV
    print("Writing submission to", OUT_PATH)
    create_csv_submission(ids, y_pred, OUT_PATH)
    print("Done.")


if __name__ == "__main__":
    main()


