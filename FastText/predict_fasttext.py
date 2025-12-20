#!/usr/bin/env python3
import numpy as np
import fasttext
import joblib
import csv
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR  = os.path.join(PROJECT_ROOT, "twitter-datasets")
TEST_PATH = os.path.join(DATA_DIR, "test_data.txt")

# or pretrained model path
FASTTEXT_MODEL_PATH = os.path.join(SCRIPT_DIR, "twitter_fasttext.bin")
MODEL_PATH          = os.path.join(SCRIPT_DIR, "mlp_fasttext_model.joblib")
OUT_PATH            = os.path.join(SCRIPT_DIR, "submission_fasttext.csv")

STOPWORDS = {
    "<user>", "<url>", "rt",
    ",", ".", ";", ":", "'",
}


def load_fasttext_model(path):
    print("Loading FastText model from:", path)
    model = fasttext.load_model(path)
    print("FastText dim =", model.get_dimension())
    return model


def tweet_to_vec(text, model):
    tokens = text.strip().split()
    tokens = [t for t in tokens if t not in STOPWORDS]

    dim = model.get_dimension()

    if not tokens:
        return np.zeros(2 * dim, dtype=np.float32)

    vecs = np.array(
        [model.get_word_vector(t) for t in tokens],
        dtype=np.float32,
    )

    mean_vec = vecs.mean(axis=0)
    max_vec  = vecs.max(axis=0)
    return np.concatenate([mean_vec, max_vec], axis=0)


def create_csv_submission(ids, y_pred, name):
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def main():
    print("Loading FastText model and classifier...")
    model = load_fasttext_model(FASTTEXT_MODEL_PATH)
    clf = joblib.load(MODEL_PATH)

    X_list = []
    ids = []

    print("Building FastText test features...")
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            # format: id,text
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                tweet_id = int(parts[0])
                text = parts[1]
            else:
                tweet_id = len(ids) + 1
                text = parts[0]
            ids.append(tweet_id)
            X_list.append(tweet_to_vec(text, model))

    X_test = np.vstack(X_list).astype(np.float32)
    print("Test shape:", X_test.shape)

    y_pred01 = clf.predict(X_test)           # 0/1
    y_pred = np.where(y_pred01 == 0, -1, 1)  # 0 -> -1, 1 -> 1

    print("Writing submission to", OUT_PATH)
    create_csv_submission(ids, y_pred, OUT_PATH)
    print("Done.")


if __name__ == "__main__":
    main()

