#!/usr/bin/env python3
import numpy as np
import fasttext
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "twitter-datasets")
POS_TRAIN_PATH = os.path.join(DATA_DIR, "train_pos_full.txt")
NEG_TRAIN_PATH = os.path.join(DATA_DIR, "train_neg_full.txt")

FASTTEXT_MODEL_PATH = os.path.join(SCRIPT_DIR, "twitter_fasttext.bin")
# or pretrained model path

OUT_X_PATH = os.path.join(SCRIPT_DIR, "train_features_fasttext.npy")
OUT_Y_PATH = os.path.join(SCRIPT_DIR, "train_labels_fasttext.npy")

STOPWORDS = {
    "<user>", "<url>", "rt",
    ",", ".", ";", ":", "'",
}


def load_fasttext_model(path):
    print("Loading FastText model from:", path)
    model = fasttext.load_model(path)
    print("FastText dim =", model.get_dimension())
    return model


def tweet_to_vec(line, model):
    """
    Convert a tweet into a feature vector using FastText:
    - Remove STOPWORDS
    - For each token, get the FastText vector
    - mean + max pooling concatenation → 2 * dim
    """
    tokens = line.strip().split()
    tokens = [t for t in tokens if t not in STOPWORDS]

    dim = model.get_dimension()

    if not tokens:
        return np.zeros(2 * dim, dtype=np.float32)

    # FastText can also provide vectors for OOV tokens, just call it directly
    vecs = np.array(
        [model.get_word_vector(t) for t in tokens],
        dtype=np.float32,
    )  # shape: (n_tokens, dim)

    mean_vec = vecs.mean(axis=0)
    max_vec  = vecs.max(axis=0)
    return np.concatenate([mean_vec, max_vec], axis=0)  # (2*dim,)


def build_features():
    model = load_fasttext_model(FASTTEXT_MODEL_PATH)
    dim = model.get_dimension()
    print(f"Using FastText dim = {dim}, tweet feature dim = {2 * dim}")

    X_list = []
    y_list = []

    # Positive tweets
    print("Processing positive tweets...")
    with open(POS_TRAIN_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            vec = tweet_to_vec(line, model)
            X_list.append(vec)
            y_list.append(1)
            if (i + 1) % 10000 == 0:
                print(f"  processed {i+1} positive tweets")

    # Negative tweets
    print("Processing negative tweets...")
    with open(NEG_TRAIN_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            vec = tweet_to_vec(line, model)
            X_list.append(vec)
            y_list.append(0)
            if (i + 1) % 10000 == 0:
                print(f"  processed {i+1} negative tweets")

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    print("Saving features:", X.shape, "labels:", y.shape)
    np.save(OUT_X_PATH, X)
    np.save(OUT_Y_PATH, y)
    print("Done. Saved to", OUT_X_PATH, "and", OUT_Y_PATH)


if __name__ == "__main__":
    build_features()

