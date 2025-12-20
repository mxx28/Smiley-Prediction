#!/usr/bin/env python3
import numpy as np
import pickle
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Paths (change to *_train.txt if you want the small version first)
DATA_DIR = os.path.join(PROJECT_ROOT, "twitter-datasets")
POS_TRAIN_PATH = os.path.join(DATA_DIR, "train_pos_full.txt")
NEG_TRAIN_PATH = os.path.join(DATA_DIR, "train_neg_full.txt")

VOCAB_PATH = os.path.join(SCRIPT_DIR, "vocab.pkl")
EMB_PATH = os.path.join(SCRIPT_DIR, "embeddings.npy")

OUT_X_PATH = os.path.join(SCRIPT_DIR, "train_features.npy")
OUT_Y_PATH = os.path.join(SCRIPT_DIR, "train_labels.npy")

STOPWORDS = {
    "<user>", "<url>", "rt",
    ",", ".", ";", ":", "'",
}


def load_vocab_and_embeddings(vocab_path, emb_path):
    """Load vocab.pkl and embeddings.npy."""
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)  # dict: word -> index
    emb = np.load(emb_path)     # shape: (V, D)
    return vocab, emb


def tweet_to_vec(line, vocab, emb):
    """
    Convert one tweet line into a feature vector:
    - split on whitespace
    - keep tokens that are in the vocab
    - average their embeddings
    - if no token is in vocab, return a zero vector
    """
    tokens = line.strip().split()

    tokens = [t for t in tokens if t not in STOPWORDS]  # remove stopwords

    idxs = [vocab[t] for t in tokens if t in vocab]

    if not idxs:
        return np.zeros(2 * emb.shape[1], dtype=np.float32)

    vecs = emb[idxs]              # (n_tokens, D)

    # concatenate the mean and max vectors as the feature vector for each tweet
    mean_vec = vecs.mean(axis=0)
    max_vec  = vecs.max(axis=0)
    return np.concatenate([mean_vec, max_vec], axis=0)

def build_features():
    print("Loading vocab and embeddings...")
    vocab, emb = load_vocab_and_embeddings(VOCAB_PATH, EMB_PATH)
    dim = emb.shape[1]
    print(f"Vocab size = {len(vocab)}, embedding dim = {dim}")

    X_list = []
    y_list = []

    # Positive tweets
    print("Processing positive tweets...")
    with open(POS_TRAIN_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            vec = tweet_to_vec(line, vocab, emb)
            X_list.append(vec)
            y_list.append(1)   # label for positive
            if (i + 1) % 10000 == 0:
                print(f"  processed {i+1} positive tweets")

    # Negative tweets
    print("Processing negative tweets...")
    with open(NEG_TRAIN_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            vec = tweet_to_vec(line, vocab, emb)
            X_list.append(vec)
            y_list.append(0)   # label for negative (change to -1 if you prefer)
            if (i + 1) % 10000 == 0:
                print(f"  processed {i+1} negative tweets")

    X = np.vstack(X_list).astype(np.float32)  # (N, D)
    y = np.array(y_list, dtype=np.int64)      # (N,)

    print("Saving features:", X.shape, "labels:", y.shape)
    np.save(OUT_X_PATH, X)
    np.save(OUT_Y_PATH, y)
    print("Done. Saved to", OUT_X_PATH, "and", OUT_Y_PATH)


if __name__ == "__main__":
    build_features()
