#!/usr/bin/env python3
from scipy.sparse import coo_matrix
import numpy as np
import pickle
import os

WINDOW = 5   # left/right window size

META_STOPWORDS = {"<user>", "<url>", "rt"}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    vocab_path = os.path.join(SCRIPT_DIR, "vocab.pkl")
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    data, row, col = [], [], []
    counter = 1

    project_root = os.path.dirname(SCRIPT_DIR)
    data_dir = os.path.join(project_root, "twitter-datasets")
    
    for fn in [os.path.join(data_dir, "train_pos_full.txt"),
               os.path.join(data_dir, "train_neg_full.txt")]:
        with open(fn) as f:
            for line in f:
                tokens = [t for t in line.strip().split() if t not in META_STOPWORDS]
                tokens = [vocab.get(t, -1) for t in tokens]
                tokens = [t for t in tokens if t >= 0]

                # sliding window co-occurrence
                for i, wi in enumerate(tokens):
                    left  = max(0, i - WINDOW)
                    right = min(len(tokens), i + WINDOW + 1)
                    for j in range(left, right):
                        if i == j:
                            continue
                        wj = tokens[j]
                        dist = abs(j - i)
                        weight = 1.0 / dist   # closer words get higher weight

                        row.append(wi)
                        col.append(wj)
                        data.append(weight)

                if counter % 10000 == 0:
                    print(counter)
                counter += 1

    cooc = coo_matrix((data, (row, col)))
    print("summing duplicates (this can take a while)")
    cooc.sum_duplicates()
    cooc_path = os.path.join(SCRIPT_DIR, "cooc.pkl")
    with open(cooc_path, "wb") as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
