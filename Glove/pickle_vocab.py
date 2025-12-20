#!/usr/bin/env python3
import pickle
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    vocab = dict()
    vocab_cut_path = os.path.join(SCRIPT_DIR, "vocab_cut.txt")
    with open(vocab_cut_path) as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    vocab_pkl_path = os.path.join(SCRIPT_DIR, "vocab.pkl")
    with open(vocab_pkl_path, "wb") as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()

# example:
# {
#     "the": 0,
#     "movie": 1,
#     "happy": 2
# }
