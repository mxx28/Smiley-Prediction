# build_all_train_txt.py
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "twitter-datasets")
pos_path = os.path.join(DATA_DIR, "train_pos_full.txt")
neg_path = os.path.join(DATA_DIR, "train_neg_full.txt")

out_path = os.path.join(DATA_DIR, "all_train.txt")

with open(out_path, "w", encoding="utf-8") as out_f:
    for path in [pos_path, neg_path]:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                out_f.write(line)

print("Saved merged corpus to", out_path)

