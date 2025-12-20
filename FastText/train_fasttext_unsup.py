# train_fasttext_unsup.py
import fasttext
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "twitter-datasets")
corpus_path = os.path.join(DATA_DIR, "all_train.txt")

model = fasttext.train_unsupervised(
    input=corpus_path,
    model="skipgram",   # or "cbow"
    dim=75,            # embedding dimension
    epoch=20,            
    lr=0.05,
    minn=3,
    maxn=6,
)

model_path = os.path.join(SCRIPT_DIR, "twitter_fasttext.bin")
model.save_model(model_path)
print(f"Saved model to {model_path}")