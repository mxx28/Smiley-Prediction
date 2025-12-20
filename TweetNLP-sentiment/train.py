import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["WANDB_DISABLED"] = "true"

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import torch
from torch.nn import CrossEntropyLoss
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import load_file


pos_data = load_file("twitter-datasets/train_pos_full.txt", 2)    # positive label 2 the same as that in the model
neg_data = load_file("twitter-datasets/train_neg_full.txt", 0)    # negative label 0

df = pd.DataFrame(pos_data + neg_data)

dataset = Dataset.from_pandas(df)

# Tokenizer

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.train_test_split(test_size=0.1, seed=42)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True)

# Training args
args = TrainingArguments(
    output_dir="model_out",

    eval_strategy="epoch",         
    save_strategy="epoch",         

    metric_for_best_model="eval_loss",  
    greater_is_better=False,            

    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,

    gradient_accumulation_steps=2,
    num_train_epochs=1,
    weight_decay=0.01,

    fp16=True,
    load_best_model_at_end=True,

    logging_steps=100,
)


# Trainer
history = {"train_loss": [], "eval_loss": []}

class SaveMetricsCallback(EarlyStoppingCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            history["train_loss"].append(logs["loss"])
        if "eval_loss" in logs:
            history["eval_loss"].append(logs["eval_loss"])

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    callbacks=[SaveMetricsCallback(early_stopping_patience=2)]
)

trainer.train()
trainer.save_model("model_out")
tokenizer.save_pretrained("model_out")

# Plot curves
plt.figure(figsize=(10,5))
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["eval_loss"], label="Eval Loss")
plt.legend()
plt.title("Training Curves")
plt.savefig("model_out/training_curves.png")
print("Saved training_curves.png")
