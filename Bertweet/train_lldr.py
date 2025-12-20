import os
import torch
import numpy as np
import re
import html
from datasets import load_dataset, concatenate_datasets, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score

# --- CONFIGURATION ---
MODEL_NAME = "vinai/bertweet-base"
MAX_LEN = 128
BATCH_SIZE = 128
ACCUMULATION_STEPS = 1
LEARNING_RATE = 2e-5  # Base Learning Rate
LLRD_DECAY = 0.95     # Layer-wise decay rate (lower layers get smaller LR)

BASE_PATH = "../"
DATA_CACHE_DIR = f"{BASE_PATH}/processed_data_lldr"
OUTPUT_DIR = f"{BASE_PATH}/checkpoints_bertweet_lldr"

def clean_text(text):
    # Unescape HTML entities to standard characters
    text = html.unescape(str(text))
    
    # BERTweet normalization: specific placeholders required by the pre-trained model
    text = re.sub(r'http\S+', 'HTTPURL', text)
    text = re.sub(r'www\.\S+', 'HTTPURL', text)
    text = re.sub(r'@\w+', '@USER', text)
    
    # Remove excess whitespace
    text = " ".join(text.split())
    return text

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return {"accuracy": acc, "f1": f1}

def get_optimizer_grouped_parameters(model, learning_rate, weight_decay, layer_decay=0.95):
    """
    Implements Layer-wise Learning Rate Decay (LLRD).
    - Top layers (Classifier) get the base learning_rate.
    - Lower layers (Encoder -> Embeddings) decay geometrically: lr * (decay ^ depth).
    """
    opt_parameters = []
    named_parameters = list(model.named_parameters())

    # Exclude Bias and LayerNorm parameters from weight decay
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # --- Group 1: Classifier/Head (Highest LR) ---
    # These parameters are specific to the task and need the most adaptation.
    opt_parameters.append({
        "params": [p for n, p in named_parameters if ("classifier" in n or "pooler" in n) and not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
        "lr": learning_rate
    })
    opt_parameters.append({
        "params": [p for n, p in named_parameters if ("classifier" in n or "pooler" in n) and any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
        "lr": learning_rate
    })

    # --- Group 2: Transformer Encoder Layers (Decaying LR) ---
    # BERTweet base has 12 layers (0-11). We iterate from top (11) down to bottom (0).
    for layer_i in range(11, -1, -1):
        # Calculate LR for this specific layer depth
        layer_lr = learning_rate * (layer_decay ** (12 - layer_i))

        opt_parameters.append({
            "params": [p for n, p in named_parameters if f"encoder.layer.{layer_i}." in n and not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
            "lr": layer_lr
        })
        opt_parameters.append({
            "params": [p for n, p in named_parameters if f"encoder.layer.{layer_i}." in n and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": layer_lr
        })

    # --- Group 3: Embeddings (Lowest LR) ---
    # These are the most fundamental features and should change the least.
    embedding_lr = learning_rate * (layer_decay ** 13)
    opt_parameters.append({
        "params": [p for n, p in named_parameters if "embeddings" in n and not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
        "lr": embedding_lr
    })
    opt_parameters.append({
        "params": [p for n, p in named_parameters if "embeddings" in n and any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
        "lr": embedding_lr
    })

    # Validation: Ensure all parameters are assigned to a group
    param_set = set()
    for group in opt_parameters:
        for p in group['params']:
            param_set.add(p)
    assert len(param_set) == len(list(model.parameters())), "LLRD grouping missed some parameters!"

    return opt_parameters

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, normalization=True)

    # --- Data Loading ---
    if os.path.exists(DATA_CACHE_DIR):
        print("Loading cached data...")
        tokenized_datasets = load_from_disk(DATA_CACHE_DIR)
    else:
        print("Processing data from scratch (FULL DATA)...")
        data_files = {"pos": f"{BASE_PATH}/twitter-datasets/train_pos_full.txt", "neg": f"{BASE_PATH}/twitter-datasets/train_neg_full.txt"}
        dataset = load_dataset("text", data_files=data_files)

        # Apply cleaning
        dataset = dataset.map(lambda x: {"text": clean_text(x["text"])}, num_proc=4)

        pos_dataset = dataset["pos"].map(lambda x: {"label": 1})
        neg_dataset = dataset["neg"].map(lambda x: {"label": 0})

        # Merge and shuffle (No de-duplication performed here to keep full distribution)
        full_dataset = concatenate_datasets([pos_dataset, neg_dataset]).shuffle(seed=42)

        split_dataset = full_dataset.train_test_split(test_size=0.1)

        tokenized_datasets = split_dataset.map(
            lambda x: tokenizer(x["text"], truncation=True, max_length=MAX_LEN),
            batched=True, num_proc=4
        )
        tokenized_datasets.save_to_disk(DATA_CACHE_DIR)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        use_safetensors=True
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",
        eval_steps=2000,   
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE*2,
        gradient_accumulation_steps=ACCUMULATION_STEPS,
        num_train_epochs=5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        label_smoothing_factor=0.1,
        bf16=True, 
        logging_steps=500,
        report_to="none",
        dataloader_num_workers=4,
    )

    # Initialize AdamW with the custom LLRD parameter groups
    optimizer = torch.optim.AdamW(
        get_optimizer_grouped_parameters(model, LEARNING_RATE, 0.01, LLRD_DECAY),
        lr=LEARNING_RATE
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None) # Inject the custom optimizer here
    )

    print("Starting training with LLRD and Label Smoothing...")
    trainer.train(resume_from_checkpoint=True)

    print("Saving best model...")
    trainer.save_model(f"{BASE_PATH}/best_bertweet_lldr")
    tokenizer.save_pretrained(f"{BASE_PATH}/best_bertweet_final")

if __name__ == "__main__":
    main()