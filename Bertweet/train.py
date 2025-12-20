import os
import torch
import numpy as np
from datasets import load_dataset, DatasetDict, concatenate_datasets, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score

# --- CONFIGURATION ---
MODEL_NAME = "vinai/bertweet-large"
MAX_LEN = 128
BATCH_SIZE = 128
ACCUMULATION_STEPS = 1

# Training Control
MAX_TRAIN_SAMPLES = None  
MAX_VAL_SAMPLES = None    

# Evaluation Control
EVAL_STRATEGY = "steps"
EVAL_STEPS = 4000         # Evaluate every X steps
SAVE_STEPS = 4000         # Save checkpoint every X steps
LOGGING_STEPS = 500       

BASE_PATH = "../"
DATA_CACHE_DIR = f"{BASE_PATH}/tokenized_data"
OUTPUT_DIR = f"{BASE_PATH}/checkpoints_bertweet_large"
FINAL_SAVE_PATH = f"{BASE_PATH}/best_bertweet_model_large"

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return {"accuracy": acc, "f1": f1}

def prepare_data(tokenizer):
    """Loads, processes, and saves data if not already cached."""

    # Check if we already have processed data on disk
    if os.path.exists(DATA_CACHE_DIR):
        print(f"Loading cached dataset from {DATA_CACHE_DIR}...")
        tokenized_datasets = load_from_disk(DATA_CACHE_DIR)
        return tokenized_datasets

    # --- If not cached, process from scratch ---
    print("No cache found. Processing data from scratch...")

    data_files = {
        "pos": "/content/drive/MyDrive/CS433/twitter-datasets/train_pos_full.txt",
        "neg": "/content/drive/MyDrive/CS433/twitter-datasets/train_neg_full.txt"
    }

    dataset = load_dataset("text", data_files=data_files)

    print("Adding labels...")
    pos_dataset = dataset["pos"].map(lambda x: {"label": 1})
    neg_dataset = dataset["neg"].map(lambda x: {"label": 0})

    print("Merging and shuffling...")
    # Shuffle with a fixed seed for reproducibility
    full_dataset = concatenate_datasets([pos_dataset, neg_dataset]).shuffle(seed=42)

    # Split Train/Validation
    print("Splitting Train/Val...")
    split_dataset = full_dataset.train_test_split(test_size=0.1)
    datasets = DatasetDict({
        'train': split_dataset['train'],
        'validation': split_dataset['test']
    })

    # Tokenize
    print("Tokenizing (this takes time once)...")
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LEN)

    tokenized_datasets = datasets.map(tokenize_function, batched=True)

    # Save to disk for future runs
    print(f"Saving tokenized dataset to {DATA_CACHE_DIR}...")
    tokenized_datasets.save_to_disk(DATA_CACHE_DIR)

    return tokenized_datasets

def main():
    if torch.cuda.is_available():
        print(f"Detected GPU: {torch.cuda.get_device_name(0)}")

    # 1. Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, normalization=True)

    # 2. Get Data (Cached or Fresh)
    tokenized_datasets = prepare_data(tokenizer)

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    # 3. Apply Subsampling (If configured)
    if MAX_TRAIN_SAMPLES is not None:
        print(f"DEBUG MODE: Truncating training set to {MAX_TRAIN_SAMPLES} samples.")
        train_dataset = train_dataset.select(range(MAX_TRAIN_SAMPLES))

    if MAX_VAL_SAMPLES is not None and len(eval_dataset) > MAX_VAL_SAMPLES:
        print(f"Truncating validation set to {MAX_VAL_SAMPLES} samples for faster evaluation.")
        eval_dataset = eval_dataset.select(range(MAX_VAL_SAMPLES))

    print(f"Final Training Size: {len(train_dataset)}")
    print(f"Final Validation Size: {len(eval_dataset)}")

    # 4. Model Setup
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        use_safetensors=True
    )

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy=EVAL_STRATEGY,
        eval_steps=EVAL_STEPS,
        save_strategy=EVAL_STRATEGY,    
        save_steps=SAVE_STEPS,
        save_total_limit=2,         
        learning_rate=2e-5,           
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=ACCUMULATION_STEPS,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_ratio=0.1,
        bf16=True,                     
        logging_dir='./logs',
        logging_steps=LOGGING_STEPS,
        load_best_model_at_end=True,     
        metric_for_best_model="accuracy",
        dataloader_num_workers=4,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    # 6. Train
    print("Starting training...")
    trainer.train()

    # 7. Final Save
    print("Saving final model...")
    save_path = FINAL_SAVE_PATH
    trainer.save_model(save_path)
    print("Done!")

if __name__ == "__main__":
    main()