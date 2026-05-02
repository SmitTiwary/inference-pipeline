"""
Training script — fine-tunes BERT on the IMDB sentiment dataset.

Run it manually when you want to (re)train the model:
    python -m app.train
    BATCH_SIZE=32 EPOCHS=5 python -m app.train

Outputs a saved model to OUTPUT_DIR (default: ./models/bert_model_sen),
which the inference service can then load via MODEL_PATH=./models/bert_model_sen.

Importing this module (e.g. `from app import train`) does NOT trigger
training — the actual work runs only inside `if __name__ == "__main__":`
at the bottom.
"""

import os

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters and paths — all overridable via environment variables.
# ──────────────────────────────────────────────────────────────────────────────

# The pretrained checkpoint we start from. "bert-base-uncased" is 12 layers,
# 110M params, lowercased English. Hugging Face downloads it automatically.
MODEL_NAME = os.getenv("BASE_MODEL", "bert-base-uncased")

# Where to save the final fine-tuned model + tokenizer.
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./models/bert_model_sen")

# Tokens past this index get truncated. 128 is plenty for the IMDB
# review snippets we're training on.
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))

# Per-device batch size for both training and eval. 16 fits comfortably
# on a single consumer GPU; bump to 32 or 64 if you have more VRAM.
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))

# Full passes through the training set.
EPOCHS = int(os.getenv("EPOCHS", "3"))

# Binary classification — positive (1) and negative (0).
NUM_LABELS = 2


def compute_metrics(eval_pred):
    """Hugging Face Trainer calls this after each eval epoch.

    Receives raw logits + true labels, returns a dict of metric → value
    that gets logged and used by `metric_for_best_model`.
    """
    logits, labels = eval_pred
    # argmax over the class axis gives the predicted label index.
    predictions = np.argmax(logits, axis=1)

    # `binary` averaging is correct for two-class problems; zero_division=0
    # silences a noisy warning that fires when a fold has only one class.
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def main():
    # Pick the fastest available compute device. fp16 is only enabled
    # for cuda below — MPS / CPU don't support it well in this version.
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # ── Data ──────────────────────────────────────────────────────────────
    # Streams 25k train + 25k test IMDB reviews from Hugging Face Datasets.
    # First run downloads ~80MB; subsequent runs use the local cache.
    dataset = load_dataset("imdb")

    # 10k train / 2k test is enough to fine-tune a 110M-param BERT to ~90%+
    # accuracy in a few minutes on a GPU. Shuffle BEFORE selecting so we
    # don't get a slice biased by the dataset's internal ordering.
    train_data = dataset["train"].shuffle(seed=42).select(range(10000))
    test_data = dataset["test"].shuffle(seed=42).select(range(2000))

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(example):
        """Convert a row of text → input_ids + attention_mask tensors."""
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",      # pad every example to MAX_LENGTH
            max_length=MAX_LENGTH,
        )

    # `map(..., batched=True)` runs tokenize on chunks of rows for speed.
    train_dataset = train_data.map(tokenize, batched=True)
    test_dataset = test_data.map(tokenize, batched=True)

    # Tell HF Datasets to hand back PyTorch tensors with these column names —
    # that's what BertForSequenceClassification expects in its forward().
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # ── Model ─────────────────────────────────────────────────────────────
    # Loads pretrained BERT and slaps a randomly-initialized classification
    # head on top. The head is what we'll actually train; the rest mostly
    # adapts via fine-tuning.
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model.to(device)

    # ── Training config ───────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir="./bert_classification",  # checkpoints during training
        learning_rate=2e-5,                  # standard BERT fine-tuning LR
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        eval_strategy="epoch",               # evaluate at the end of each epoch
        save_strategy="epoch",               # save a checkpoint each epoch
        logging_steps=100,                   # print loss every 100 steps
        save_total_limit=2,                  # keep at most 2 checkpoints on disk
        load_best_model_at_end=True,         # at the end, restore the best weights
        metric_for_best_model="f1",          # "best" is judged by F1 score
        fp16=torch.cuda.is_available(),      # half-precision for speed (CUDA only)
        report_to="none",                    # don't push to wandb/tensorboard
    )

    # ── Trainer wires it all together ────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("\nEvaluation Results:", trainer.evaluate())

    # Persist final model + tokenizer to OUTPUT_DIR. Both are needed at
    # inference time — the tokenizer's vocab and config must match.
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")


# Guard so `import app.train` doesn't kick off a multi-hour training run.
# Only `python -m app.train` (or `python app/train.py`) actually trains.
if __name__ == "__main__":
    main()
