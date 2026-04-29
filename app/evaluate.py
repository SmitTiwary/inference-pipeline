"""
Standalone model evaluation script.

Loads a trained model from OUTPUT_DIR (default: ./models/bert_model_sen),
runs it against a sample of the IMDB test set, and writes metrics to
eval_results.json in the same directory.

Usage:
    python -m app.evaluate
    OUTPUT_DIR=./my_model EVAL_SAMPLES=5000 python -m app.evaluate
"""

import json
import os
import time

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./models/bert_model_sen")
EVAL_SAMPLES = int(os.getenv("EVAL_SAMPLES", "2000"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
LABELS = ["negative", "positive"]


def _batched(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def evaluate(model_dir: str = OUTPUT_DIR, n_samples: int = EVAL_SAMPLES) -> dict:
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Loading model from {model_dir} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    print(f"Loading IMDB test set ({n_samples} samples)")
    dataset = load_dataset("imdb", split="test").shuffle(seed=42).select(range(n_samples))
    texts = dataset["text"]
    true_labels = dataset["label"]  # 0=negative, 1=positive

    preds, confidences = [], []
    t0 = time.perf_counter()

    for batch_texts in _batched(texts, BATCH_SIZE):
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        batch_preds = torch.argmax(probs, dim=1).cpu().tolist()
        batch_confs = probs.max(dim=1).values.cpu().tolist()
        preds.extend(batch_preds)
        confidences.extend(batch_confs)

    elapsed = time.perf_counter() - t0
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, preds, average="binary", zero_division=0
    )

    results = {
        "model_dir": model_dir,
        "n_samples": n_samples,
        "accuracy": round(accuracy_score(true_labels, preds), 4),
        "f1": round(float(f1), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "mean_confidence": round(float(np.mean(confidences)), 4),
        "latency_ms_per_sample": round(elapsed * 1000 / n_samples, 2),
    }

    out_path = os.path.join(model_dir, "eval_results.json")
    os.makedirs(model_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")
    return results


if __name__ == "__main__":
    evaluate()
