"""
Standalone model evaluation script.

Loads a trained model from OUTPUT_DIR (default: ./models/bert_model_sen),
runs it against a sample of the IMDB test set, and writes metrics to
eval_results.json in the same directory.

Used in two places:
  • Manual quality checks after retraining (run on your laptop).
  • The CI 'evaluate' job, which uploads eval_results.json as an artifact
    so reviewers can see how a candidate model actually performs before
    promoting it to a higher environment.

Usage:
    python -m app.evaluate
    OUTPUT_DIR=./my_model EVAL_SAMPLES=5000 python -m app.evaluate
    OUTPUT_DIR=Glsmit/bert-imdb-sentiment python -m app.evaluate   # HF repo also works
"""

import json
import os
import time

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# ──────────────────────────────────────────────────────────────────────────────
# Config — all env-var overridable.
# ──────────────────────────────────────────────────────────────────────────────

# Model location: local directory OR Hugging Face repo id (transformers'
# from_pretrained accepts either). Same convention as MODEL_PATH in the
# inference service — kept under a different name (OUTPUT_DIR) since this
# script is also where eval_results.json gets written.
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./models/bert_model_sen")

# How many examples from the IMDB test set to evaluate on. Smaller is
# faster (CI uses 200–500); larger is more statistically meaningful.
EVAL_SAMPLES = int(os.getenv("EVAL_SAMPLES", "2000"))

# Same tokenizer cap as inference. Stay consistent so eval reflects what
# production actually sees.
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))

# Batch size used while running inference. Bigger = faster (more parallelism)
# but more memory. 32 is safe almost everywhere.
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

# Same label convention as inference (0 = negative, 1 = positive).
LABELS = ["negative", "positive"]


def _batched(lst, n):
    """Yield successive chunks of size n from lst.

    Generator — doesn't allocate a new list of lists. Used so we can feed
    examples through the tokenizer/model in batches without loading everything
    onto the GPU at once.
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def evaluate(model_dir: str = OUTPUT_DIR, n_samples: int = EVAL_SAMPLES) -> dict:
    """Run the loaded model on n_samples IMDB reviews and write metrics.

    Returns the metrics dict (also printed to stdout and saved to disk).
    """
    # Pick the fastest available device — same logic as inference.
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # ── Load model + tokenizer ───────────────────────────────────────────
    print(f"Loading model from {model_dir} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    # eval() disables dropout etc. — important for deterministic predictions.
    model.eval()

    # ── Load + sample data ───────────────────────────────────────────────
    print(f"Loading IMDB test set ({n_samples} samples)")
    # Same shuffle seed as train.py so we never accidentally evaluate on
    # something we trained on (would over-report accuracy).
    dataset = load_dataset("imdb", split="test").shuffle(seed=42).select(range(n_samples))
    texts = dataset["text"]
    true_labels = dataset["label"]

    # ── Inference loop, batched ──────────────────────────────────────────
    preds, confidences = [], []
    t0 = time.perf_counter()       # measure end-to-end latency

    for batch_texts in _batched(texts, BATCH_SIZE):
        # Tokenize a whole batch in one call. `padding=True` pads to the
        # longest example IN THE BATCH (not MAX_LENGTH), which saves compute.
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        # Move every tensor to the same device as the model.
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # `no_grad` skips the autograd graph — we're not training so we don't
        # need to track gradients. Saves a lot of memory and some time.
        with torch.no_grad():
            logits = model(**inputs).logits

        # Convert logits → probabilities → predicted class index + confidence.
        # Detach to CPU before .tolist() so the GPU memory can be reused for
        # the next batch.
        probs = torch.softmax(logits, dim=1)
        batch_preds = torch.argmax(probs, dim=1).cpu().tolist()
        batch_confs = probs.max(dim=1).values.cpu().tolist()

        preds.extend(batch_preds)
        confidences.extend(batch_confs)

    elapsed = time.perf_counter() - t0

    # ── Aggregate metrics ────────────────────────────────────────────────
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, preds, average="binary", zero_division=0
    )

    # `round(...)` to keep the JSON output readable; floats are coerced
    # explicitly because numpy types don't always serialize cleanly.
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

    # ── Persist results ──────────────────────────────────────────────────
    # Write next to the model. The CI workflow's "Upload eval results" step
    # globs **/eval_results.json, so this gets caught regardless of where
    # model_dir is.
    out_path = os.path.join(model_dir, "eval_results.json")
    os.makedirs(model_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")
    return results


if __name__ == "__main__":
    evaluate()
