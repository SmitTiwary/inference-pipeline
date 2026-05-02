"""
Model loading and inference wrapper.

Handles three concerns:
  1. Reads the model location and tuning knobs from environment variables.
  2. Loads the BERT classifier from either a local directory OR a Hugging
     Face Hub repo id (transformers' from_pretrained() accepts both).
  3. Wraps the loaded model in a small `LoadedModel` class that exposes
     a single `classify(text)` method — that's all the rest of the app
     needs to know about.

If transformers/torch isn't installed OR the model can't be found, the
loader silently falls back to a tiny keyword-matching stub. That keeps
the API responsive in CI / on developer machines without the real
model — useful for testing the HTTP layer.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration — all driven by env vars so deploys (dev/stag/prod) can
# load different models without code changes.
# ──────────────────────────────────────────────────────────────────────────────

# Where to load the model from. Two valid formats:
#   • a local directory path, e.g. "./bert_model"
#   • a Hugging Face Hub repo id, e.g. "Glsmit/bert-imdb-sentiment"
MODEL_PATH = os.getenv("MODEL_PATH", "./bert_model")

# When MODEL_PATH points at HF Hub, you can pin to a specific revision
# (branch / tag / commit SHA) for reproducibility. None = "latest on main".
MODEL_REVISION = os.getenv("MODEL_REVISION")

# A free-form version label that's attached to every Prometheus metric.
# Bump it whenever you retrain so dashboards can compare model versions
# side-by-side (e.g. for A/B tests after a redeploy).
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

# Only needed when pulling from a *private* HF repo. Leave empty for public.
HF_TOKEN = os.getenv("HF_TOKEN")

# Tokenizer truncates inputs longer than this. BERT's hard limit is 512;
# 128 is a good speed/accuracy tradeoff for short reviews.
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))

# Index → human-readable label. Order matters: must match how the model
# was trained (label 0 = negative, 1 = positive in the IMDB convention).
LABELS = ["negative", "positive"]


# ──────────────────────────────────────────────────────────────────────────────
# LoadedModel — what the rest of the app sees.
#
# We use a dataclass to bundle the model, tokenizer, and target device
# together. The leading-underscore fields are private; only `classify()`
# is meant to be called from outside.
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class LoadedModel:
    name: str                                 # human-readable name (for /health, logs)
    version: str                              # version tag for metrics
    _model: object = field(repr=False)        # the actual transformers model
    _tokenizer: object = field(repr=False)    # the matching tokenizer
    _device: object = field(repr=False)       # torch.device — cuda / mps / cpu

    def classify(self, text: str) -> Tuple[str, float, int]:
        """Run BERT on a single string and return (label, confidence, token_count)."""
        # Imported here (not at the top) so the *file* still imports cleanly
        # in environments without torch — the stub fallback can serve traffic.
        import torch

        # Tokenize: convert raw text into the integer IDs BERT expects.
        # `return_tensors="pt"` gives us a PyTorch tensor.
        # `truncation` + `max_length` cap long inputs to MAX_LENGTH tokens.
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )

        # Record the input length BEFORE moving to GPU so we can return it
        # to the caller for the bert_input_tokens histogram metric.
        token_count = int(inputs["input_ids"].shape[1])

        # Move every tensor in the batch to the same device as the model.
        # Mixing devices (some on CPU, some on GPU) raises a runtime error.
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Switch the model into evaluation mode (disables dropout, etc.)
        # and skip gradient tracking — we're not training, so don't waste
        # memory/compute building the autograd graph.
        self._model.eval()
        with torch.no_grad():
            logits = self._model(**inputs).logits

        # Logits → probabilities (softmax normalizes them to sum to 1).
        # `[0]` extracts the single example out of the batch dimension.
        probs = torch.softmax(logits, dim=1)[0]

        # The predicted class is the index with the highest probability.
        idx = int(torch.argmax(probs).item())

        return LABELS[idx], float(probs[idx].item()), token_count


# ──────────────────────────────────────────────────────────────────────────────
# ModelLoader — runs once at app startup. Returns a `LoadedModel`.
# ──────────────────────────────────────────────────────────────────────────────
class ModelLoader:
    def load(self) -> LoadedModel:
        try:
            # Late imports so the loader still works in stub mode when
            # torch / transformers aren't installed.
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            # Pick the fastest device available:
            #   cuda → NVIDIA GPU (Linux/Windows boxes with an NVIDIA card)
            #   mps  → Apple Silicon GPU (M1/M2/M3 Macs)
            #   cpu  → fallback that always works
            device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )

            # Just for logging. `os.path.isdir` distinguishes a local dir
            # from a HF repo id — both work as inputs to from_pretrained.
            source = "local dir" if os.path.isdir(MODEL_PATH) else "HF Hub"
            logger.info(
                "Loading BERT classifier from %s (%s) on %s", MODEL_PATH, source, device
            )

            # Optional kwargs only added if set, so we don't pass `token=None`
            # which would override the cached HF auth.
            kwargs = {}
            if MODEL_REVISION:
                kwargs["revision"] = MODEL_REVISION
            if HF_TOKEN:
                kwargs["token"] = HF_TOKEN

            # Tokenizer must come from the SAME model dir/repo as the weights —
            # they share a vocab. Mismatched tokenizer → garbage predictions.
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, **kwargs)

            # Loads the model weights; `.to(device)` moves the parameters
            # onto the chosen accelerator (cuda/mps) so inference is fast.
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_PATH, **kwargs
            ).to(device)

            return LoadedModel(
                # Strip trailing slashes then take the last path component,
                # so "./bert_model/" → "bert_model" and "Glsmit/foo" → "foo".
                name=os.path.basename(MODEL_PATH.rstrip("/")),
                version=MODEL_VERSION,
                _model=model,
                _tokenizer=tokenizer,
                _device=device,
            )

        except (ImportError, OSError) as e:
            # ImportError: torch / transformers not installed.
            # OSError: model dir / HF repo doesn't exist.
            # In either case the API still needs to come up — return the stub.
            logger.warning("Falling back to stub classifier (reason: %s)", e)
            return _StubModel(
                name="stub-bert",
                version=MODEL_VERSION,
                _model=None,
                _tokenizer=None,
                _device=None,
            )


class _StubModel(LoadedModel):
    """Tiny keyword-based classifier used when no real model can be loaded.

    Counts positive-vibe vs negative-vibe words in the text and picks
    whichever wins. Always reports confidence = 0.5 so monitoring can
    spot stub usage in production (real BERT confidence is usually
    above 0.9 on clear inputs).

    Used by:
      • CI runs without the trained weights mounted in
      • Frontend devs who just need the API shape, not real predictions
    """

    def classify(self, text: str) -> Tuple[str, float, int]:
        positive_words = {"good", "great", "amazing", "love", "excellent", "awesome", "best"}
        negative_words = {"bad", "terrible", "awful", "hate", "worst", "horrible"}
        tokens = text.lower().split()
        score = sum(1 for t in tokens if t in positive_words) - sum(
            1 for t in tokens if t in negative_words
        )
        # Tie goes to "positive" — arbitrary but consistent.
        label = "positive" if score >= 0 else "negative"
        return label, 0.5, len(tokens)
