import os
import logging
from dataclasses import dataclass, field
from typing import Tuple

logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "./bert_model")
MODEL_REVISION = os.getenv("MODEL_REVISION")  # optional: HF branch / tag / commit SHA
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
HF_TOKEN = os.getenv("HF_TOKEN")  # required only for private HF repos
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))
LABELS = ["negative", "positive"]


@dataclass
class LoadedModel:
    name: str
    version: str
    _model: object = field(repr=False)
    _tokenizer: object = field(repr=False)
    _device: object = field(repr=False)

    def classify(self, text: str) -> Tuple[str, float, int]:
        """Returns (label, confidence, input_token_count)."""
        import torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )
        token_count = int(inputs["input_ids"].shape[1])
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        self._model.eval()
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(probs).item())
        return LABELS[idx], float(probs[idx].item()), token_count


class ModelLoader:
    def load(self) -> LoadedModel:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
            source = "local dir" if os.path.isdir(MODEL_PATH) else "HF Hub"
            logger.info(
                "Loading BERT classifier from %s (%s) on %s", MODEL_PATH, source, device
            )
            kwargs = {}
            if MODEL_REVISION:
                kwargs["revision"] = MODEL_REVISION
            if HF_TOKEN:
                kwargs["token"] = HF_TOKEN
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, **kwargs)
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_PATH, **kwargs
            ).to(device)
            return LoadedModel(
                name=os.path.basename(MODEL_PATH.rstrip("/")),
                version=MODEL_VERSION,
                _model=model,
                _tokenizer=tokenizer,
                _device=device,
            )
        except (ImportError, OSError) as e:
            logger.warning("Falling back to stub classifier (reason: %s)", e)
            return _StubModel(
                name="stub-bert",
                version=MODEL_VERSION,
                _model=None,
                _tokenizer=None,
                _device=None,
            )


class _StubModel(LoadedModel):
    """Returned when transformers/torch isn't available or no trained model exists."""

    def classify(self, text: str) -> Tuple[str, float, int]:
        positive_words = {"good", "great", "amazing", "love", "excellent", "awesome", "best"}
        negative_words = {"bad", "terrible", "awful", "hate", "worst", "horrible"}
        tokens = text.lower().split()
        score = sum(1 for t in tokens if t in positive_words) - sum(
            1 for t in tokens if t in negative_words
        )
        label = "positive" if score >= 0 else "negative"
        return label, 0.5, len(tokens)
