"""
Inference HTTP routes — the public API of the service.

Exposes two endpoints under the /v1 prefix (set in app/main.py):
  POST /v1/classify        — classify a single text
  POST /v1/classify/batch  — classify up to 32 texts in one request

The router itself doesn't load or know about the model. Instead it reads
`request.app.state.model` (set by the lifespan handler in app/main.py).
This decoupling makes the router easy to test with a stub model.
"""

import uuid
from typing import List

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.utils.cleaner import clean_text

# An APIRouter is FastAPI's mini-app. The `tags=` value groups these routes
# together in the auto-generated Swagger docs at /docs.
router = APIRouter(tags=["inference"])


# ──────────────────────────────────────────────────────────────────────────────
# Request / response shapes (Pydantic models).
#
# FastAPI uses these for THREE things automatically:
#   1. Validation — bad inputs (empty string, too long, wrong type) are
#      rejected with HTTP 422 before the handler runs.
#   2. Serialization — return values get converted to JSON.
#   3. OpenAPI — Swagger docs & client-SDK generators read these definitions.
# ──────────────────────────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    # `Field(...)` marks this as required. min/max_length enforce sane limits:
    # empty strings are pointless, and 32k chars caps payload size to defend
    # against accidental DoS / memory blowups.
    text: str = Field(..., min_length=1, max_length=32_000)


class ClassifyResponse(BaseModel):
    request_id: str          # unique ID per call — useful for tracing in logs
    label: str               # "positive" or "negative"
    confidence: float        # softmax probability of the predicted class (0–1)
    input_tokens: int        # how many tokens BERT actually saw after tokenization


class BatchClassifyRequest(BaseModel):
    # Cap batch at 32 to bound per-request CPU time and memory.
    texts: List[str] = Field(..., min_length=1, max_length=32)


class BatchClassifyResponse(BaseModel):
    outputs: List[ClassifyResponse]


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/classify", response_model=ClassifyResponse)
async def classify(body: ClassifyRequest, request: Request):
    """Classify a single text. Pipeline: clean → tokenize+predict → record metrics → respond."""
    # Strip control chars, normalize unicode, etc. See app/utils/cleaner.py.
    text = clean_text(body.text)

    # Pull the loaded model off app.state (put there by the lifespan handler).
    model = request.app.state.model

    # The model handles tokenization + the actual neural net forward pass.
    label, confidence, n_tokens = model.classify(text)

    # Bump Prometheus counters/histograms. Defined back in app/main.py.
    request.app.state.record_prediction(label, confidence, n_tokens)

    return ClassifyResponse(
        request_id=_new_id(),
        label=label,
        confidence=confidence,
        input_tokens=n_tokens,
    )


@router.post("/classify/batch", response_model=BatchClassifyResponse)
async def classify_batch(body: BatchClassifyRequest, request: Request):
    """Classify many texts in one call. Sequentially loops the same path as
    /classify above. Could be batched at the model level for higher throughput,
    but a sequential loop is plenty for this demo and keeps the code simple."""
    model = request.app.state.model
    outputs = []
    for raw in body.texts:
        text = clean_text(raw)
        label, confidence, n_tokens = model.classify(text)
        request.app.state.record_prediction(label, confidence, n_tokens)
        outputs.append(
            ClassifyResponse(
                request_id=_new_id(),
                label=label,
                confidence=confidence,
                input_tokens=n_tokens,
            )
        )
    return BatchClassifyResponse(outputs=outputs)


def _new_id() -> str:
    """Generate a random request id. UUID4 = random, collision chance is
    negligible. Used for log correlation / debugging."""
    return str(uuid.uuid4())
