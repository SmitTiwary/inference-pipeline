"""
FastAPI application entry point.

This module wires the whole service together:
  - Defines the Prometheus metrics that the BERT classifier emits.
  - Loads the BERT model once on startup (via the `lifespan` handler) and
    stores it on `app.state` so request handlers can reach it.
  - Registers the metrics middleware (times every HTTP request).
  - Mounts the inference router under the `/v1` URL prefix.
  - Exposes `/health` for liveness checks and `/metrics` for Prometheus
    to scrape.

Run locally:
    uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from app.routers import inference
from app.middleware.metrics import MetricsMiddleware
from app.utils.model_loader import ModelLoader


# ──────────────────────────────────────────────────────────────────────────────
# Prometheus metrics
#
# These objects are global counters / histograms that Prometheus scrapes from
# the /metrics endpoint. Each unique combination of label values becomes its
# own time series — e.g. {label="positive", model_version="v1"} and
# {label="negative", model_version="v1"} are tracked separately.
#
# Keep label cardinality LOW. Never put request_id or raw text in labels;
# those would explode the time-series database.
# ──────────────────────────────────────────────────────────────────────────────

# Generic request counter — bumped from the global exception handler below
# whenever an unhandled error escapes a route. The middleware tracks the
# normal success/error counts; this one is just a safety net.
REQUEST_COUNT = Counter(
    "inference_requests_total", "Total inference requests", ["endpoint", "status"]
)

# How many predictions we've returned, sliced by label and model version.
# Watching the ratio of positive vs negative over time is a cheap drift signal:
# if your input distribution suddenly skews 99% positive, something upstream
# changed (new client, bad preprocessing, label flip, ...).
PREDICTION_COUNT = Counter(
    "bert_predictions_total", "BERT predictions by label", ["label", "model_version"]
)

# Confidence (softmax probability of the predicted class) per request, bucketed.
# A real BERT on clear inputs sits in the 0.95–0.99 buckets. A creep down into
# the 0.5–0.7 buckets means the model is seeing inputs unlike its training data.
CONFIDENCE = Histogram(
    "bert_confidence",
    "Softmax confidence of the predicted class",
    ["model_version"],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
)

# Tokenized input length per request. Useful to catch when real traffic
# starts pushing past MAX_LENGTH (128) — that's invisible truncation that
# silently degrades accuracy.
INPUT_TOKENS = Histogram(
    "bert_input_tokens",
    "Tokenized input length per classify request",
    ["model_version"],
    buckets=[16, 32, 64, 128, 256, 512],
)


# ──────────────────────────────────────────────────────────────────────────────
# Lifespan — runs once at startup, once at shutdown.
#
# FastAPI calls this async context manager when the app boots. The code BEFORE
# `yield` runs at startup; code AFTER would run at shutdown (we don't need any).
# We use it to:
#   1. Load the BERT model into memory ONCE (loading on every request would
#      add ~5s of latency and waste memory).
#   2. Stash the model on `app.state.model` so request handlers can reach it.
#   3. Define a small `record_prediction` helper and put it on app.state too,
#      so the inference router doesn't need to import the metric objects.
# ──────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load BERT — see app/utils/model_loader.py. May download from HF Hub
    # the first time, which is why this is at startup, not per-request.
    app.state.model = ModelLoader().load()

    # A closure that the router calls after every prediction. Keeps Prometheus
    # bookkeeping out of the routing code.
    def record_prediction(label: str, confidence: float, n_tokens: int) -> None:
        version = app.state.model.version
        PREDICTION_COUNT.labels(label=label, model_version=version).inc()
        CONFIDENCE.labels(model_version=version).observe(confidence)
        INPUT_TOKENS.labels(model_version=version).observe(n_tokens)

    app.state.record_prediction = record_prediction

    # Control returns to FastAPI here. The app serves traffic until it's
    # shut down; nothing after `yield` because the BERT model has no
    # explicit cleanup needed.
    yield


# ──────────────────────────────────────────────────────────────────────────────
# Build the FastAPI app.
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="BERT Sentiment Classifier",
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware runs on EVERY incoming request, before/after the route handler.
# Ours times each request and records HTTP-level metrics.
app.add_middleware(MetricsMiddleware)

# Mount the inference routes under /v1/... — that's where /classify lives.
app.include_router(inference.router, prefix="/v1")


# ──────────────────────────────────────────────────────────────────────────────
# Operational endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Liveness probe used by Docker, Kubernetes, load balancers, etc.
    Returns the loaded model name + version so clients can confirm what's
    actually running on the other end of the wire."""
    return {
        "status": "ok",
        "model": app.state.model.name,
        "version": app.state.model.version,
    }


@app.get("/metrics")
async def metrics():
    """Prometheus scrape endpoint. `generate_latest()` serializes every
    Counter/Histogram defined anywhere in the process into the text format
    Prometheus expects."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Catch-all error handler. If any route raises an unhandled exception,
# this fires: bumps the error counter (so dashboards show a spike) and
# returns a clean JSON 500 instead of a stack trace.
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    REQUEST_COUNT.labels(endpoint=request.url.path, status="error").inc()
    return JSONResponse(status_code=500, content={"detail": str(exc)})
