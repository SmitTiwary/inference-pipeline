from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from app.routers import inference
from app.middleware.metrics import MetricsMiddleware
from app.utils.model_loader import ModelLoader

REQUEST_COUNT = Counter(
    "inference_requests_total", "Total inference requests", ["endpoint", "status"]
)
PREDICTION_COUNT = Counter(
    "bert_predictions_total", "BERT predictions by label", ["label", "model_version"]
)
CONFIDENCE = Histogram(
    "bert_confidence",
    "Softmax confidence of the predicted class",
    ["model_version"],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
)
INPUT_TOKENS = Histogram(
    "bert_input_tokens",
    "Tokenized input length per classify request",
    ["model_version"],
    buckets=[16, 32, 64, 128, 256, 512],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = ModelLoader().load()

    def record_prediction(label: str, confidence: float, n_tokens: int) -> None:
        version = app.state.model.version
        PREDICTION_COUNT.labels(label=label, model_version=version).inc()
        CONFIDENCE.labels(model_version=version).observe(confidence)
        INPUT_TOKENS.labels(model_version=version).observe(n_tokens)

    app.state.record_prediction = record_prediction
    yield


app = FastAPI(
    title="BERT Sentiment Classifier",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(MetricsMiddleware)
app.include_router(inference.router, prefix="/v1")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": app.state.model.name,
        "version": app.state.model.version,
    }


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    REQUEST_COUNT.labels(endpoint=request.url.path, status="error").inc()
    return JSONResponse(status_code=500, content={"detail": str(exc)})
