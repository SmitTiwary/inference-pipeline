import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from asgi_lifespan import LifespanManager
from app.main import app


@pytest_asyncio.fixture
async def client():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c


# ── Health ───────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model" in body and "version" in body


# ── Single classify ──────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_classify(client):
    r = await client.post("/v1/classify", json={"text": "This movie was amazing!"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in ("positive", "negative")
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["input_tokens"] >= 1
    assert "request_id" in body


@pytest.mark.asyncio
async def test_classify_empty_text_rejected(client):
    r = await client.post("/v1/classify", json={"text": ""})
    assert r.status_code == 422


# ── Batch classify ───────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_classify_batch(client):
    r = await client.post(
        "/v1/classify/batch",
        json={"texts": ["I loved it.", "Worst experience ever.", "It was fine."]},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["outputs"]) == 3
    for item in body["outputs"]:
        assert item["label"] in ("positive", "negative")
        assert "request_id" in item


# ── Data cleaning ────────────────────────────────────────────────────────────
def test_cleaner_strips_control_chars():
    from app.utils.cleaner import clean_text
    assert "\x00" not in clean_text("Hello\x00\x01\x07World")
    assert "Hello" in clean_text("Hello\x00World")


def test_cleaner_collapses_whitespace():
    from app.utils.cleaner import clean_text
    assert clean_text("  hello   world  ") == "hello world"


def test_cleaner_truncates():
    from app.utils.cleaner import clean_text
    assert len(clean_text("a" * 100_000)) == 32_000


# ── Stub classifier unit tests ───────────────────────────────────────────────
def test_stub_classify_positive():
    from app.utils.model_loader import _StubModel
    m = _StubModel(name="stub", version="0", _model=None, _tokenizer=None, _device=None)
    label, conf, n = m.classify("This was amazing and great")
    assert label == "positive"
    assert n > 0


def test_stub_classify_negative():
    from app.utils.model_loader import _StubModel
    m = _StubModel(name="stub", version="0", _model=None, _tokenizer=None, _device=None)
    label, _, _ = m.classify("worst terrible awful")
    assert label == "negative"


# ── Metrics ──────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_metrics_endpoint(client):
    # Trigger a classify so prediction metrics get emitted
    await client.post("/v1/classify", json={"text": "great"})
    r = await client.get("/metrics")
    assert r.status_code == 200
    assert b"http_requests_total" in r.content
    assert b"bert_predictions_total" in r.content
    assert b"bert_confidence" in r.content
    assert b"bert_input_tokens" in r.content
