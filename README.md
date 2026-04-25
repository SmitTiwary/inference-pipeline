# Inference Pipeline — BERT Sentiment Classifier

A production-style FastAPI service that serves a fine-tuned BERT model for binary sentiment classification (positive / negative). Includes Prometheus metrics, a Grafana dashboard, Docker Compose deployment, and a stub mode for CI.

Model: `bert-base-uncased` fine-tuned on the IMDB reviews dataset (10k train / 2k test subset).

---

## Quickstart

### 1. Set up the environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Get the model
**Option A — pull from Hugging Face Hub (recommended)**
```bash
export MODEL_PATH=Glsmit/bert-imdb-sentiment
```
The first request downloads the model into `~/.cache/huggingface/`; subsequent runs are instant.

**Option B — use a local model directory**
```bash
export MODEL_PATH=./bert_model           # default
```

**Option C — train your own**
```bash
python -m app.train                      # writes ./models/bert_model_sen/
export MODEL_PATH=./models/bert_model_sen
```

### 3. Run the server
```bash
uvicorn app.main:app --reload
```
Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the interactive Swagger UI.

---

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/health` | — | `{"status","model","version"}` |
| `GET` | `/metrics` | — | Prometheus text format |
| `POST` | `/v1/classify` | `{"text": "..."}` | `{request_id, label, confidence, input_tokens}` |
| `POST` | `/v1/classify/batch` | `{"texts": ["...", "..."]}` (max 32) | `{outputs: [...]}` |

### Example
```bash
curl -s localhost:8000/v1/classify \
  -H 'content-type: application/json' \
  -d '{"text":"This movie was amazing!"}'
```
```json
{"request_id":"...","label":"positive","confidence":0.998,"input_tokens":7}
```

---

## Configuration

All settings come from environment variables. Defaults work out of the box.

| Env var | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `./bert_model` | Local directory **or** Hugging Face repo id (`user/repo`) |
| `MODEL_REVISION` | (latest) | Pin to a specific HF branch / tag / commit SHA |
| `MODEL_VERSION` | `v1` | Tag attached to all Prometheus metrics — bump on retrain |
| `HF_TOKEN` | — | Required only when pulling a private HF repo |
| `MAX_LENGTH` | `128` | Tokenizer max sequence length |

### Training-only env vars
| Env var | Default | Purpose |
|---|---|---|
| `BASE_MODEL` | `bert-base-uncased` | Starting checkpoint for fine-tuning |
| `OUTPUT_DIR` | `./models/bert_model_sen` | Where the trained model is saved |
| `BATCH_SIZE` | `16` | Per-device batch size |
| `EPOCHS` | `3` | Training epochs |

---

## Monitoring

The service exports Prometheus metrics at `/metrics`:

| Metric | Type | Labels | Use |
|---|---|---|---|
| `http_requests_total` | counter | `method, endpoint, status_code` | Request rate, error rate |
| `http_request_duration_seconds` | histogram | `method, endpoint` | p50/p95/p99 latency |
| `bert_predictions_total` | counter | `label, model_version` | Class mix over time (drift detection) |
| `bert_confidence` | histogram | `model_version` | Prediction confidence distribution |
| `bert_input_tokens` | histogram | `model_version` | Input length distribution |

A pre-provisioned Grafana dashboard lives at [`monitoring/grafana/dashboards/inference.json`](monitoring/grafana/dashboards/inference.json).

### Bring up Prometheus + Grafana
```bash
docker compose up
```
| Service | URL |
|---|---|
| Inference API | http://localhost:8000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000  (admin / admin) |

The compose file mounts `./bert_model/` into the container at `/bert_model`.

---

## Tests
```bash
pytest
```
The test suite uses a stub classifier when no real model is on disk, so it passes without the BERT weights or a GPU. CI-friendly.

---

## Project layout
```
app/
  main.py               # FastAPI app, lifespan, metrics registration
  train.py              # Fine-tunes BERT on IMDB (run with `python -m app.train`)
  routers/inference.py  # /v1/classify and /v1/classify/batch
  middleware/metrics.py # Prometheus request/latency middleware
  utils/
    model_loader.py     # Loads from local dir OR HF Hub; CUDA / MPS / CPU autodetect
    cleaner.py          # Input sanitization
monitoring/
  prometheus/           # Scrape config
  grafana/dashboards/   # Provisioned dashboards
tests/test_api.py       # API + stub model + cleaner tests
examples/client.py      # Sample client (single + batch)
Dockerfile              # python:3.11-slim, CPU-friendly
docker-compose.yml      # inference + prometheus + grafana
```

---

## Notes

- **Apple Silicon**: `torch` autodetects the MPS backend, so inference runs on the Mac GPU automatically — no config needed.
- **`bert_model/` is gitignored.** Trained weights live on Hugging Face Hub, not in this repo. See [.gitignore](.gitignore).
- **Stub fallback**: if `transformers` isn't installed or `MODEL_PATH` doesn't resolve, [model_loader.py](app/utils/model_loader.py) returns a keyword-based stub classifier so the API still responds. Useful for CI and frontend development.
