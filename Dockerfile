FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 inference \
    && chown inference:inference /app

USER inference
ENV PATH="/home/inference/.local/bin:${PATH}"

COPY --chown=inference:inference requirements-serve.txt .
RUN pip install --no-cache-dir --user -r requirements-serve.txt

COPY --chown=inference:inference app/ ./app/

ENV MODEL_PATH=/bert_model \
    MODEL_VERSION=v1 \
    MAX_LENGTH=128 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
