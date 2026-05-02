"""
HTTP-level Prometheus middleware.

Middleware in Starlette/FastAPI wraps every incoming request: code BEFORE
`call_next` runs as the request comes in, code AFTER runs as the response
goes out. We use that sandwich to:
  - record how long the request took (latency histogram), and
  - count requests by method, path, and status code.

These metrics are HTTP-generic — they don't know anything about BERT or
predictions. The model-specific metrics live in app/main.py.
"""

import time
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram


# Total HTTP requests, labelled so dashboards can slice by endpoint and
# isolate error spikes (status_code starts with "5" for server errors).
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status_code"]
)

# Request duration in seconds. A Histogram pre-buckets observations so
# Prometheus can compute percentiles (p50/p95/p99) cheaply via
# `histogram_quantile()` in PromQL.
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency", ["method", "endpoint"]
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Times every request and records counter + latency histogram."""

    async def dispatch(self, request, call_next):
        # `perf_counter` is a high-resolution monotonic timer — it never goes
        # backwards (unlike wall-clock time), so it's the right tool for
        # measuring elapsed time of short operations.
        start = time.perf_counter()

        # Hand control off to the rest of the app (route handlers, other
        # middleware). When this returns we have the response in hand.
        response = await call_next(request)

        duration = time.perf_counter() - start

        # request.url.path is the URL without query string — what we want
        # as the metric label so /v1/classify?foo=bar still groups under
        # /v1/classify.
        endpoint = request.url.path
        REQUEST_COUNT.labels(request.method, endpoint, response.status_code).inc()
        REQUEST_LATENCY.labels(request.method, endpoint).observe(duration)
        return response
