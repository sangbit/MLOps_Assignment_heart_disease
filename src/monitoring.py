"""Logging and metrics for the API.

LOGGING  -- one JSON line per request (and one per prediction), written to stdout.
            JSON is machine-readable, so log tools (docker logs, kubectl logs, Loki, ELK...)
            can filter and search it. Patient values are never logged (health data).

METRICS  -- counters and histograms exposed at GET /metrics in Prometheus format.
            Prometheus scrapes them every few seconds; Grafana draws them.

    Service health:  requests per endpoint/status, request latency
    Model behaviour: predictions per class, distribution of predicted probabilities,
                     missing input fields (a simple data-quality / drift signal)
"""

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime

from prometheus_client import Counter, Gauge, Histogram
from starlette.middleware.base import BaseHTTPMiddleware

# ---------------------------------------------------------------- logging

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        entry.update(getattr(record, "fields", {}))
        return json.dumps(entry)


def get_logger(name: str = "heart_api") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:                      # configure once
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


log = get_logger()

# ---------------------------------------------------------------- metrics

REQUESTS = Counter(
    "http_requests_total", "HTTP requests handled", ["method", "endpoint", "status"])
LATENCY = Histogram(
    "http_request_duration_seconds", "Time to handle a request", ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
PREDICTIONS = Counter(
    "model_predictions_total", "Predictions returned, by predicted class", ["label"])
PROBABILITY = Histogram(
    "model_disease_probability", "Distribution of predicted disease probabilities",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
MISSING_FIELDS = Counter(
    "model_input_missing_fields_total", "Prediction requests with a field left empty", ["field"])
MODEL_INFO = Gauge(
    "model_info", "Model being served (value is always 1)", ["model_type", "version"])

SKIP_LOGGING = {"/metrics"}       # scraped every few seconds; logging it would flood the logs


def record_prediction(result: dict, patient: dict) -> None:
    """Update model metrics and write a prediction log line (no patient values)."""
    PREDICTIONS.labels(label=result["label"]).inc()
    PROBABILITY.observe(result["disease_probability"])
    missing = [k for k, v in patient.items() if v is None]
    for field in missing:
        MISSING_FIELDS.labels(field=field).inc()
    log.info("prediction", extra={"fields": {
        "event": "prediction", "label": result["label"],
        "disease_probability": result["disease_probability"],
        "confidence": result["confidence"], "missing_fields": len(missing)}})


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Times every request, records metrics, logs one JSON line, and adds an X-Request-ID header."""

    async def dispatch(self, request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            duration = time.perf_counter() - start
            route = request.scope.get("route")
            endpoint = route.path if route else "unmatched"      # route template keeps labels few
            if endpoint != "/metrics":
                REQUESTS.labels(request.method, endpoint, str(status)).inc()
                LATENCY.labels(endpoint).observe(duration)
            if endpoint not in SKIP_LOGGING:
                log.info("request", extra={"fields": {
                    "event": "request", "method": request.method, "path": request.url.path,
                    "status": status, "duration_ms": round(duration * 1000, 2),
                    "client": request.client.host if request.client else None}})
            request_id_var.reset(token)
