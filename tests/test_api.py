"""Tests for the REST API (src/api.py), using FastAPI's built-in test client."""

import io
import json
import logging

import joblib
import pytest
from fastapi.testclient import TestClient

import src.predict as predict_module
from src.api import app
from src.monitoring import JsonFormatter, log

PATIENT = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
           "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}


@pytest.fixture
def client(fitted_pipeline, tmp_path, monkeypatch):
    """API client serving a small test model from a temporary folder."""
    joblib.dump(fitted_pipeline, tmp_path / "heart_disease_model.joblib")
    (tmp_path / "model_card.json").write_text(json.dumps(
        {"decision_threshold": 0.5, "created_at": "2026-01-01T00:00:00+00:00", "model_type": "test"}))
    monkeypatch.setattr(predict_module, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(predict_module, "_model", None)
    monkeypatch.setattr(predict_module, "_card", None)
    with TestClient(app) as c:          # "with" runs the startup step that loads the model
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    assert client.get("/model").json()["model_type"] == "test"


def test_predict_returns_prediction_and_confidence(client):
    r = client.post("/predict", json=PATIENT)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("Disease", "No disease")
    assert 0.5 <= body["confidence"] <= 1
    assert 0 <= body["disease_probability"] <= 1
    assert body["model_version"] == "2026-01-01T00:00:00+00:00"


def test_predict_accepts_missing_optional_fields(client):
    r = client.post("/predict", json={"age": 54, "sex": 0, "chol": 0, "ca": None})
    assert r.status_code == 200


@pytest.mark.parametrize("bad_input", [
    {"sex": 1},                              # age missing (required)
    {**PATIENT, "age": 250},                 # out of range
    {**PATIENT, "cp": 9},                    # invalid category
    {**PATIENT, "thal": 5},                  # not 3, 6 or 7
    {**PATIENT, "age": "sixty"},             # wrong type
    {**PATIENT, "cholesterol": 200},         # unknown field
])
def test_predict_rejects_invalid_input(client, bad_input):
    r = client.post("/predict", json=bad_input)
    assert r.status_code == 422              # FastAPI's "validation error" status


# ---------- Monitoring: metrics and logging (Deliverable 8) ----------

def test_metrics_endpoint_exposes_prometheus_metrics(client):
    client.post("/predict", json=PATIENT)
    text = client.get("/metrics").text
    assert "http_requests_total" in text
    assert 'endpoint="/predict"' in text
    assert "model_predictions_total" in text
    assert "model_disease_probability_bucket" in text
    assert "model_info" in text


def test_prediction_counter_increases(client):
    def total():
        lines = [line for line in client.get("/metrics").text.splitlines()
                 if line.startswith("model_predictions_total{")]
        return sum(float(line.rsplit(" ", 1)[1]) for line in lines)

    before = total()
    client.post("/predict", json=PATIENT)
    client.post("/predict", json=PATIENT)
    assert total() == before + 2


def test_missing_fields_are_counted(client):
    client.post("/predict", json={"age": 50, "sex": 1})
    assert 'model_input_missing_fields_total{field="chol"}' in client.get("/metrics").text


def test_every_response_has_a_request_id(client):
    r = client.get("/health")
    assert len(r.headers["X-Request-ID"]) > 0
    r = client.get("/health", headers={"X-Request-ID": "my-trace-123"})
    assert r.headers["X-Request-ID"] == "my-trace-123"     # caller's ID is kept for tracing


def test_requests_are_logged_as_json_without_patient_values(client):
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(JsonFormatter())
    log.addHandler(handler)
    try:
        client.post("/predict", json=PATIENT)
    finally:
        log.removeHandler(handler)
    lines = [json.loads(line) for line in buffer.getvalue().splitlines()]
    events = {entry.get("event") for entry in lines}
    assert {"request", "prediction"} <= events
    request_log = next(e for e in lines if e.get("event") == "request" and e["path"] == "/predict")
    assert request_log["status"] == 200 and request_log["duration_ms"] >= 0
    assert "chol" not in json.dumps(lines), "patient values must never be logged"
