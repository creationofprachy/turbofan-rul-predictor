import json
import os

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


@pytest.fixture(scope="module")
def sample_readings():
    with open(os.path.join(MODELS_DIR, "sample_engines.json")) as f:
        return json.load(f)[0]["readings"]


def test_health_endpoint():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info_endpoint():
    r = client.get("/api/model-info")
    assert r.status_code == 200
    body = r.json()
    assert "results" in body and "best_model" in body


def test_samples_endpoint():
    r = client.get("/api/samples")
    assert r.status_code == 200
    assert len(r.json()) >= 1


def test_feature_schema_endpoint():
    r = client.get("/api/feature-schema")
    assert r.status_code == 200
    assert "required_fields" in r.json()


def test_predict_endpoint_success(sample_readings):
    r = client.post("/api/predict", json=sample_readings)
    assert r.status_code == 200
    body = r.json()
    assert "predicted_rul_cycles" in body
    assert body["status"] in ("healthy", "warning", "critical")


def test_predict_endpoint_missing_field_returns_422(sample_readings):
    bad = dict(sample_readings)
    del bad["sensor_2"]
    r = client.post("/api/predict", json=bad)
    assert r.status_code == 422


def test_explain_endpoint_success(sample_readings):
    r = client.post("/api/explain", json=sample_readings)
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list) and len(body) > 0
    assert "feature" in body[0] and "impact" in body[0]


def test_index_page_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "Turbofan" in r.text
