import json
import os

import pytest

from src.predict import RULPredictor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")


@pytest.fixture(scope="module")
def predictor():
    return RULPredictor(MODELS_DIR)


@pytest.fixture(scope="module")
def sample_readings():
    with open(os.path.join(MODELS_DIR, "sample_engines.json")) as f:
        samples = json.load(f)
    return samples


def test_predictor_loads_artifacts(predictor):
    assert predictor.model is not None
    assert predictor.scaler is not None
    assert predictor.model_name in ("LinearRegression", "RandomForest", "XGBoost")


def test_predict_returns_expected_schema(predictor, sample_readings):
    result = predictor.predict(sample_readings[0]["readings"])
    assert set(result.keys()) == {
        "predicted_rul_cycles", "status", "urgency", "model_used"
    }
    assert isinstance(result["predicted_rul_cycles"], float)
    assert result["status"] in ("healthy", "warning", "critical")


def test_predict_missing_field_raises(predictor, sample_readings):
    bad_input = dict(sample_readings[0]["readings"])
    del bad_input["sensor_2"]
    with pytest.raises(ValueError):
        predictor.predict(bad_input)


def test_predict_ordering_matches_engine_health(predictor, sample_readings):
    """A sample labelled 'Critical' should predict a lower RUL than one
    labelled 'Healthy' -- sanity check that the model learned a sensible
    direction, not an exact value match."""
    critical = next(s for s in sample_readings if "Critical" in s["label"])
    healthy = next(s for s in sample_readings if "Healthy" in s["label"])
    pred_critical = predictor.predict(critical["readings"])["predicted_rul_cycles"]
    pred_healthy = predictor.predict(healthy["readings"])["predicted_rul_cycles"]
    assert pred_critical < pred_healthy


def test_predict_output_bounded(predictor, sample_readings):
    for s in sample_readings:
        result = predictor.predict(s["readings"])
        assert 0 <= result["predicted_rul_cycles"] <= predictor.rul_clip


def test_explain_returns_top_k_features(predictor, sample_readings):
    result = predictor.explain(sample_readings[0]["readings"], top_k=5)
    assert len(result) == 5
    for item in result:
        assert "feature" in item and "impact" in item
