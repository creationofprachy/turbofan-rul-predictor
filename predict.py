"""Inference-time utilities: load the trained model/scaler once and expose
a simple predict() function used by both the API and the tests.
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")


class RULPredictor:
    """Loads the persisted model + scaler + metadata and serves predictions.

    Because a single live sensor reading has no history to compute rolling
    statistics from, at inference time we fall back to using the raw
    instantaneous readings for the rolling-mean features and zero for the
    rolling-std features (i.e. "assume this reading is representative of
    the recent trend"). This keeps the API usable for a single snapshot
    reading while the offline evaluation (src/train.py) uses true rolling
    windows computed over full run-to-failure sequences.
    """

    def __init__(self, models_dir: str = MODELS_DIR):
        self.model = joblib.load(os.path.join(models_dir, "best_model.joblib"))
        self.scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
        with open(os.path.join(models_dir, "metadata.json")) as f:
            self.metadata = json.load(f)
        self.base_feature_cols = self.metadata["base_feature_cols"]
        self.feature_cols = self.metadata["feature_cols"]
        self.rolling_window = self.metadata["rolling_window"]
        self.rul_clip = self.metadata["rul_clip"]
        self.model_name = self.metadata["best_model"]

    def _build_feature_row(self, sensor_values: dict) -> pd.DataFrame:
        missing = [c for c in self.base_feature_cols if c not in sensor_values]
        if missing:
            raise ValueError(f"Missing required sensor readings: {missing}")

        row = {c: float(sensor_values[c]) for c in self.base_feature_cols}
        for c in self.base_feature_cols:
            row[f"{c}_rollmean{self.rolling_window}"] = row[c]
            row[f"{c}_rollstd{self.rolling_window}"] = 0.0
        return pd.DataFrame([row])[self.feature_cols]

    def predict(self, sensor_values: dict) -> dict:
        X = self._build_feature_row(sensor_values)
        X_scaled = self.scaler.transform(X.values)
        pred = float(self.model.predict(X_scaled)[0])
        pred = max(0.0, min(pred, self.rul_clip))

        if pred <= 20:
            status, urgency = "critical", "high"
        elif pred <= 50:
            status, urgency = "warning", "medium"
        else:
            status, urgency = "healthy", "low"

        return {
            "predicted_rul_cycles": round(pred, 1),
            "status": status,
            "urgency": urgency,
            "model_used": self.model_name,
        }

    def explain(self, sensor_values: dict, top_k: int = 6) -> list:
        """Return the top-k features driving this specific prediction via
        SHAP (tree-model exact explainer)."""
        import shap

        X = self._build_feature_row(sensor_values)
        X_scaled = self.scaler.transform(X.values)
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_scaled)[0]
        pairs = sorted(
            zip(self.feature_cols, shap_values), key=lambda p: abs(p[1]), reverse=True
        )[:top_k]
        return [
            {"feature": name, "impact": round(float(val), 3)} for name, val in pairs
        ]


_predictor: RULPredictor | None = None


def get_predictor() -> RULPredictor:
    global _predictor
    if _predictor is None:
        _predictor = RULPredictor()
    return _predictor
