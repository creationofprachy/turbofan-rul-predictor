"""
FastAPI backend for the Turbofan Engine RUL (Remaining Useful Life)
Predictive Maintenance application.
"""
from __future__ import annotations

import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, create_model

from src.predict import get_predictor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(
    title="Turbofan Engine RUL Predictor API",
    description="Predicts Remaining Useful Life (RUL) of aircraft turbofan "
    "engines from live sensor readings, using the NASA C-MAPSS dataset.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = get_predictor()

# Build a Pydantic model dynamically from the actual feature columns the
# trained model expects, so the request schema can never drift from the
# model's real inputs.
_fields = {c: (float, Field(..., description=f"Sensor reading: {c}")) for c in predictor.base_feature_cols}
SensorReadings = create_model("SensorReadings", **_fields)


class PredictionResponse(BaseModel):
    predicted_rul_cycles: float
    status: str
    urgency: str
    model_used: str


class ExplanationItem(BaseModel):
    feature: str
    impact: float


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": predictor.model_name}


@app.get("/api/model-info")
def model_info():
    return predictor.metadata


@app.get("/api/samples")
def samples():
    path = os.path.join(MODELS_DIR, "sample_engines.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No sample engines available.")
    with open(path) as f:
        return json.load(f)


@app.get("/api/feature-schema")
def feature_schema():
    return {"required_fields": predictor.base_feature_cols}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(readings: SensorReadings):
    try:
        result = predictor.predict(readings.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return result


@app.post("/api/explain", response_model=list[ExplanationItem])
def explain(readings: SensorReadings):
    try:
        result = predictor.explain(readings.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return result


# Serve the frontend
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))
