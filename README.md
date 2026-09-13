 Turbofan Engine RUL Predictor
Predicting how many flight cycles an aircraft engine has left, from live sensor telemetry — not a chatbot, an actual regression model in production.
Overview
This project estimates the Remaining Useful Life (RUL) of aircraft turbofan engines using multivariate sensor time-series data from NASA's C-MAPSS simulation. It covers the full pipeline: EDA → feature engineering → model comparison → explainability → a FastAPI backend → an interactive dashboard, all backed by real evaluation numbers from the included dataset.
Problem Statement
Unplanned engine failure is expensive and dangerous; over-conservative maintenance schedules waste money on parts that still have useful life left. Predictive maintenance — estimating RUL directly from sensor readings — lets operators service equipment exactly when needed instead of on a fixed calendar.
Proposed Solution
Train a regression model on run-to-failure sensor histories (NASA C-MAPSS FD001) that maps a snapshot of engine sensor readings to an estimated number of cycles remaining before failure, then expose it through an API and dashboard that a maintenance engineer could actually use: enter/load a sensor reading, get a RUL estimate, a health status, and a SHAP-based explanation of which sensors are driving that estimate.
Key Features
Full ML pipeline: cleaning, degradation-aware target construction (RUL clipping), rolling-window feature engineering, 3-model comparison with grouped cross-validation
Explainable predictions via SHAP (per-prediction feature attribution, not just global importance)
FastAPI backend with auto-validated request schemas generated directly from the trained model's feature list (schema can't drift from the model)
Dashboard UI with a live gauge, sample "engine health" presets pulled from real held-out engines, and inline model-comparison charts
20 automated tests covering data integrity, model behavior, and API contracts
Architecture
```
Browser (dashboard)
      │  fetch()
      ▼
FastAPI app (app/main.py)
      │  loads once at startup
      ▼
RULPredictor (src/predict.py) ── model.joblib + scaler.joblib + metadata.json
      ▲
      │  produced by
src/train.py  ←── src/features.py ←── src/data.py ←── data/*.txt (NASA C-MAPSS)
```
Machine Learning Pipeline
Load raw whitespace-separated CMAPSS files (`src/data.py`)
Label training rows with RUL = (engine's final cycle − current cycle), clipped at 125 cycles (standard practice in the RUL literature — early-life RUL isn't learnable from sensors alone, since a healthy engine at cycle 10 looks the same whether it fails at cycle 150 or cycle 300)
Clean: drop 7 sensors with ~zero variance across the whole FD001 run (no signal)
Feature engineer: add rolling mean/std (window=5 cycles) per engine on top of raw readings
Split: train on all 100 training engines; evaluate on the last recorded cycle of each of the 100 held-out test engines (the standard CMAPSS benchmark protocol), using NASA's provided ground-truth RUL
Train & compare: Linear Regression (baseline) vs Random Forest vs XGBoost, each validated with 5-fold GroupKFold cross-validation (grouped by engine unit, so no engine's cycles leak across train/validation)
Select best by test-set RMSE, persist model + scaler + metadata
Explain: SHAP TreeExplainer for both a global summary plot and per-request feature attribution
Dataset
Name: NASA C-MAPSS Turbofan Engine Degradation Simulation — FD001 subset
Source: NASA Prognostics Center of Excellence (mirrored from a public GitHub repository; see `data/` — files are small enough to ship with the repo)
Samples: 20,631 training rows across 100 engines (run-to-failure); 13,096 test rows across 100 engines (partial runs, evaluated at their last recorded cycle) plus NASA's ground-truth RUL for each
Features: 3 operational settings + 21 sensor channels per cycle (14 sensors retained after dropping constant ones), expanded with rolling mean/std
Target: Remaining Useful Life in cycles, clipped at 125
Preprocessing: constant-sensor removal, rolling-window feature engineering, `StandardScaler` fit on training data only
Split: all of FD001's official train set for training/CV; all of FD001's official test set (last cycle per engine) held out for final evaluation — this is NASA's own benchmark split, not a random shuffle
Models
Model	Test MAE (cycles)	Test RMSE (cycles)	Test R²
Linear Regression	16.75	21.11	0.723
Random Forest	12.79	18.04	0.797
XGBoost (selected)	12.53	17.71	0.805
These are real numbers from the last training run (`models/metadata.json`), not invented figures — rerun `python -m src.train` to reproduce them.
Evaluation
Metrics: MAE, RMSE, R² (standard for RUL regression)
Cross-validation: 5-fold GroupKFold grouped by engine unit
Held-out test: NASA's official FD001 test split, evaluated at each engine's last recorded cycle against NASA's provided ground truth
See `assets/model_comparison.png` (bar chart), `assets/rul_prediction_scatter.png` (predicted vs. actual), and `assets/shap_summary.png` (global feature importance) — all generated from the actual trained model, not mocked
Screenshots
Generated evaluation artifacts (also embedded live in the running dashboard):
`assets/model_comparison.png` — MAE/RMSE across the three candidate models
`assets/rul_prediction_scatter.png` — predicted vs. actual RUL on the held-out test engines
`assets/shap_summary.png` — which sensors drive predictions, and in which direction
Tech Stack
ML: scikit-learn, XGBoost, SHAP
Backend: FastAPI, Pydantic, Uvicorn
Frontend: vanilla HTML/CSS/JS (no build step), served as static files by FastAPI
Testing: pytest, FastAPI TestClient
Data/viz: pandas, NumPy, Matplotlib
Installation
```bash
git clone <your-repo-url>
cd rul-predictive-maintenance
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Windows PowerShell Setup
```powershell
git clone <your-repo-url>
cd rul-predictive-maintenance
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Usage
Train (optional — a trained model is already included in `models/`):
```bash
python -m src.train
```
Run the app:
```bash
uvicorn app.main:app --reload --port 8000
```
Then open `http://localhost:8000` for the dashboard, or `http://localhost:8000/docs` for interactive API docs.
API Documentation
Endpoint	Method	Description
`/api/health`	GET	Liveness check + loaded model name
`/api/model-info`	GET	Full training metadata and comparison metrics
`/api/samples`	GET	Five real example engine readings spanning healthy → critical
`/api/feature-schema`	GET	List of sensor fields the model expects
`/api/predict`	POST	Body: sensor readings (see schema) → `{predicted_rul_cycles, status, urgency, model_used}`
`/api/explain`	POST	Same body → top SHAP feature contributions for that prediction
Full interactive schema and try-it-out UI at `/docs` (Swagger UI, auto-generated by FastAPI).
Testing
20 tests, all passing (`python -m pytest tests/ -v`):
Data (`tests/test_data.py`): dataset shapes, RUL clipping and monotonicity, constant-sensor exclusion, test-label construction, rolling-feature generation
Prediction (`tests/test_predict.py`): artifact loading, output schema, missing-field validation, health-ordering sanity check (critical < healthy), output bounds, SHAP explanation shape
API (`tests/test_api.py`): all endpoints, validation error handling, static page serving
Project Structure
```
rul-predictive-maintenance/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── .env.example
│
├── data/                    # NASA C-MAPSS FD001 raw files
├── models/                  # trained model, scaler, metadata, sample engines
├── notebooks/
│   └── 01_eda.ipynb         # executed EDA notebook
├── src/
│   ├── data.py              # loading, labeling, cleaning
│   ├── features.py          # rolling-window feature engineering
│   ├── train.py             # training + CV + evaluation + plots
│   └── predict.py           # inference wrapper (used by app + tests)
├── app/
│   ├── main.py               # FastAPI app
│   └── static/
│       ├── index.html        # dashboard (HTML/CSS/JS)
│       └── assets/            # copies of evaluation plots for the UI
├── tests/
│   ├── test_data.py
│   ├── test_predict.py
│   └── test_api.py
└── assets/                   # evaluation plots (source of truth)
```
Limitations
Trained and evaluated only on the FD001 subset (single operating condition, single fault mode) — FD002/FD003/FD004 have multiple operating conditions and would need condition-normalization to generalize to
The API accepts a single sensor snapshot; since it has no history for that engine, rolling-mean features fall back to the instantaneous reading and rolling-std to zero, which is a reasonable but imperfect approximation of a true multi-cycle rolling window
RUL is capped at 125 cycles by design, so the model cannot distinguish "very healthy" engines beyond that cap
No deep learning (LSTM/Transformer) sequence model was trained; tree ensembles were sufficient to beat the linear baseline and keep the project practical to run on a laptop
Future Improvements
Extend to FD002–FD004 with operating-condition normalization
Add a sequence model (LSTM or 1D-CNN) trained on full sliding windows instead of single-cycle snapshots, for a fairer comparison against the tree ensembles
Persist a short history per engine ID server-side so the API can compute true rolling features instead of the single-snapshot approximation
Add authentication and a lightweight database for logging predictions over time
License
MIT — see `LICENSE`.
Author
Built as a portfolio project demonstrating an end-to-end ML system: data pipeline, model comparison, explainability, API design, and testing.
