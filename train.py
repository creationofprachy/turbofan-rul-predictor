"""
End-to-end training pipeline for the RUL prediction models.

Run with:  python -m src.train
Produces (all paths relative to the project root):
    models/best_model.joblib
    models/scaler.joblib
    models/metadata.json
    assets/model_comparison.png
    assets/shap_summary.png
    assets/rul_prediction_scatter.png
"""
from __future__ import annotations

import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.data import load_cmapss, build_test_labels
from src.features import engineer_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
ASSETS_DIR = os.path.join(ROOT, "assets")
SUBSET = "FD001"


def evaluate(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "R2": round(r2, 4)}


def cross_validate(model_ctor, X, y, groups, n_splits=5):
    """GroupKFold CV so that the same engine unit never appears in both
    the train and validation fold (avoids leakage across cycles of the
    same engine)."""
    gkf = GroupKFold(n_splits=n_splits)
    fold_scores = []
    for train_idx, val_idx in gkf.split(X, y, groups):
        model = model_ctor()
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        fold_scores.append(evaluate(y[val_idx], preds))
    avg = {k: round(float(np.mean([f[k] for f in fold_scores])), 3) for k in fold_scores[0]}
    return avg


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)

    print(f"Loading {SUBSET} dataset ...")
    ds = load_cmapss(DATA_DIR, SUBSET)
    train_raw, test_raw, rul_df = ds.train, ds.test, ds.test_rul
    base_feature_cols = ds.feature_cols

    print("Engineering rolling-window features ...")
    train_feat, feature_cols = engineer_features(train_raw, base_feature_cols)
    test_feat_full, _ = engineer_features(test_raw, base_feature_cols)
    test_feat = build_test_labels(test_feat_full, rul_df)

    X_train_raw = train_feat[feature_cols].values
    y_train = train_feat["RUL"].values.astype(float)
    groups = train_feat["unit_number"].values

    X_test_raw = test_feat[feature_cols].values
    y_test = test_feat["RUL"].values.astype(float)

    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    candidates = {
        "LinearRegression": lambda: LinearRegression(),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        ),
        "XGBoost": lambda: XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=-1, objective="reg:squarederror",
        ),
    }

    results = {}
    fitted_models = {}
    print("\nRunning 5-fold GroupKFold cross-validation on the training set ...")
    for name, ctor in candidates.items():
        t0 = time.time()
        cv_scores = cross_validate(ctor, X_train, y_train, groups)
        model = ctor()
        model.fit(X_train, y_train)
        test_preds = model.predict(X_test)
        test_scores = evaluate(y_test, test_preds)
        elapsed = round(time.time() - t0, 1)
        print(f"  {name:15s} CV(train)={cv_scores}  Test={test_scores}  ({elapsed}s)")
        results[name] = {"cv": cv_scores, "test": test_scores}
        fitted_models[name] = model

    best_name = min(results, key=lambda k: results[k]["test"]["RMSE"])
    best_model = fitted_models[best_name]
    print(f"\nBest model on held-out test set: {best_name}")

    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))

    metadata = {
        "subset": SUBSET,
        "best_model": best_name,
        "feature_cols": feature_cols,
        "base_feature_cols": base_feature_cols,
        "rul_clip": 125,
        "rolling_window": 5,
        "results": results,
        "n_train_rows": int(len(train_feat)),
        "n_test_units": int(len(test_feat)),
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # ---- Visualizations -------------------------------------------------
    plt.figure(figsize=(7, 4.5))
    names = list(results.keys())
    rmses = [results[n]["test"]["RMSE"] for n in names]
    maes = [results[n]["test"]["MAE"] for n in names]
    x = np.arange(len(names))
    width = 0.35
    plt.bar(x - width / 2, rmses, width, label="RMSE")
    plt.bar(x + width / 2, maes, width, label="MAE")
    plt.xticks(x, names)
    plt.ylabel("Cycles")
    plt.title(f"Model comparison on held-out test set ({SUBSET})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "model_comparison.png"), dpi=150)
    plt.close()

    best_preds = best_model.predict(X_test)
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y_test, best_preds, alpha=0.6, edgecolor="k", linewidth=0.3)
    lims = [0, max(y_test.max(), best_preds.max()) + 5]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlabel("Actual RUL (cycles)")
    plt.ylabel("Predicted RUL (cycles)")
    plt.title(f"{best_name}: Predicted vs Actual RUL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "rul_prediction_scatter.png"), dpi=150)
    plt.close()

    # SHAP explainability (tree models only — fast, exact TreeExplainer)
    if best_name in ("RandomForest", "XGBoost"):
        try:
            import shap
            explainer = shap.TreeExplainer(best_model)
            sample = X_test[: min(200, len(X_test))]
            shap_values = explainer.shap_values(sample)
            plt.figure()
            shap.summary_plot(
                shap_values, sample, feature_names=feature_cols, show=False, max_display=12
            )
            plt.tight_layout()
            plt.savefig(os.path.join(ASSETS_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
            plt.close()
            print("Saved SHAP summary plot.")
        except Exception as e:  # pragma: no cover - visualization is best-effort
            print(f"SHAP plot generation skipped: {e}")

    print("\nSaved: models/best_model.joblib, models/scaler.joblib, models/metadata.json")
    print("Saved: assets/model_comparison.png, assets/rul_prediction_scatter.png")


if __name__ == "__main__":
    main()
