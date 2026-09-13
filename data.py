"""
Data loading and preprocessing for the NASA C-MAPSS Turbofan Engine
Remaining Useful Life (RUL) prediction task (FD001 sub-dataset).

The raw files are whitespace-separated text files with no header:
    unit_number, time_in_cycles, op_setting_1..3, sensor_1..21
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column layout of the raw CMAPSS files
# ---------------------------------------------------------------------------
INDEX_COLS = ["unit_number", "time_in_cycles"]
SETTING_COLS = [f"op_setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# Sensors that are constant (or near-constant) for FD001 and carry no
# discriminative signal for RUL estimation. Identified during EDA
# (zero or near-zero variance across the whole training set).
CONSTANT_SENSORS = [
    "sensor_1", "sensor_5", "sensor_6", "sensor_10",
    "sensor_16", "sensor_18", "sensor_19",
]

# Cap applied to the training RUL target. Early in an engine's life the
# degradation signal is flat, so the true RUL is not learnable from the
# sensor readings alone. Clipping to a max (a standard trick in the RUL
# literature, e.g. Heimes 2008) turns the target into a piecewise-linear
# curve: constant while healthy, then linearly decreasing near failure.
RUL_CLIP = 125


@dataclass
class CMAPSSDataset:
    """Container for a processed train/test split of one FD00x subset."""

    train: pd.DataFrame
    test: pd.DataFrame
    test_rul: pd.DataFrame
    feature_cols: list = field(default_factory=list)


def _read_raw_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(ALL_COLS)]
    df.columns = ALL_COLS
    return df


def _compute_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a RUL column to the training data.

    For training data every unit runs to failure, so RUL at a given cycle
    is simply (max cycle for that unit) - (current cycle), clipped to
    RUL_CLIP.
    """
    max_cycle = df.groupby("unit_number")["time_in_cycles"].transform("max")
    rul = max_cycle - df["time_in_cycles"]
    df = df.copy()
    df["RUL"] = rul.clip(upper=RUL_CLIP)
    return df


def load_cmapss(data_dir: str, subset: str = "FD001") -> CMAPSSDataset:
    """Load and label the train/test/RUL files for a CMAPSS subset."""
    train_path = os.path.join(data_dir, f"train_{subset}.txt")
    test_path = os.path.join(data_dir, f"test_{subset}.txt")
    rul_path = os.path.join(data_dir, f"RUL_{subset}.txt")

    train_df = _read_raw_file(train_path)
    test_df = _read_raw_file(test_path)
    rul_df = pd.read_csv(rul_path, header=None, names=["RUL"])

    train_df = _compute_train_rul(train_df)

    feature_cols = [c for c in SETTING_COLS + SENSOR_COLS if c not in CONSTANT_SENSORS]

    return CMAPSSDataset(train=train_df, test=test_df, test_rul=rul_df, feature_cols=feature_cols)


def build_test_labels(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
    """The public test set only records partial run-to-failure histories.

    The ground-truth RUL at the *last observed cycle* of each test unit is
    given in RUL_FD001.txt. We only need to evaluate models on that last
    cycle (this mirrors the standard CMAPSS benchmark protocol), so we
    take the final row per unit and attach the known RUL.
    """
    last_cycle = test_df.groupby("unit_number")["time_in_cycles"].idxmax()
    last_rows = test_df.loc[last_cycle].reset_index(drop=True)
    last_rows["RUL"] = rul_df["RUL"].clip(upper=RUL_CLIP).values
    return last_rows


def add_rolling_features(df: pd.DataFrame, feature_cols: list, window: int = 5) -> pd.DataFrame:
    """Add rolling mean/std features per engine unit to capture short-term
    degradation trends, in addition to the raw instantaneous sensor values.
    """
    df = df.sort_values(["unit_number", "time_in_cycles"]).copy()
    grouped = df.groupby("unit_number")[feature_cols]
    roll_mean = grouped.rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
    roll_std = grouped.rolling(window=window, min_periods=1).std().reset_index(level=0, drop=True).fillna(0.0)
    roll_mean.columns = [f"{c}_rollmean{window}" for c in feature_cols]
    roll_std.columns = [f"{c}_rollstd{window}" for c in feature_cols]
    return pd.concat([df, roll_mean, roll_std], axis=1)
