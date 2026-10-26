import os

import pandas as pd
import pytest

from src.data import load_cmapss, build_test_labels, add_rolling_features, RUL_CLIP

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@pytest.fixture(scope="module")
def dataset():
    return load_cmapss(DATA_DIR, "FD001")


def test_load_cmapss_shapes(dataset):
    assert len(dataset.train) == 20631
    assert dataset.train["unit_number"].nunique() == 100
    assert len(dataset.test_rul) == 100


def test_train_rul_is_clipped(dataset):
    assert dataset.train["RUL"].max() <= RUL_CLIP
    assert dataset.train["RUL"].min() >= 0


def test_train_rul_decreases_within_unit(dataset):
    unit1 = dataset.train[dataset.train["unit_number"] == 1].sort_values("time_in_cycles")
    diffs = unit1["RUL"].diff().dropna()
    # RUL should be non-increasing as cycles progress
    assert (diffs <= 0).all()


def test_constant_sensors_excluded_from_features(dataset):
    from src.data import CONSTANT_SENSORS
    assert not any(c in dataset.feature_cols for c in CONSTANT_SENSORS)


def test_build_test_labels_one_row_per_unit(dataset):
    labeled = build_test_labels(dataset.test, dataset.test_rul)
    assert len(labeled) == dataset.test["unit_number"].nunique()
    assert "RUL" in labeled.columns
    assert labeled["RUL"].min() >= 0


def test_add_rolling_features_adds_expected_columns(dataset):
    cols = ["sensor_2", "sensor_3"]
    out = add_rolling_features(dataset.train.head(50), cols, window=5)
    assert "sensor_2_rollmean5" in out.columns
    assert "sensor_3_rollstd5" in out.columns
    assert not out["sensor_3_rollstd5"].isna().any()
