"""Feature engineering helpers shared by training and inference."""
from __future__ import annotations

from src.data import add_rolling_features

ROLLING_WINDOW = 5


def engineer_features(df, feature_cols, window: int = ROLLING_WINDOW):
    """Add rolling-window statistics on top of raw sensor/setting readings.

    Returns the augmented dataframe and the final ordered list of feature
    column names to feed into a model.
    """
    augmented = add_rolling_features(df, feature_cols, window=window)
    rolling_cols = [f"{c}_rollmean{window}" for c in feature_cols] + [
        f"{c}_rollstd{window}" for c in feature_cols
    ]
    final_cols = feature_cols + rolling_cols
    return augmented, final_cols
