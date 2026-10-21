"""Feature engineering and the chronological train/hold-out split."""
from __future__ import annotations

import pandas as pd

from forecast.config import TrainingConfig
from forecast.exception import DataTransformationError
from forecast.logger import get_logger
from forecast.utils import timed

log = get_logger(__name__)

MIN_TRAIN_ROWS = 30


def make_features(
    series: pd.Series, lags: tuple[int, ...], rolling_windows: tuple[int, ...]
) -> pd.DataFrame:
    """Build features indexed like `series`.

    Every feature at day t uses only data from days before t (no leakage).
    Rows without enough history contain NaN; callers decide whether to drop them.
    Column set and order depend only on the arguments, so training and serving match.
    """
    feats = pd.DataFrame(index=series.index)
    feats["doy"] = series.index.dayofyear
    for lag in lags:
        feats[f"lag_{lag}"] = series.shift(lag)
    previous = series.shift(1)
    for window in rolling_windows:
        feats[f"roll_{window}"] = previous.rolling(window).mean()
    for day in range(7):
        feats[f"dow_{day}"] = (series.index.dayofweek == day).astype(float)
    for month in range(1, 13):
        feats[f"month_{month}"] = (series.index.month == month).astype(float)
    return feats


class DataTransformation:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    @property
    def min_series_length(self) -> int:
        history = max(max(self.config.lags), max(self.config.rolling_windows, default=0))
        return history + self.config.holdout_days + MIN_TRAIN_ROWS

    @timed
    def transform(self, series: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        if len(series) < self.min_series_length:
            raise DataTransformationError(
                "Series too short for the configured lags and hold-out",
                n_rows=len(series),
                required=self.min_series_length,
            )
        feats = make_features(series, self.config.lags, self.config.rolling_windows)
        data = feats.assign(y=series).dropna()
        X, y = data.drop(columns="y"), data["y"]
        log.info("Feature matrix: %d rows x %d features", *X.shape)
        return X, y

    @timed
    def split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Chronological split: the last `holdout_days` rows are the hold-out set."""
        h = self.config.holdout_days
        X_train, X_test = X.iloc[:-h], X.iloc[-h:]
        y_train, y_test = y.iloc[:-h], y.iloc[-h:]
        log.info(
            "Train: %d rows (%s to %s) | Hold-out: %d rows (%s to %s)",
            len(X_train), X_train.index.min().date(), X_train.index.max().date(),
            len(X_test), X_test.index.min().date(), X_test.index.max().date(),
        )
        return X_train, X_test, y_train, y_test
