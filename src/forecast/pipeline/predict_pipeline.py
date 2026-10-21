"""Load a saved model and produce day-ahead forecasts from daily history."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from forecast.components.data_transformation import make_features
from forecast.exception import DataTransformationError, ForecastError, ModelLoadError
from forecast.logger import get_logger
from forecast.utils import load_json, load_object

log = get_logger(__name__)


@dataclass(frozen=True)
class ModelBundle:
    version: str
    model: Pipeline
    metadata: dict[str, Any]
    metrics: dict[str, float]


def load_bundle(model_dir: Path) -> ModelBundle:
    """Load model + metadata + metrics and check they belong together."""
    model_dir = Path(model_dir)
    model = load_object(model_dir / "model.joblib")
    metadata = load_json(model_dir / "metadata.json")
    metrics = load_json(model_dir / "metrics.json")

    expected = metadata.get("feature_columns")
    actual = list(getattr(model, "feature_names_in_", []))
    if expected != actual:
        raise ModelLoadError(
            "Model features do not match its metadata", version=model_dir.name
        )
    return ModelBundle(version=model_dir.name, model=model, metadata=metadata, metrics=metrics)


def forecast_next_days(bundle: ModelBundle, history: pd.Series, horizon: int) -> pd.DataFrame:
    """Recursive forecast: predict tomorrow, append it, repeat `horizon` times.

    Uses the same `make_features` as training, so features cannot drift apart.
    """
    if horizon < 1:
        raise ForecastError("horizon must be at least 1", horizon=horizon)

    lags = tuple(bundle.metadata["lags"])
    windows = tuple(bundle.metadata["rolling_windows"])
    columns = bundle.metadata["feature_columns"]
    needed = max(max(lags), max(windows, default=0)) + 1

    if len(history) < needed:
        raise DataTransformationError(
            "History too short to build forecast features", n_rows=len(history), required=needed
        )

    series = history.iloc[-needed:].astype(float)
    rows: list[tuple[Any, float]] = []
    for _ in range(horizon):
        next_date = series.index[-1] + pd.Timedelta(days=1)
        placeholder = pd.Series([np.nan], index=pd.DatetimeIndex([next_date]))
        extended = pd.concat([series, placeholder])
        features = make_features(extended, lags, windows).iloc[[-1]][columns]
        if features.isna().any().any():
            raise DataTransformationError("Incomplete features for forecast", date=str(next_date))

        prediction = float(bundle.model.predict(features)[0])
        if not np.isfinite(prediction):
            raise ForecastError("Model produced a non-finite prediction", date=str(next_date))
        prediction = max(prediction, 0.0)  # demand cannot be negative

        rows.append((next_date.date(), prediction))
        series = pd.concat([series, pd.Series([prediction], index=pd.DatetimeIndex([next_date]))])
        series = series.iloc[-needed:]

    return pd.DataFrame(rows, columns=["date", "predicted_sales"])
