import datetime as dt

import pytest

from forecast.exception import DataTransformationError, ForecastError, ModelLoadError
from forecast.pipeline.predict_pipeline import forecast_next_days, load_bundle
from forecast.pipeline.train_pipeline import run_training
from forecast.utils import save_json


def test_forecast_shape_dates_and_range(config, daily_series):
    out_dir = run_training(config, version="v1")
    bundle = load_bundle(out_dir)
    result = forecast_next_days(bundle, daily_series, horizon=14)

    assert list(result.columns) == ["date", "predicted_sales"]
    assert len(result) == 14
    assert result["date"].iloc[0] == daily_series.index[-1].date() + dt.timedelta(days=1)
    gaps = {(b - a).days for a, b in zip(result["date"], result["date"][1:])}
    assert gaps == {1}
    assert (result["predicted_sales"] >= 0).all()

    recent_mean = daily_series.iloc[-28:].mean()
    assert (result["predicted_sales"] > 0.5 * recent_mean).all()
    assert (result["predicted_sales"] < 1.5 * recent_mean).all()


def test_short_history_raises(config, daily_series):
    bundle = load_bundle(run_training(config, version="v1"))
    with pytest.raises(DataTransformationError, match="too short"):
        forecast_next_days(bundle, daily_series.iloc[-100:], horizon=3)


def test_bad_horizon_raises(config, daily_series):
    bundle = load_bundle(run_training(config, version="v1"))
    with pytest.raises(ForecastError, match="at least 1"):
        forecast_next_days(bundle, daily_series, horizon=0)


def test_metadata_mismatch_is_rejected(config):
    out_dir = run_training(config, version="v1")
    save_json({"feature_columns": ["wrong"]}, out_dir / "metadata.json")
    with pytest.raises(ModelLoadError, match="do not match"):
        load_bundle(out_dir)
