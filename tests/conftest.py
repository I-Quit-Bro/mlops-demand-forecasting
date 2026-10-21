import numpy as np
import pandas as pd
import pytest

from forecast.config import Config, DataConfig, ServingConfig, TrainingConfig


def make_raw_df(days: int = 1100, seed: int = 0) -> pd.DataFrame:
    """Synthetic sales: trend + yearly + weekly pattern, two stores, small noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=days, freq="D")
    t = np.arange(days)
    base = (
        1000
        + 0.5 * t
        + 200 * np.sin(2 * np.pi * t / 365.25)
        + 150 * np.sin(2 * np.pi * dates.dayofweek / 7)
    )
    frames = []
    for store in (1, 2):
        sales = np.maximum((base / 2 + rng.normal(0, 10, days)).round(), 0)
        frames.append(pd.DataFrame({"date": dates, "store": store, "sales": sales}))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def raw_csv(tmp_path):
    path = tmp_path / "train.csv"
    make_raw_df().to_csv(path, index=False)
    return path


@pytest.fixture
def config(tmp_path, raw_csv):
    return Config(
        data=DataConfig(raw_path=raw_csv, date_column="date", target_column="sales"),
        training=TrainingConfig(
            holdout_days=90,
            ridge_alpha=0.01,
            lags=(1, 7, 14, 28, 364),
            rolling_windows=(7, 28),
        ),
        serving=ServingConfig(history_path=raw_csv),
        artifacts_dir=tmp_path / "artifacts",
    )


@pytest.fixture
def daily_series(config):
    from forecast.components.data_ingestion import DataIngestion

    return DataIngestion(config.data).run()
