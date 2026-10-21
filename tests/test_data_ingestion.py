from dataclasses import replace

import pandas as pd
import pytest

from forecast.components.data_ingestion import DataIngestion
from forecast.exception import DataIngestionError, DataValidationError


def test_run_returns_gap_free_daily_series(config):
    series = DataIngestion(config.data).run()
    assert series.index.freqstr == "D"
    assert len(series) == 1100
    assert (series > 0).all()


def test_missing_file_raises(config):
    bad = replace(config.data, raw_path=config.data.raw_path.parent / "nope.csv")
    with pytest.raises(DataIngestionError, match="not found"):
        DataIngestion(bad).run()


def test_negative_target_raises(config):
    df = pd.read_csv(config.data.raw_path)
    df.loc[0, "sales"] = -5
    df.to_csv(config.data.raw_path, index=False)
    with pytest.raises(DataValidationError, match="Negative"):
        DataIngestion(config.data).run()


def test_missing_dates_raise(config):
    df = pd.read_csv(config.data.raw_path)
    df = df[df["date"] != "2020-03-01"]
    df.to_csv(config.data.raw_path, index=False)
    with pytest.raises(DataValidationError, match="Missing dates"):
        DataIngestion(config.data).run()


def test_missing_target_column_raises(config):
    df = pd.read_csv(config.data.raw_path).drop(columns="sales")
    df.to_csv(config.data.raw_path, index=False)
    with pytest.raises(DataValidationError, match="Missing required columns"):
        DataIngestion(config.data).run()
