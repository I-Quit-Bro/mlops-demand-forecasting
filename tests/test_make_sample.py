from dataclasses import replace

import pandas as pd
import pytest

from forecast.exception import DataValidationError
from forecast.pipeline.make_sample import make_sample
from forecast.pipeline.train_pipeline import run_training


def test_sample_is_small_and_trainable(config, tmp_path):
    out = make_sample(config, tmp_path / "sample" / "daily_sample.csv", days=730)
    sample = pd.read_csv(out)
    assert list(sample.columns) == ["date", "sales"]
    assert len(sample) == 730

    small = replace(config, data=replace(config.data, raw_path=out))
    run_training(small, version="from-sample")  # CI will do exactly this


def test_sample_too_small_is_rejected(config, tmp_path):
    with pytest.raises(DataValidationError, match="too small"):
        make_sample(config, tmp_path / "s.csv", days=100)


def test_sample_larger_than_data_is_rejected(config, tmp_path):
    with pytest.raises(DataValidationError, match="fewer days"):
        make_sample(config, tmp_path / "s.csv", days=5000)
