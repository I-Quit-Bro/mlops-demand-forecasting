
import pytest

from forecast.config import load_config
from forecast.exception import ConfigError

VALID_YAML = """
data:
  raw_path: data/raw/train.csv
  date_column: date
  target_column: sales
training:
  holdout_days: 90
  ridge_alpha: 0.01
  lags: [1, 7, 14]
  rolling_windows: [7]
artifacts_dir: out
"""


def test_load_valid_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(VALID_YAML)
    cfg = load_config(path)
    assert cfg.training.lags == (1, 7, 14)
    assert cfg.artifacts_dir.name == "out"


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_missing_key_raises(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(VALID_YAML.replace("  ridge_alpha: 0.01\n", ""))
    with pytest.raises(ConfigError, match="Missing config key"):
        load_config(path)


def test_lags_without_7_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(VALID_YAML.replace("[1, 7, 14]", "[1, 14]"))
    with pytest.raises(ConfigError, match="include 7"):
        load_config(path)
