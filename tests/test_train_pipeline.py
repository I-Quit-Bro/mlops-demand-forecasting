
import pytest

from forecast.exception import ModelTrainingError
from forecast.pipeline.train_pipeline import main, run_training
from forecast.utils import load_json, load_object

CONFIG_YAML = """
data:
  raw_path: {raw}
  date_column: date
  target_column: sales
training:
  holdout_days: 90
  ridge_alpha: 0.01
  lags: [1, 7, 14, 28, 364]
  rolling_windows: [7, 28]
artifacts_dir: {out}
"""


def test_end_to_end_creates_artifacts(config):
    out_dir = run_training(config, version="test-v1")
    assert out_dir.name == "test-v1"
    for name in ("model.joblib", "metrics.json", "metadata.json"):
        assert (out_dir / name).exists()

    metrics = load_json(out_dir / "metrics.json")
    assert metrics["mae"] < metrics["baseline_mae"]

    metadata = load_json(out_dir / "metadata.json")
    model = load_object(out_dir / "model.joblib")
    assert list(model.feature_names_in_) == metadata["feature_columns"]


def test_existing_version_is_never_overwritten(config):
    run_training(config, version="dup")
    with pytest.raises(ModelTrainingError, match="already exists"):
        run_training(config, version="dup")


def test_cli_returns_1_on_missing_data(tmp_path, raw_csv):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG_YAML.format(raw=raw_csv, out=tmp_path / "art"))
    assert main(["--config", str(cfg), "--data", str(tmp_path / "nope.csv")]) == 1


def test_cli_returns_0_on_success(tmp_path, raw_csv):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG_YAML.format(raw=raw_csv, out=tmp_path / "art"))
    assert main(["--config", str(cfg)]) == 0
    assert any((tmp_path / "art" / "models").iterdir())
