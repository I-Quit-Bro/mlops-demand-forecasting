import json

import pytest

from forecast.exception import ModelNotFoundError, RegistryError
from forecast.pipeline.train_pipeline import run_training
from forecast.registry import ModelRegistry, main

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


def test_set_and_get_stage(config):
    run_training(config, version="v1")
    registry = ModelRegistry(config.artifacts_dir)
    assert registry.get_stage("production") is None
    registry.set_stage("production", "v1")
    assert registry.get_stage("production") == "v1"
    assert json.loads(registry.registry_file.read_text()) == {"production": "v1"}


def test_unknown_stage_rejected(config):
    run_training(config, version="v1")
    with pytest.raises(RegistryError, match="Unknown stage"):
        ModelRegistry(config.artifacts_dir).set_stage("staging", "v1")


def test_unknown_version_rejected(config):
    with pytest.raises(ModelNotFoundError, match="not found"):
        ModelRegistry(config.artifacts_dir).set_stage("production", "ghost")


def test_list_versions_ignores_unfinished_dirs(config):
    run_training(config, version="v1")
    (config.artifacts_dir / "models" / "v2.tmp").mkdir()
    assert ModelRegistry(config.artifacts_dir).list_versions() == ["v1"]


def test_corrupt_registry_file_raises(config):
    registry = ModelRegistry(config.artifacts_dir)
    registry.root.mkdir(parents=True)
    registry.registry_file.write_text("{not json")
    with pytest.raises(RegistryError, match="unreadable"):
        registry.get_stage("production")


def test_cli_set_and_show(tmp_path, config, raw_csv):
    run_training(config, version="v1")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(CONFIG_YAML.format(raw=raw_csv, out=config.artifacts_dir))
    assert main(["--config", str(cfg), "set", "production", "v1"]) == 0
    assert main(["--config", str(cfg), "show"]) == 0
    assert main(["--config", str(cfg), "set", "production", "ghost"]) == 1
    assert ModelRegistry(config.artifacts_dir).get_stage("production") == "v1"
