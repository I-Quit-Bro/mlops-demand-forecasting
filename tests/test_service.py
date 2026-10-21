from dataclasses import replace

import pytest

from forecast.api.service import ModelService
from forecast.exception import ModelNotFoundError
from forecast.pipeline.train_pipeline import run_training
from forecast.registry import ModelRegistry


def _train_two_versions(config):
    run_training(config, version="v1")
    other = replace(config, training=replace(config.training, ridge_alpha=1000.0))
    run_training(other, version="v2")
    return ModelRegistry(config.artifacts_dir)


def test_not_ready_without_production_model(config):
    service = ModelService(config)
    assert service.is_ready() is False
    with pytest.raises(ModelNotFoundError, match="No production model"):
        service.forecast(3)


def test_forecast_uses_production_version(config):
    registry = _train_two_versions(config)
    registry.set_stage("production", "v1")
    service = ModelService(config)
    out = service.forecast(5)
    assert out["version"] == "v1"
    assert len(out["forecast"]) == 5
    assert out["history_end"] == service.history.index[-1].date()


def test_promotion_and_rollback_switch_models_without_restart(config):
    registry = _train_two_versions(config)
    registry.set_stage("production", "v1")
    service = ModelService(config)
    assert service.forecast(1)["version"] == "v1"

    registry.set_stage("production", "v2")  # promote
    assert service.forecast(1)["version"] == "v2"

    registry.set_stage("production", "v1")  # roll back
    assert service.forecast(1)["version"] == "v1"


def test_broken_new_version_keeps_serving_old_one(config):
    registry = _train_two_versions(config)
    registry.set_stage("production", "v1")
    service = ModelService(config)
    assert service.forecast(1)["version"] == "v1"

    (config.artifacts_dir / "models" / "v2" / "metadata.json").write_text("{broken")
    registry.set_stage("production", "v2")
    assert service.forecast(1)["version"] == "v1"


def test_model_info_has_metrics(config):
    registry = _train_two_versions(config)
    registry.set_stage("production", "v1")
    info = ModelService(config).model_info()
    assert info["version"] == "v1"
    assert info["metrics"]["mae"] > 0
    assert info["n_features"] == 27
