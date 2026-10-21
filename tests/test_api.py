import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from forecast.api.main import create_app  # noqa: E402
from forecast.pipeline.train_pipeline import run_training  # noqa: E402
from forecast.registry import ModelRegistry  # noqa: E402


def test_health_is_up_even_without_a_model(config):
    with TestClient(create_app(config)) as client:
        assert client.get("/health").status_code == 200


def test_not_ready_and_forecast_503_without_production_model(config):
    with TestClient(create_app(config)) as client:
        assert client.get("/ready").status_code == 503
        assert client.get("/forecast").status_code == 503
        assert client.get("/model").status_code == 503


def test_forecast_and_model_endpoints(config):
    run_training(config, version="v1")
    ModelRegistry(config.artifacts_dir).set_stage("production", "v1")
    with TestClient(create_app(config)) as client:
        assert client.get("/ready").status_code == 200

        body = client.get("/forecast", params={"horizon": 3}).json()
        assert body["version"] == "v1"
        assert len(body["forecast"]) == 3
        assert body["forecast"][0]["predicted_sales"] > 0

        info = client.get("/model").json()
        assert info["version"] == "v1"
        assert info["metrics"]["mae"] > 0


@pytest.mark.parametrize("horizon", [0, 31, -1])
def test_invalid_horizon_is_rejected(config, horizon):
    run_training(config, version="v1")
    ModelRegistry(config.artifacts_dir).set_stage("production", "v1")
    with TestClient(create_app(config)) as client:
        assert client.get("/forecast", params={"horizon": horizon}).status_code == 422


def test_request_id_header_is_returned(config):
    with TestClient(create_app(config)) as client:
        response = client.get("/health", headers={"X-Request-ID": "abc123"})
        assert response.headers["X-Request-ID"] == "abc123"
