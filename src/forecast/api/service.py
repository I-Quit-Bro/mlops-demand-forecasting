"""Model-serving logic, kept free of FastAPI so it is easy to test and debug.

The service re-reads the registry's `production` pointer on every request (one tiny
file read) and hot-swaps the model when the pointer changes. That means a promotion
or a rollback takes effect on the next request, with no restart and no admin endpoint.
"""
from __future__ import annotations

import os
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from forecast.components.data_ingestion import DataIngestion
from forecast.config import Config
from forecast.exception import ForecastError, ModelNotFoundError
from forecast.logger import get_logger
from forecast.pipeline.predict_pipeline import ModelBundle, forecast_next_days, load_bundle
from forecast.registry import ModelRegistry
from forecast.utils import timed

log = get_logger(__name__)

HISTORY_ENV_VAR = "FORECAST_HISTORY_PATH"


class ModelService:
    def __init__(self, config: Config) -> None:
        self.registry = ModelRegistry(config.artifacts_dir)
        self._lock = threading.Lock()
        self._bundle: ModelBundle | None = None
        self._failed_versions: set[str] = set()

        history_path = Path(os.getenv(HISTORY_ENV_VAR, str(config.serving.history_path)))
        # Fail fast at startup: a service without history cannot forecast anything.
        self.history: pd.Series = DataIngestion(
            replace(config.data, raw_path=history_path)
        ).run()
        log.info("History ready: %d days ending %s", len(self.history), self.history.index[-1].date())

    def _current_bundle(self) -> ModelBundle:
        version = self.registry.get_stage("production")
        if version is None:
            raise ModelNotFoundError("No production model is registered")

        bundle = self._bundle
        if bundle is not None and (bundle.version == version or version in self._failed_versions):
            return bundle

        with self._lock:
            bundle = self._bundle
            if bundle is not None and bundle.version == version:
                return bundle
            try:
                new_bundle = load_bundle(self.registry.model_dir(version))
            except ForecastError as exc:
                self._failed_versions.add(version)
                if bundle is None:
                    raise
                log.error(
                    "Could not load production version %s, still serving %s: %s",
                    version, bundle.version, exc,
                )
                return bundle
            self._failed_versions.discard(version)
            self._bundle = new_bundle
            log.info(
                "Now serving model version=%s (previous: %s)",
                new_bundle.version, bundle.version if bundle else None,
            )
            return new_bundle

    def is_ready(self) -> bool:
        try:
            self._current_bundle()
        except ForecastError:
            return False
        return True

    @timed
    def forecast(self, horizon: int) -> dict[str, Any]:
        bundle = self._current_bundle()
        result = forecast_next_days(bundle, self.history, horizon)
        return {
            "version": bundle.version,
            "history_end": self.history.index[-1].date(),
            "horizon": horizon,
            "forecast": result.to_dict(orient="records"),
        }

    def model_info(self) -> dict[str, Any]:
        bundle = self._current_bundle()
        meta = bundle.metadata
        return {
            "version": bundle.version,
            "git_sha": meta.get("git_sha"),
            "created_at": meta.get("created_at"),
            "train_start": meta.get("train_start"),
            "train_end": meta.get("train_end"),
            "test_start": meta.get("test_start"),
            "test_end": meta.get("test_end"),
            "n_features": len(meta.get("feature_columns", [])),
            "metrics": bundle.metrics,
            "history_end": self.history.index[-1].date(),
        }
