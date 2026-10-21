"""End-to-end training: ingest -> transform -> train -> evaluate -> save.

CLI:  python -m forecast.pipeline.train_pipeline [--config PATH] [--data PATH]
      (or the `forecast-train` command after `pip install -e .`)
Exit codes: 0 ok, 1 expected project error, 2 unexpected error.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from forecast.components.data_ingestion import DataIngestion
from forecast.components.data_transformation import DataTransformation
from forecast.components.model_trainer import ModelTrainer
from forecast.config import Config, load_config
from forecast.exception import ForecastError
from forecast.logger import get_logger
from forecast.utils import get_git_sha, make_model_version

log = get_logger(__name__)


def run_training(config: Config, version: str | None = None) -> Path:
    version = version or make_model_version()
    log.info("Starting training run, version=%s", version)

    series = DataIngestion(config.data).run()

    transformation = DataTransformation(config.training)
    X, y = transformation.transform(series)
    X_train, X_test, y_train, y_test = transformation.split(X, y)

    trainer = ModelTrainer(config.training, config.artifacts_dir)
    model = trainer.train(X_train, y_train)
    metrics = trainer.evaluate(model, X_test, y_test)

    metadata = {
        "version": version,
        "git_sha": get_git_sha(),
        "created_at": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
        "data_path": str(config.data.raw_path),
        "feature_columns": list(X.columns),
        "ridge_alpha": config.training.ridge_alpha,
        "holdout_days": config.training.holdout_days,
        "lags": list(config.training.lags),
        "rolling_windows": list(config.training.rolling_windows),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "train_start": X_train.index.min().date().isoformat(),
        "train_end": X_train.index.max().date().isoformat(),
        "test_start": X_test.index.min().date().isoformat(),
        "test_end": X_test.index.max().date().isoformat(),
    }
    out_dir = trainer.save(model, metrics, metadata, version)
    log.info("Training finished: version=%s mae=%.2f", version, metrics["mae"])
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the demand forecasting model")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--data", default=None, help="Override data.raw_path from the config")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        if args.data:
            config = replace(config, data=replace(config.data, raw_path=Path(args.data)))
        run_training(config)
    except ForecastError as exc:
        log.error("Training failed: %s", exc)
        if exc.__cause__ is not None:
            log.error("Caused by: %r", exc.__cause__)
        log.debug("Full traceback", exc_info=True)  # visible with LOG_LEVEL=DEBUG
        return 1
    except Exception:
        log.exception("Unexpected error during training")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
