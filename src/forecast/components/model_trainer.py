"""Train the model, evaluate it against a baseline, and save versioned artifacts."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from forecast.config import TrainingConfig
from forecast.exception import ModelTrainingError
from forecast.logger import get_logger
from forecast.utils import save_json, save_object, timed

log = get_logger(__name__)


class ModelTrainer:
    def __init__(self, config: TrainingConfig, artifacts_dir: Path) -> None:
        self.config = config
        self.artifacts_dir = Path(artifacts_dir)

    def build_model(self) -> Pipeline:
        return make_pipeline(StandardScaler(), Ridge(alpha=self.config.ridge_alpha))

    @timed
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        if X_train.empty or len(X_train) != len(y_train):
            raise ModelTrainingError(
                "Invalid training data", n_features_rows=len(X_train), n_targets=len(y_train)
            )
        model = self.build_model()
        try:
            model.fit(X_train, y_train)
        except Exception as exc:
            raise ModelTrainingError("Model fitting failed") from exc
        return model

    @timed
    def evaluate(
        self, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict[str, float]:
        """MAE/RMSE on the hold-out set, plus the seasonal-naive (lag_7) baseline."""
        preds = model.predict(X_test)
        finite = np.isfinite(preds)
        if not finite.all():
            raise ModelTrainingError(
                "Model produced non-finite predictions", n_bad=int((~finite).sum())
            )
        errors = y_test.to_numpy() - preds
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors**2)))
        baseline_mae = float(np.mean(np.abs(y_test.to_numpy() - X_test["lag_7"].to_numpy())))
        metrics = {
            "mae": mae,
            "rmse": rmse,
            "baseline_mae": baseline_mae,
            "improvement_vs_baseline_pct": 100 * (baseline_mae - mae) / baseline_mae,
        }
        log.info("Hold-out metrics: %s", {k: round(v, 2) for k, v in metrics.items()})
        return metrics

    @timed
    def save(
        self, model: Pipeline, metrics: dict[str, Any], metadata: dict[str, Any], version: str
    ) -> Path:
        """Write model + metrics + metadata to artifacts/models/<version>/.

        Files go to a temp folder first and are renamed in one step, so a failed
        run never leaves a half-written version. Existing versions are never overwritten.
        """
        final_dir = self.artifacts_dir / "models" / version
        if final_dir.exists():
            raise ModelTrainingError("Model version already exists", path=str(final_dir))
        tmp_dir = final_dir.with_name(final_dir.name + ".tmp")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True)
        try:
            save_object(model, tmp_dir / "model.joblib")
            save_json(metrics, tmp_dir / "metrics.json")
            save_json(metadata, tmp_dir / "metadata.json")
            tmp_dir.rename(final_dir)
        except Exception as exc:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise ModelTrainingError("Failed to save model artifacts", path=str(final_dir)) from exc
        log.info("Saved model artifacts to %s", final_dir)
        return final_dir
