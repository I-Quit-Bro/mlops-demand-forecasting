"""Typed configuration loaded from YAML.

Run commands from the repo root, or point to a file with the
FORECAST_CONFIG environment variable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # pyright: ignore[reportMissingModuleSource]

from forecast.exception import ConfigError

DEFAULT_CONFIG_PATH = "configs/config.yaml"
CONFIG_ENV_VAR = "FORECAST_CONFIG"


@dataclass(frozen=True)
class DataConfig:
    raw_path: Path
    date_column: str
    target_column: str
    source_url: str | None = None  # where `forecast-fetch-data` downloads raw_path from
    sha256: str | None = None 


@dataclass(frozen=True)
class TrainingConfig:
    holdout_days: int
    ridge_alpha: float
    lags: tuple[int, ...]
    rolling_windows: tuple[int, ...]


@dataclass(frozen=True)
class ServingConfig:
    history_path: Path  # daily history the API uses to build features


@dataclass(frozen=True)
class Config:
    data: DataConfig
    training: TrainingConfig
    serving: ServingConfig
    artifacts_dir: Path


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    section = raw.get(name)
    if not isinstance(section, dict):
        raise ConfigError("Config section missing or not a mapping", section=name)
    return section


def _require(section: dict[str, Any], key: str, section_name: str) -> Any:
    if key not in section:
        raise ConfigError("Missing config key", section=section_name, key=key)
    return section[key]


def _validate(cfg: Config) -> None:
    t = cfg.training
    if t.holdout_days <= 0:
        raise ConfigError("holdout_days must be positive", value=t.holdout_days)
    if t.ridge_alpha < 0:
        raise ConfigError("ridge_alpha must be >= 0", value=t.ridge_alpha)
    if not t.lags or any(lag <= 0 for lag in t.lags):
        raise ConfigError("lags must be a non-empty list of positive ints", value=t.lags)
    if 7 not in t.lags:
        raise ConfigError("lags must include 7 (seasonal-naive baseline uses lag_7)", value=t.lags)
    if any(w <= 0 for w in t.rolling_windows):
        raise ConfigError("rolling_windows must be positive ints", value=t.rolling_windows)


def load_config(path: str | Path | None = None) -> Config:
    config_path = Path(path or os.getenv(CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH))
    if not config_path.exists():
        raise ConfigError("Config file not found", path=str(config_path))
    try:
        raw = yaml.safe_load(config_path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError("Config file is not valid YAML", path=str(config_path)) from exc
    if not isinstance(raw, dict):
        raise ConfigError("Config root must be a mapping", path=str(config_path))

    data = _section(raw, "data")
    training = _section(raw, "training")
    serving = raw.get("serving") or {}
    if not isinstance(serving, dict):
        raise ConfigError("Config section is not a mapping", section="serving")
    try:
        cfg = Config(
            data=DataConfig(
                raw_path=Path(_require(data, "raw_path", "data")),
                date_column=str(_require(data, "date_column", "data")),
                target_column=str(_require(data, "target_column", "data")),
            ),
            training=TrainingConfig(
                holdout_days=int(_require(training, "holdout_days", "training")),
                ridge_alpha=float(_require(training, "ridge_alpha", "training")),
                lags=tuple(int(x) for x in _require(training, "lags", "training")),
                rolling_windows=tuple(
                    int(x) for x in _require(training, "rolling_windows", "training")
                ),
            ),
            serving=ServingConfig(
                history_path=Path(serving.get("history_path", data["raw_path"]))
            ),
            artifacts_dir=Path(raw.get("artifacts_dir", "artifacts")),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError("Invalid value type in config", path=str(config_path)) from exc

    _validate(cfg)
    return cfg
