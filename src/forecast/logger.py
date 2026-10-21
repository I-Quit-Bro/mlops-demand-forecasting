"""Central logging setup.

Usage:  log = get_logger(__name__)

Logs go to stdout (captured by CI runners and containers) and to a rotating
file. Control behaviour with environment variables:
    LOG_LEVEL          DEBUG | INFO (default) | WARNING | ERROR
    FORECAST_LOG_DIR   directory for app.log (default: logs)
"""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured, avoid duplicate handlers
        return logger

    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    logger.propagate = False
    formatter = logging.Formatter(_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_dir = Path(os.getenv("FORECAST_LOG_DIR", "logs"))
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "app.log", maxBytes=5_000_000, backupCount=3
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        logger.warning("File logging disabled, cannot write to %s", log_dir)

    return logger
