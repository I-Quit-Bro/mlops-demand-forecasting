"""Small shared helpers: timing decorator, versioning, safe file IO."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar
from zoneinfo import ZoneInfo
import joblib

from forecast.exception import ForecastError, ModelNotFoundError
from forecast.logger import get_logger

F = TypeVar("F", bound=Callable[..., Any])


def _emit(log: logging.Logger, level: int, func: Callable[..., Any], msg: str, *args: Any) -> None:
    """Log as if the call came from `func`'s own file/line, not from this wrapper.

    Without this, every START/DONE line would show a line number inside utils.py,
    which is misleading when debugging.
    """
    if not log.isEnabledFor(level):
        return
    code = func.__code__
    record = log.makeRecord(
        log.name, level, code.co_filename, code.co_firstlineno, msg, args, None, func=func.__name__
    )
    log.handle(record)


def timed(func: F) -> F:
    """Log START / DONE / FAILED with duration for the wrapped function.

    Failures are logged as one line here; the full traceback is logged once
    at the pipeline entry point, so nested steps don't repeat it.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        log = get_logger(func.__module__)
        start = time.perf_counter()
        _emit(log, logging.INFO, func, "START %s", func.__qualname__)
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            _emit(
                log, logging.ERROR, func, "FAILED %s after %.2fs: %s",
                func.__qualname__, time.perf_counter() - start, exc,
            )
            raise
        _emit(log, logging.INFO, func, "DONE  %s in %.2fs",
              func.__qualname__, time.perf_counter() - start)
        return result

    return wrapper  # type: ignore[return-value]


def get_git_sha() -> str:
    """Short commit hash: from CI env var, then git, else 'nogit'."""
    ci_sha = os.getenv("GITHUB_SHA")
    if ci_sha:
        return ci_sha[:7]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return out.stdout.strip() or "nogit"
    except (OSError, subprocess.SubprocessError):
        return "nogit"


def make_model_version() -> str:
    """Sortable, traceable version id, e.g. 20261015-143205-ab12cd3."""
    stamp = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{get_git_sha()}"


def save_json(obj: Any, path: Path) -> None:
    try:
        Path(path).write_text(json.dumps(obj, indent=2, default=str))
    except (OSError, TypeError) as exc:
        raise ForecastError("Could not write JSON file", path=str(path)) from exc


def load_json(path: Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise ModelNotFoundError("JSON file not found", path=str(path))
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ForecastError("Could not read JSON file", path=str(path)) from exc


def save_object(obj: Any, path: Path) -> None:
    try:
        joblib.dump(obj, path)
    except OSError as exc:
        raise ForecastError("Could not write object", path=str(path)) from exc


def load_object(path: Path) -> Any:
    """Load a joblib file. Only load files you produced or trust."""
    path = Path(path)
    if not path.exists():
        raise ModelNotFoundError("Model file not found", path=str(path))
    try:
        return joblib.load(path)
    except Exception as exc:
        raise ForecastError("Could not load object", path=str(path)) from exc
