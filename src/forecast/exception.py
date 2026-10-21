"""Project-specific exceptions.

Every error carries optional keyword context (paths, counts, names) that is
appended to the message, so a log line alone tells you what went wrong.
Raise with `raise X("message", key=value) from original_exc` to keep the
original traceback chained.
"""
from __future__ import annotations


class ForecastError(Exception):
    """Base class for all errors raised by this project."""

    def __init__(self, message: str, **context: object) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def __str__(self) -> str:
        if not self.context:
            return self.message
        details = ", ".join(f"{key}={value!r}" for key, value in self.context.items())
        return f"{self.message} [{details}]"


class ConfigError(ForecastError):
    """Config file missing, malformed, or has invalid values."""


class DataIngestionError(ForecastError):
    """Raw data could not be found or read."""


class DataValidationError(ForecastError):
    """Raw data was read but failed quality checks."""


class DataTransformationError(ForecastError):
    """Feature engineering or splitting failed."""


class ModelTrainingError(ForecastError):
    """Training, evaluation, or saving a model failed."""


class ModelNotFoundError(ForecastError):
    """A requested model artifact does not exist."""


class RegistryError(ForecastError):
    """Registry file is unreadable or an invalid stage/pointer was requested."""


class ModelLoadError(ForecastError):
    """A model artifact exists but is inconsistent or cannot be used."""
