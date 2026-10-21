"""Request and response models for the HTTP API."""
import datetime as dt

from pydantic import BaseModel

MAX_HORIZON = 30  # recursive forecasts get unreliable beyond about a month


class ForecastPoint(BaseModel):
    date: dt.date
    predicted_sales: float


class ForecastResponse(BaseModel):
    version: str
    history_end: dt.date
    horizon: int
    forecast: list[ForecastPoint]


class ModelInfo(BaseModel):
    version: str
    git_sha: str | None = None
    created_at: str | None = None
    train_start: str | None = None
    train_end: str | None = None
    test_start: str | None = None
    test_end: str | None = None
    n_features: int
    metrics: dict[str, float]
    history_end: dt.date
