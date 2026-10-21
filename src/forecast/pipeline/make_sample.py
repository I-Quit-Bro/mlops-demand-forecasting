"""Create the small committed sample used by CI from the full raw dataset.

The full Kaggle file lives in data/raw/ (gitignored) and is too big for git and CI.
This writes the last N days of *daily totals* (columns: date, sales), a few KB.

CLI:  forecast-make-sample [--days 730] [--out data/sample/daily_sample.csv]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from forecast.components.data_ingestion import DataIngestion
from forecast.components.data_transformation import DataTransformation
from forecast.config import Config, load_config
from forecast.exception import DataValidationError, ForecastError
from forecast.logger import get_logger

log = get_logger(__name__)

DEFAULT_OUT = "data/sample/daily_sample.csv"
DEFAULT_DAYS = 730


def make_sample(config: Config, out_path: Path, days: int = DEFAULT_DAYS) -> Path:
    minimum = DataTransformation(config.training).min_series_length
    if days < minimum:
        raise DataValidationError(
            "Sample too small for the configured lags and hold-out", days=days, minimum=minimum
        )
    series = DataIngestion(config.data).run()
    if len(series) < days:
        raise DataValidationError("Raw data has fewer days than requested", have=len(series), want=days)

    sample = series.iloc[-days:].rename_axis(config.data.date_column).reset_index()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_path, index=False)
    log.info("Wrote %d-day sample to %s", days, out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create the committed CI sample dataset")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default=None, help="Override data.raw_path")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.data:
            config = replace(config, data=replace(config.data, raw_path=Path(args.data)))
        make_sample(config, Path(args.out), args.days)
    except ForecastError as exc:
        log.error("Could not create sample: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
