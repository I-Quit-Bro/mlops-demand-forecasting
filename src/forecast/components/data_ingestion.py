"""Read raw sales data, validate it, and turn it into one daily series."""
from __future__ import annotations

import pandas as pd

from forecast.config import DataConfig
from forecast.exception import DataIngestionError, DataValidationError
from forecast.logger import get_logger
from forecast.utils import timed

log = get_logger(__name__)


class DataIngestion:
    def __init__(self, config: DataConfig) -> None:
        self.config = config

    @timed
    def load_raw(self) -> pd.DataFrame:
        path = self.config.raw_path
        if not path.exists():
            raise DataIngestionError("Raw data file not found", path=str(path))
        try:
            df = pd.read_csv(path, parse_dates=[self.config.date_column])
        except (ValueError, OSError) as exc:  # includes parser and empty-file errors
            raise DataIngestionError("Could not read raw data", path=str(path)) from exc
        log.info("Loaded %d rows from %s", len(df), path)
        return df

    @timed
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        date_col, target_col = self.config.date_column, self.config.target_column

        missing = {date_col, target_col} - set(df.columns)
        if missing:
            raise DataValidationError(
                "Missing required columns", missing=sorted(missing), found=list(df.columns)
            )
        if df.empty:
            raise DataValidationError("Raw data is empty")
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            raise DataValidationError("Date column could not be parsed as dates", column=date_col)
        if df[date_col].isna().any():
            raise DataValidationError("Null dates found", count=int(df[date_col].isna().sum()))
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            raise DataValidationError("Target column is not numeric", column=target_col)
        if df[target_col].isna().any():
            raise DataValidationError("Null target values", count=int(df[target_col].isna().sum()))
        if (df[target_col] < 0).any():
            raise DataValidationError(
                "Negative target values", count=int((df[target_col] < 0).sum())
            )
        return df

    @timed
    def to_daily_series(self, df: pd.DataFrame) -> pd.Series:
        """Sum the target per day and require a gap-free daily index."""
        date_col, target_col = self.config.date_column, self.config.target_column
        daily = df.groupby(date_col)[target_col].sum().sort_index()

        expected = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        missing_dates = expected.difference(daily.index)
        if len(missing_dates) > 0:
            raise DataValidationError(
                "Missing dates in daily series",
                n_missing=len(missing_dates),
                first_missing=str(missing_dates[0].date()),
            )

        daily = daily.asfreq("D")
        daily.name = target_col
        log.info(
            "Daily series: %d days, %s to %s",
            len(daily), daily.index.min().date(), daily.index.max().date(),
        )
        return daily

    def run(self) -> pd.Series:
        df = self.validate(self.load_raw())
        return self.to_daily_series(df)
