import pytest

from forecast.components.data_transformation import DataTransformation, make_features
from forecast.exception import DataTransformationError


def test_feature_columns_do_not_depend_on_data_length(daily_series):
    lags, windows = (1, 7, 14), (7,)
    long_cols = list(make_features(daily_series, lags, windows).columns)
    short_cols = list(make_features(daily_series.iloc[:60], lags, windows).columns)
    assert long_cols == short_cols


def test_features_use_only_past_values(daily_series):
    feats = make_features(daily_series, (1, 7), (7,))
    t = 50
    assert feats["lag_1"].iloc[t] == daily_series.iloc[t - 1]
    assert feats["lag_7"].iloc[t] == daily_series.iloc[t - 7]
    assert feats["roll_7"].iloc[t] == pytest.approx(daily_series.iloc[t - 7 : t].mean())


def test_split_is_chronological(config, daily_series):
    tr = DataTransformation(config.training)
    X, y = tr.transform(daily_series)
    X_train, X_test, y_train, y_test = tr.split(X, y)
    assert len(X_test) == config.training.holdout_days
    assert X_train.index.max() < X_test.index.min()
    assert len(X_train) == len(y_train) and len(X_test) == len(y_test)
    assert not X.isna().any().any()


def test_too_short_series_raises(config, daily_series):
    with pytest.raises(DataTransformationError, match="too short"):
        DataTransformation(config.training).transform(daily_series.iloc[:200])
