import pytest
import os
import pandas as pd
from src.data.feature_engineering import (
    add_target,
    add_calender_features,
    add_rolling_features,
    add_lag_features,
    engineer_features
)

def make_sample_df():
    dates = pd.date_range("2011-01-01", periods=20, freq="D")

    return pd.DataFrame({
        "dteday": dates,
        "cnt": range(100, 120),
    })

def test_adding_lag_features_shifts_correctly():
    df = add_lag_features(make_sample_df())

    assert df.loc[1, "lag_1"] == df.loc[0, "cnt"]

def test_rolling_mean_excludes_current_day():
    df = add_rolling_features(make_sample_df())
    row = 10

    expected_mean = make_sample_df()["cnt"].iloc[row - 7:row].mean()

    assert abs(df.loc[row, "rolling_mean_7"] - expected_mean) < 1e-6

def test_calender_features_are_derived_correctly():
    df = add_calender_features(make_sample_df())

    assert df.loc[0, "day_of_week"] == pd.Timestamp("2011-01-01").day_of_week
    assert set(df["is_weekend"].unique()).issubset({0, 1})

def test_target_is_shifted_forward():
    df = add_target(make_sample_df())

    assert df.loc[0, "next_day_sales"] == df.loc[1, "cnt"]

def test_engineer_feature_drops_incomplete_rows():
    df = engineer_features(make_sample_df())

    assert df.isnull().sum().sum() == 0
    assert len(df) < len(make_sample_df())

def test_feature_columns_are_numeric():
    df = engineer_features(make_sample_df())
    feature_cols = [
        "lag_1", "lag_2", "lag_3", "lag_7", "lag_14",
        "rolling_mean_7", "rolling_mean_14",
        "rolling_std_7", "rolling_std_14",
        "day_of_week", "week_of_year", "month", "quarter",
        "is_weekend", "is_month_start", "is_month_end",
    ]
    for col in feature_cols:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is not numeric"

@pytest.mark.skipif(not os.path.exists("data/raw/day.csv"), reason="Full dataset not available in CI")
def test_engineer_features_on_real_data():
    from src.data.validation import run_validation

    raw = run_validation("data/raw/day.csv")
    features_df = engineer_features(raw)

    assert "next_day_sales" in features_df.columns
    assert features_df.isnull().sum().sum() == 0
