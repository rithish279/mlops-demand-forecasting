import pandas as pd

def add_lag_features(df: pd.DataFrame, target_col: str = "cnt") -> pd.DataFrame:
    df = df.copy()
    for lag in [1, 2, 3, 7, 14]:
        df[f"lag_{lag}"] = df[target_col].shift(lag)

    return df

def add_rolling_features(df: pd.DataFrame, target_col: str = "cnt") -> pd.DataFrame:
    df = df.copy()
    for window in [7, 14]:
        df[f"rolling_mean_{window}"] = df[target_col].shift(1).rolling(window).mean()
        df[f"rolling_std_{window}"] = df[target_col].shift(1).rolling(window).std()

    return df

def add_calender_features(df: pd.DataFrame, date_col: str = "dteday") ->pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["day_of_week"] = df[date_col].dt.day_of_week
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
    df["month"] = df[date_col].dt.month
    df["quarter"] = df[date_col].dt.quarter
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_month_start"] = df[date_col].dt.is_month_start.astype(int)
    df["is_month_end"] = df[date_col].dt.is_month_end.astype(int)

    return df

def add_target(df: pd.DataFrame, target_col: str = "cnt") -> pd.DataFrame:
    df = df.copy()
    df["next_day_sales"] = df[target_col].shift(-1)

    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_calender_features(df)
    df = add_target(df)

    df = df.dropna().reset_index(drop=True)
    return df

if __name__ == "__main__":
    from src.data.validation import run_validation

    df = run_validation("data/raw/day.csv")
    features_df = engineer_features(df)

    print(f"Engineered features: {len(features_df)} rows, {len(features_df.columns)} columns")
    print(features_df[["dteday", "cnt", "lag_1", "rolling_mean_7", "day_of_week", "next_day_sales"]].head())

    features_df.to_csv("data/processed/features.csv", index=False)
    print("Saved to data/processed/features.csv")