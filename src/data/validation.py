import pandas as pd

EXPECTED_COLUMNS = {
    "instant", "dteday", "season", "yr", "mnth", 
    "holiday", "weekday", "weathersit", "temp", "atemp",
    "hum", "windspeed", "casual", "registered", "cnt"
}

def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def validate_schema(df: pd.DataFrame) -> None:
    actual_columns = set(df.columns)
    missing = EXPECTED_COLUMNS - actual_columns
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    
def validate_no_missing_values(df: pd.DataFrame) -> None:
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]

    if not missing_cols.empty:
        raise ValueError(f"Found missing values: \n{missing_cols}")

def validate_no_duplicates(df: pd.DataFrame) -> None:
    duplicate_count = df.duplicated(subset=["dteday"]).sum()
    if duplicate_count > 0:
        raise ValueError(f"Found {duplicate_count} duplicate data rows")

def validate_target_range(df: pd.DataFrame) -> None:
    if (df["cnt"] < 0).any():
        raise ValueError("Found negative values in target column 'cnt'")

def run_validation(path: str) -> pd.DataFrame:
    df = load_raw_data(path)
    validate_schema(df)
    validate_no_missing_values(df)
    validate_no_duplicates(df) 
    validate_target_range(df)
    print(f"Validation passed: {len(df)} rows, {len(df.columns)} columns")
    return df

if __name__ == "__main__":
    run_validation("data/raw/day.csv")  