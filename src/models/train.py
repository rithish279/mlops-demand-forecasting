import mlflow
import mlflow.xgboost
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

FEATURE_COLUMNS = [
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14",
    "rolling_mean_7", "rolling_mean_14",
    "rolling_std_7", "rolling_std_14", 
    "day_of_week", "week_of_year", "month", "quarter",
    "is_weekend", "is_month_start", "is_month_end",
    "holiday", "workingday", "weathersit", "temp", "hum", "windspeed",
]

TARGET_COLUMN = "next_day_sales"

DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": 42,
}

def load_features(path: str = "data/processed/features.csv") -> pd.DataFrame:
    return pd.read_csv(path)

def split_data(df: pd.DataFrame, test_size: float = 0.2):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    return train_test_split(X, y, test_size=test_size, shuffle=False)

def train_model(X_train, y_train, params: dict = None) -> XGBRegressor:
    params = params or DEFAULT_PARAMS
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: XGBRegressor, X_test, y_test) -> dict:
    predictions = model.predict(X_test)

    return {
        "rmse": mean_squared_error(y_test, predictions) ** 0.5,
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }

def run_training(params: dict = None):
    params = params or DEFAULT_PARAMS
    df = load_features()

    X_train, X_test, y_train, y_test = split_data(df)

    with mlflow.start_run():
        model = train_model(X_train, y_train, params)
        metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_metric("rmse", metrics["rmse"])
        mlflow.log_metric("mae", metrics["mae"])
        mlflow.log_metric("r2", metrics["r2"])
        mlflow.log_param("feature_count", len(FEATURE_COLUMNS))
        mlflow.xgboost.log_model(model, "model")

        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"MAE:  {metrics['mae']:.2f}")
        print(f"R2:   {metrics['r2']:.4f}")

if __name__ == "__main__":
    run_training()