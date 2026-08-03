import pandas as pd
from src.models.train import (
    split_data,
    train_model,
    evaluate_model,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)

def make_fake_features_df(n=50):
    import numpy as np
    data = {col: np.random.rand(n) * 100 for col in FEATURE_COLUMNS}
    data[TARGET_COLUMN] = np.random.rand(n) * 1000

    return pd.DataFrame(data)

def test_split_preserves_time_order():
    df = make_fake_features_df()
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)

    assert X_train.index.max() < X_test.index.min()

def test_train_model_returns_fitted_model():
    df = make_fake_features_df()
    X_train, X_test, y_train, y_test = split_data(df)

    model = train_model(X_train, y_train, params={"n_estimators": 10, "max_depth": 3})
    assert hasattr(model, "predict")

    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test)

def test_evaluate_model_returns_expected_metrics():
    df = make_fake_features_df()

    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train, params={"n_estimators": 10, "max_depth": 3})
    metrics = evaluate_model(model, X_test, y_test)

    assert set(metrics.keys()) == {"rmse", "mae", "r2"}
    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0