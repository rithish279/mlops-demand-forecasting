from src.data.validation import run_validation
from src.data.feature_engineering import engineer_features
from src.models.train import split_data, train_model, evaluate_model

def test_full_pipeline_on_sample_data():
    df = run_validation("data/sample/day_sample.csv")
    features_df = engineer_features(df)

    assert len(features_df) > 0, "Feature engineering produced no usable rows"

    X_train, X_test, y_train, y_test = split_data(features_df, test_size=0.2)
    
    assert len(X_train) > 0
    assert len(X_test) > 0

    model = train_model(X_train, y_train, params={"n_estimators": 10, "max_depth": 3})
    metrics = evaluate_model(model, X_test, y_test)

    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0