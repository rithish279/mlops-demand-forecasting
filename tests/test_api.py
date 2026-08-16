import pytest
import os
from fastapi.testclient import TestClient
from api.main import app

MLFLOW_AVAILABLE = os.environ.get("MLFLOW_TRACKING_URI") is not None

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

@pytest.mark.skipif(
        not MLFLOW_AVAILABLE,
        reason="Requires a running MLflow server with promoted production model"
)
def test_predict_returns_valid_response(client):
    payload = {
        "lag_1": 100, "lag_2": 105, "lag_3": 110, "lag_7": 120, "lag_14": 115,
        "rolling_mean_7": 108.5, "rolling_mean_14": 112.0,
        "rolling_std_7": 5.2, "rolling_std_14": 6.1,
        "day_of_week": 2, "week_of_year": 15, "month": 4, "quarter": 2,
        "is_weekend": 0, "is_month_start": 0, "is_month_end": 0,
        "holiday": 0, "workingday": 1, "weathersit": 1,
        "temp": 0.5, "hum": 0.6, "windspeed": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "predicted_demand" in body
    assert isinstance(body["predicted_demand"], float)


def test_predict_rejects_missing_fields(client):
    response = client.post("/predict", json={"lag_1": 100})
    assert response.status_code == 422