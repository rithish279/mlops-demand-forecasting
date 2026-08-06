from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.models.registry import get_production_model
from src.models.train import FEATURE_COLUMNS

import pandas as pd

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = get_production_model()
        print("Production model loaded successfully")
    except Exception as e:
        print(f"Warning: could not load production model at startup: {e}")
        model = None

    yield


app = FastAPI(title="Bike Demand Forecasting API", lifespan=lifespan)


class PredictionRequest(BaseModel):
    lag_1: float
    lag_2: float
    lag_3: float
    lag_7: float
    lag_14: float
    rolling_mean_7: float
    rolling_mean_14: float
    rolling_std_7: float
    rolling_std_14: float
    day_of_week: int
    week_of_year: int
    month: int
    quarter: int
    is_weekend: int
    is_month_start: int
    is_month_end: int
    holiday: int
    workingday: int
    weathersit: int
    temp: float
    hum: float
    windspeed: float


class PredictionResponse(BaseModel):
    predicted_demand: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    input_df = pd.DataFrame([request.model_dump()])[FEATURE_COLUMNS]
    prediction = model.predict(input_df)[0]

    return PredictionResponse(predicted_demand=round(float(prediction), 1))