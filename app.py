from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from prophet import Prophet
import os
import asyncio

app = FastAPI(title="Accident Prediction API")

# Model configuration
MODEL_FILE = "prophet_model.joblib"
model = None

def initialize_model():
    """Initialize and return a Prophet model with sample data"""
    print("Initializing new Prophet model...")
    dates = pd.date_range(start='2010-01-01', periods=120, freq='ME')  # Changed 'M' to 'ME'
    data = pd.DataFrame({
        'ds': dates,
        'y': [100 + i*2 + (i%7)*5 for i in range(len(dates))]
    })
    model = Prophet()
    model.fit(data)
    joblib.dump(model, MODEL_FILE)
    return model

@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = joblib.load(MODEL_FILE) if os.path.exists(MODEL_FILE) else initialize_model()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to initialize model: {str(e)}")
        raise

class PredictionRequest(BaseModel):
    year: int
    month: int

@app.post("/predict", response_model=dict)
async def predict(request: PredictionRequest):
    """Make prediction for given year and month"""
    try:
        future_date = f"{request.year}-{request.month:02d}-01"
        future_df = pd.DataFrame({'ds': [pd.to_datetime(future_date)]})
        forecast = model.predict(future_df)
        return {"prediction": float(forecast['yhat'].iloc[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/", response_model=dict)
async def root():
    return {
        "message": "Accident Prediction API is running",
        "endpoints": {
            "documentation": "/docs",
            "prediction": "/predict (POST)"
        }
    }

# Keep the application running
async def keep_alive():
    while True:
        await asyncio.sleep(1)

@app.on_event("startup")
async def run_keep_alive():
    asyncio.create_task(keep_alive())