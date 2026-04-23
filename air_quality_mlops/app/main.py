from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Suppress tf warnings if keras is used
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

app = FastAPI(title="PM2.5 Time-Series Prediction API", description="API to predict next hour PM2.5 levels based on past data")

MODEL_PKL_PATH = "models/best_model.pkl"
MODEL_KERAS_PATH = "models/best_model.keras"
SCALER_PATH = "models/scaler.pkl"

class AirQualityFeatures(BaseModel):
    pm25_lag_1: float
    pm25_lag_2: float
    pm25_lag_3: float

@app.get("/")
def home():
    return {"message": "Welcome to the PM2.5 Time-Series API", "status": "active"}

@app.post("/predict")
def predict_aqi(features: AirQualityFeatures):
    try:
        if not os.path.exists(SCALER_PATH):
            raise HTTPException(status_code=500, detail="Models not trained. Please run src/train.py first.")
            
        scaler = joblib.load(SCALER_PATH)
        
        # Prepare input data
        input_data = np.array([[
            features.pm25_lag_1, 
            features.pm25_lag_2, 
            features.pm25_lag_3
        ]])
        
        # Scale Data
        scaled_data = scaler.transform(input_data)
        
        # Determine which model is best and predicting
        if os.path.exists(MODEL_PKL_PATH):
            model = joblib.load(MODEL_PKL_PATH)
            prediction = model.predict(scaled_data)
            model_used = "SVR (Support Vector Regressor)"
        elif os.path.exists(MODEL_KERAS_PATH):
            from keras.models import load_model
            model = load_model(MODEL_KERAS_PATH)
            scaled_data_lstm = scaled_data.reshape(scaled_data.shape[0], 1, scaled_data.shape[1])
            prediction = model.predict(scaled_data_lstm, verbose=0).flatten()
            model_used = "LSTM"
        else:
            raise HTTPException(status_code=500, detail="No trained model found.")
        
        return {
            "prediction_pm25_next_hour": float(prediction[0]),
            "unit": "µg/m³",
            "model_used": model_used,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
