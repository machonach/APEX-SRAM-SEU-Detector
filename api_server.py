from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import json
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import joblib

# Load ML model if available
try:
    sys.path.append('ml_pipeline')
    from anomaly_detection import AnomalyDetectionModel
    model = AnomalyDetectionModel()
    model_loaded = True
except Exception as e:
    print(f"Warning: Could not load anomaly detection model: {e}")
    model = None
    model_loaded = False

# FastAPI app
app = FastAPI(
    title="SEU Detector API",
    description="API for accessing SEU (Single Event Upset) detector data and predictions",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class SEUEvent(BaseModel):
    timestamp: str
    altitude: float
    temperature: float
    pressure: float
    bit_flips_count: int
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    cosmic_intensity: Optional[float] = None
    max_run_length: Optional[int] = None

class SEUPrediction(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    anomaly_probability: Optional[float] = None
    estimated_bit_flips: Optional[int] = None
    
class SEUEventWithPrediction(BaseModel):
    event: SEUEvent
    prediction: Optional[SEUPrediction] = None

# Load data helper function
def load_data(file_path):
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
    return None

# Routes
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "SEU Detector API",
        "version": "1.0.0",
        "status": "operational",
        "docs_url": "/docs",
        "model_loaded": model_loaded
    }

@app.get("/data/latest", response_model=SEUEvent, tags=["Data"])
async def get_latest_data():
    """Get the most recent SEU data point"""
    # Try real-time data first
    real_time_file = "real_time_seu_data.csv"
    df = load_data(real_time_file)
    
    # Fall back to synthetic data if needed
    if df is None:
        df = load_data("ml_pipeline/seu_synthetic_data.csv")
    
    if df is None:
        raise HTTPException(status_code=404, detail="No data available")
    
    # Return the latest data point
    latest = df.iloc[-1].to_dict()
    
    # Ensure timestamp is a string
    if isinstance(latest.get('timestamp'), pd.Timestamp):
        latest['timestamp'] = latest['timestamp'].isoformat()
    
    return latest

@app.get("/data/range", response_model=List[SEUEvent], tags=["Data"])
async def get_data_range(
    start: Optional[str] = Query(None, description="Start timestamp (ISO format)"),
    end: Optional[str] = Query(None, description="End timestamp (ISO format)"),
    limit: int = Query(100, description="Maximum number of records to return")
):
    """Get SEU data for a specified time range"""
    # First try real-time data
    real_time_file = "real_time_seu_data.csv"
    df = load_data(real_time_file)
    
    # Fall back to synthetic or analysis data
    if df is None:
        df = load_data("ml_pipeline/seu_analysis_results.csv")
    
    if df is None:
        df = load_data("ml_pipeline/seu_synthetic_data.csv")
    
    if df is None:
        raise HTTPException(status_code=404, detail="No data available")
    
    # Parse timestamps if they exist
    if 'timestamp' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except:
                pass
    
    # Filter by date range if timestamps are available
    if 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        if start:
            start_dt = pd.to_datetime(start)
            df = df[df['timestamp'] >= start_dt]
        
        if end:
            end_dt = pd.to_datetime(end)
            df = df[df['timestamp'] <= end_dt]
    
    # Limit records
    df = df.tail(limit)
    
    # Convert timestamps to strings
    if 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    
    # Convert to list of dicts
    result = df.to_dict('records')
    return result

@app.get("/analytics/summary", tags=["Analytics"])
async def get_summary_stats():
    """Get summary statistics from SEU data"""
    # Try to load analysis results first
    df = load_data("ml_pipeline/seu_analysis_results.csv")
    
    # Fall back to synthetic data
    if df is None:
        df = load_data("ml_pipeline/seu_synthetic_data.csv")
    
    if df is None:
        raise HTTPException(status_code=404, detail="No data available")
    
    # Create summary statistics
    summary = {
        "total_records": len(df),
        "max_altitude": float(df['altitude'].max()) if 'altitude' in df.columns else None,
        "total_bit_flips": int(df['bit_flips_count'].sum()) if 'bit_flips_count' in df.columns else None,
        "avg_bit_flips": float(df['bit_flips_count'].mean()) if 'bit_flips_count' in df.columns else None,
        "max_bit_flips": int(df['bit_flips_count'].max()) if 'bit_flips_count' in df.columns else None,
        "anomalies_detected": int(df['is_anomaly'].sum()) if 'is_anomaly' in df.columns else None,
    }
    
    # Add altitude correlation if both columns exist
    if 'altitude' in df.columns and 'bit_flips_count' in df.columns:
        summary["altitude_correlation"] = float(df['altitude'].corr(df['bit_flips_count']))
    
    # Add time range if timestamp column exists
    if 'timestamp' in df.columns:
        try:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            summary["time_range"] = {
                "start": df['timestamp'].min().isoformat(),
                "end": df['timestamp'].max().isoformat(),
            }
        except:
            pass
    
    return summary

@app.post("/predict/anomaly", response_model=SEUPrediction, tags=["Predictions"])
async def predict_anomaly(event: SEUEvent):
    """Predict if an SEU event is anomalous"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="ML model not available")
    
    # Convert to dict for model
    data_dict = event.dict()
    
    # Make prediction
    try:
        prediction = model.predict_real_time(data_dict)
        
        # Add estimated bit flips if not provided
        if event.bit_flips_count == 0:
            # Simple estimation based on altitude and cosmic intensity
            altitude = event.altitude
            cosmic = event.cosmic_intensity if event.cosmic_intensity else altitude / 10000
            prediction['estimated_bit_flips'] = int(max(0, np.random.poisson(cosmic * 0.1)))
        
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """Get information about the ML model"""
    if not model_loaded:
        return {"status": "not_loaded", "reason": "Model could not be loaded"}
    
    return {
        "status": "loaded",
        "type": model.classifier.__class__.__name__ if model.classifier else "IsolationForest",
        "features_used": ["altitude", "temperature", "bit_flips_count", "cosmic_intensity"]
    }

if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("Starting SEU Detector API")
    print("API docs available at: http://localhost:8000/docs")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
