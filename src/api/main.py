
"""
Credit Risk API - FastAPI application for credit risk prediction
Serves the best model from MLflow registry

Author: Muslihm
Date: June 2026
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.pydantic_models import (
    CreditApplication,
    BatchCreditRequest,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ErrorResponse
)


# Global variables
model = None
model_version = "v1.0.0"
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pkl")


def load_model():
    """Load the best model from disk"""
    global model, model_version
    
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"✅ Model loaded from {MODEL_PATH}")
        else:
            # Try to find model in MLflow runs
            import mlflow
            mlflow.set_tracking_uri("mlruns")
            
            # Get the latest registered model
            client = mlflow.tracking.MlflowClient()
            try:
                model_version_detail = client.get_latest_versions("Credit_Risk_Best_Model", stages=["None"])
                if model_version_detail:
                    model_uri = model_version_detail[0].source
                    model = mlflow.sklearn.load_model(model_uri)
                    model_version = model_version_detail[0].version
                    print(f"✅ Model loaded from MLflow: {model_uri}")
            except Exception as e:
                print(f"⚠️ Could not load from MLflow: {e}")
                return False
        
        return model is not None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def preprocess_input(application: CreditApplication) -> pd.DataFrame:
    """
    Convert API request to model input format.
    Creates the same features used during training.
    """
    # Convert to dictionary
    data = application.dict()
    
    # Create DataFrame with expected columns
    df = pd.DataFrame([data])
    
    # Add derived features (matching training)
    # Transaction hour (default to 12 if not provided)
    df['hour'] = 12
    df['day_of_month'] = 15
    df['month'] = 6
    df['year'] = 2024
    df['day_of_week'] = 3
    df['is_weekend'] = 0
    df['is_business_hour'] = 1
    
    # One-hot encode categorical columns
    categorical_cols = ['ProductId', 'ProductCategory', 'ProviderId', 'ChannelId']
    for col in categorical_cols:
        dummy_cols = [c for c in model.feature_names_in_ if c.startswith(f"{col}_")]
        for dummy in dummy_cols:
            expected_value = dummy.split('_', 1)[1]
            df[dummy] = (df[col] == expected_value).astype(int)
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Ensure all expected columns are present
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match training
    df = df[model.feature_names_in_]
    
    # Fill any NaN
    df = df.fillna(0)
    
    return df


def get_risk_category(probability: float) -> str:
    """Convert probability to risk category"""
    if probability < 0.02:
        return "Low Risk"
    elif probability < 0.05:
        return "Medium Risk"
    elif probability < 0.10:
        return "High Risk"
    else:
        return "Very High Risk"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    # Startup
    print("Starting Credit Risk API...")
    success = load_model()
    if not success:
        print("⚠️ WARNING: Model failed to load. API will return 503 errors.")
    yield
    # Shutdown
    print("Shutting down Credit Risk API...")


# Create FastAPI app
app = FastAPI(
    title="Credit Risk Scoring API",
    description="""
    API for predicting credit risk / fraud probability.
    
    This API serves the best model trained on transaction data.
    Returns probability of high-risk behavior and risk category.
    
    ## Features
    - Single prediction endpoint
    - Batch prediction endpoint
    - Model health check
    - Automatic feature engineering
    """,
    version=model_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Returns API status and model information.
    """
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_version=model_version,
        timestamp=datetime.now()
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        200: {"description": "Successful prediction"},
        503: {"description": "Model not loaded", "model": ErrorResponse},
        422: {"description": "Validation error", "model": ErrorResponse}
    },
    tags=["Predictions"]
)
async def predict(application: CreditApplication):
    """
    Predict risk probability for a single credit application.
    
    Returns:
    - default_probability: Probability of high-risk (0-1)
    - risk_category: Low/Medium/High/Very High
    - is_high_risk: Binary flag (1=High Risk)
    - model_version: Version of model used
    - timestamp: When prediction was made
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        # Preprocess input
        X = preprocess_input(application)
        
        # Make prediction
        probability = float(model.predict_proba(X)[0, 1])
        is_high_risk = int(probability >= 0.05)  # Threshold for high risk
        risk_category = get_risk_category(probability)
        
        return PredictionResponse(
            default_probability=probability,
            risk_category=risk_category,
            is_high_risk=is_high_risk,
            model_version=model_version,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses={
        200: {"description": "Successful batch predictions"},
        503: {"description": "Model not loaded"},
        422: {"description": "Validation error"}
    },
    tags=["Predictions"]
)
async def predict_batch(batch_request: BatchCreditRequest):
    """
    Predict risk probabilities for multiple credit applications.
    
    Maximum 1000 applications per request.
    """
    import time
    
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )
    
    start_time = time.time()
    predictions = []
    
    for i, application in enumerate(batch_request.applications):
        try:
            X = preprocess_input(application)
            probability = float(model.predict_proba(X)[0, 1])
            is_high_risk = int(probability >= 0.05)
            risk_category = get_risk_category(probability)
            
            predictions.append(PredictionResponse(
                default_probability=probability,
                risk_category=risk_category,
                is_high_risk=is_high_risk,
                model_version=model_version,
                timestamp=datetime.now()
            ))
        except Exception as e:
            # For batch, continue with other predictions
            predictions.append(PredictionResponse(
                default_probability=0.0,
                risk_category="Error",
                is_high_risk=0,
                model_version=model_version,
                timestamp=datetime.now()
            ))
    
    processing_time = (time.time() - start_time) * 1000
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_processed=len(predictions),
        processing_time_ms=round(processing_time, 2)
    )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """
    Get information about the loaded model.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "model_version": model_version,
        "model_type": type(model).__name__,
        "features": model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else [],
        "n_features": len(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else 0,
        "loaded": True
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
