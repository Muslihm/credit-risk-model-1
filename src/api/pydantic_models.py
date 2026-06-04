
"""
Pydantic models for request and response validation in the Credit Risk API

Author: Muslihm
Date: June 2026
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from datetime import datetime


class CreditApplication(BaseModel):
    """
    Request model for a single credit risk prediction.
    Matches the features used in model training.
    """
    
    # Transaction features
    Amount: float = Field(..., gt=0, description="Transaction amount in local currency")
    Value: float = Field(..., ge=0, description="Transaction value")
    PricingStrategy: int = Field(..., ge=0, le=10, description="Pricing strategy used")
    
    # Location features
    CountryCode: int = Field(..., description="Country code of the transaction")
    
    # Product features
    ProductId: str = Field(..., description="Product identifier")
    ProductCategory: str = Field(..., description="Product category")
    
    # Provider features
    ProviderId: str = Field(..., description="Provider identifier")
    ChannelId: str = Field(..., description="Channel identifier")
    
    class Config:
        schema_extra = {
            "example": {
                "Amount": 250.50,
                "Value": 250.50,
                "PricingStrategy": 2,
                "CountryCode": 256,
                "ProductId": "ProductId_10",
                "ProductCategory": "financial_services",
                "ProviderId": "ProviderId_2",
                "ChannelId": "ChannelId_1"
            }
        }


class BatchCreditRequest(BaseModel):
    """
    Request model for batch predictions.
    """
    applications: List[CreditApplication] = Field(..., min_items=1, max_items=1000)


class PredictionResponse(BaseModel):
    """
    Response model for a single prediction.
    """
    default_probability: float = Field(..., ge=0, le=1, description="Probability of default/high risk")
    risk_category: str = Field(..., description="Risk category: Low, Medium, High, Very High")
    is_high_risk: int = Field(..., ge=0, le=1, description="Binary risk flag (1=High Risk, 0=Low Risk)")
    model_version: str = Field(..., description="Model version used")
    timestamp: datetime = Field(..., description="Prediction timestamp")


class BatchPredictionResponse(BaseModel):
    """
    Response model for batch predictions.
    """
    predictions: List[PredictionResponse]
    total_processed: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """
    Health check response.
    """
    status: str
    model_loaded: bool
    model_version: str
    timestamp: datetime


class ErrorResponse(BaseModel):
    """
    Error response model.
    """
    error: str
    detail: Optional[str] = None
    timestamp: datetime
