"""
Pydantic schemas for FastAPI request/response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
from datetime import datetime


class CreditApplication(BaseModel):
    """Schema for a single credit application request."""
    
    income: float = Field(..., gt=0, description="Annual income in USD")
    debt: float = Field(..., ge=0, description="Total outstanding debt in USD")
    credit_age_years: float = Field(..., ge=0, le=50, description="Age of oldest credit account in years")
    num_delinquencies: int = Field(..., ge=0, le=50, description="Number of past due payments in last 2 years")
    credit_utilization: float = Field(..., ge=0, le=1, description="Credit utilization ratio (0 to 1)")
    num_credit_cards: int = Field(..., ge=0, le=20, description="Number of active credit cards")
    
    # Optional fields for richer modeling
    employment_years: Optional[float] = Field(None, ge=0, description="Years at current employer")
    home_ownership: Optional[str] = Field(None, description="Rent, Own, Mortgage")
    
    @validator('credit_utilization')
    def validate_utilization(cls, v):
        if v < 0 or v > 1:
            raise ValueError('credit_utilization must be between 0 and 1')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "income": 65000,
                "debt": 15000,
                "credit_age_years": 8,
                "num_delinquencies": 1,
                "credit_utilization": 0.35,
                "num_credit_cards": 3,
                "employment_years": 4,
                "home_ownership": "Mortgage"
            }
        }


class BatchCreditRequest(BaseModel):
    """Schema for batch prediction requests."""
    
    applications: List[CreditApplication] = Field(..., min_items=1, max_items=1000)


class PredictionResponse(BaseModel):
    """Schema for single prediction response."""
    
    application_id: Optional[str] = None
    default_probability: float = Field(..., ge=0, le=1)
    risk_category: str
    model_version: str
    timestamp: datetime
    
    class Config:
        schema_extra = {
            "example": {
                "application_id": "APP-12345",
                "default_probability": 0.0345,
                "risk_category": "Medium Risk",
                "model_version": "v1.0.0",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response."""
    
    predictions: List[PredictionResponse]
    total_processed: int
    processing_time_ms: float


class ModelExplanationResponse(BaseModel):
    """Schema for model explanation (regulatory compliance)."""
    
    default_probability: float
    risk_category: str
    model_type: str
    is_interpretable: bool
    feature_contributions: Optional[Dict[str, float]] = None
    intercept: Optional[float] = None
    
    class Config:
        schema_extra = {
            "example": {
                "default_probability": 0.0345,
                "risk_category": "Medium Risk",
                "model_type": "logistic_regression",
                "is_interpretable": True,
                "feature_contributions": {
                    "num_delinquencies": -0.234,
                    "credit_utilization": -0.187,
                    "debt_to_income": -0.098
                },
                "intercept": -1.234
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    version: str
    timestamp: datetime