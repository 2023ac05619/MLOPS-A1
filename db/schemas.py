from pydantic import BaseModel, Field, validator
from typing import List, Optional
import numpy as np


class PredictionRequest(BaseModel):
    """Pydantic model for prediction request validation."""
    features: List[float] = Field(
        ..., 
        description="List of 4 Iris features: sepal length, sepal width, petal length, petal width",
        min_items=4,
        max_items=4,
        example=[5.1, 3.5, 1.4, 0.2]
    )
    
    @validator('features')
    def validate_features(cls, v):
        """Validate that all features are positive numbers."""
        if len(v) != 4:
            raise ValueError('Exactly 4 features required for Iris classification')
        
        for i, feature in enumerate(v):
            if not isinstance(feature, (int, float)):
                raise ValueError(f'Feature {i+1} must be a number')
            if feature < 0:
                raise ValueError(f'Feature {i+1} must be positive')
            if feature > 50:  # Reasonable upper bound for Iris features
                raise ValueError(f'Feature {i+1} seems unreasonably large (>{50})')
        
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2]
            }
        }


class PredictionResponse(BaseModel):
    """Pydantic model for prediction response."""
    prediction: int = Field(..., description="Predicted class (0=setosa, 1=versicolor, 2=virginica)")
    prediction_name: str = Field(..., description="Predicted class name")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score between 0 and 1")
    probabilities: dict = Field(..., description="Probability for each class")
    latency: float = Field(..., description="Prediction latency in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 0,
                "prediction_name": "setosa",
                "confidence": 1.0,
                "probabilities": {
                    "setosa": 1.0,
                    "versicolor": 0.0,
                    "virginica": 0.0
                },
                "latency": 0.003
            }
        }


class RetrainingRequest(BaseModel):
    """Pydantic model for retraining request validation."""
    trigger_threshold: Optional[int] = Field(
        default=10, 
        description="Number of new samples needed to trigger retraining",
        ge=1,
        le=1000
    )
    force_retrain: Optional[bool] = Field(
        default=False, 
        description="Force retraining regardless of threshold"
    )


class NewDataSample(BaseModel):
    """Pydantic model for adding new training data."""
    features: List[float] = Field(
        ..., 
        description="List of 4 Iris features",
        min_items=4,
        max_items=4
    )
    target: int = Field(
        ..., 
        description="True class label (0=setosa, 1=versicolor, 2=virginica)",
        ge=0,
        le=2
    )
    
    @validator('features')
    def validate_features(cls, v):
        """Validate that all features are positive numbers."""
        if len(v) != 4:
            raise ValueError('Exactly 4 features required')
        
        for i, feature in enumerate(v):
            if not isinstance(feature, (int, float)):
                raise ValueError(f'Feature {i+1} must be a number')
            if feature < 0:
                raise ValueError(f'Feature {i+1} must be positive')
        
        return v 