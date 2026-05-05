"""
FastAPI application for Heart Disease prediction.

Endpoints:
    GET  /health   - Health check
    POST /predict  - Make prediction from patient features
    GET  /metrics  - Prometheus metrics (auto-instrumented)
"""

import os
import sys
import json
import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

# Add src to path for inference module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from inference import load_model, predict, EXPECTED_FEATURES

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts presence of heart disease based on clinical features.",
    version="1.0.0",
)

# Prometheus instrumentation
Instrumentator().instrument(app).expose(app)


class PatientFeatures(BaseModel):
    """Input schema for patient clinical features."""
    age: float = Field(..., ge=0, le=120, description="Age in years")
    sex: float = Field(..., ge=0, le=1, description="Sex (0=female, 1=male)")
    cp: float = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=0, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=0, description="Serum cholesterol (mg/dl)")
    fbs: float = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (0/1)")
    restecg: float = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., ge=0, description="Max heart rate achieved")
    exang: float = Field(..., ge=0, le=1, description="Exercise induced angina (0/1)")
    oldpeak: float = Field(..., ge=0, description="ST depression induced by exercise")
    slope: float = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)")
    ca: float = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy (0-4)")
    thal: float = Field(..., ge=0, le=3, description="Thalassemia (0=normal, 1=fixed defect, 2=reversible defect, 3=?)")


class PredictionResponse(BaseModel):
    """Output schema for prediction results."""
    prediction: int
    prediction_label: str
    confidence: float
    model_name: str
    timestamp: str


class HealthResponse(BaseModel):
    """Output schema for health check."""
    status: str
    model_loaded: bool
    model_name: str
    timestamp: str


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint - verifies model is loaded and ready."""
    try:
        _, model_name = load_model()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_name=model_name,
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")


@app.post("/predict", response_model=PredictionResponse)
def make_prediction(features: PatientFeatures):
    """Make a heart disease prediction from patient features."""
    start_time = datetime.now()

    input_data = features.model_dump()

    try:
        result = predict(input_data)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    response = PredictionResponse(
        prediction=result["prediction"],
        prediction_label=result["prediction_label"],
        confidence=result["confidence"],
        model_name=result["model_name"],
        timestamp=start_time.isoformat(),
    )

    logger.info(
        f"prediction_request | input={json.dumps(input_data)} | "
        f"prediction={result['prediction']} | confidence={result['confidence']} | "
        f"model={result['model_name']}"
    )

    return response
