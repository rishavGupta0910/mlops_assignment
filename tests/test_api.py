"""Tests for the FastAPI application."""

from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api.app import app

client = TestClient(app)

SAMPLE_INPUT = {
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1,
}


class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_format(self):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "model_name" in data
        assert "timestamp" in data


class TestPredictEndpoint:
    """Test /predict endpoint."""

    def test_predict_returns_200(self):
        response = client.post("/predict", json=SAMPLE_INPUT)
        assert response.status_code == 200

    def test_predict_response_format(self):
        response = client.post("/predict", json=SAMPLE_INPUT)
        data = response.json()
        assert "prediction" in data
        assert "prediction_label" in data
        assert "confidence" in data
        assert "model_name" in data
        assert "timestamp" in data

    def test_predict_binary_output(self):
        response = client.post("/predict", json=SAMPLE_INPUT)
        data = response.json()
        assert data["prediction"] in [0, 1]

    def test_predict_confidence_range(self):
        response = client.post("/predict", json=SAMPLE_INPUT)
        data = response.json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_missing_field_returns_422(self):
        incomplete = {"age": 63, "sex": 1}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_predict_invalid_value_returns_422(self):
        invalid = SAMPLE_INPUT.copy()
        invalid["age"] = -5  # age must be >= 0
        response = client.post("/predict", json=invalid)
        assert response.status_code == 422

    def test_predict_label_matches_prediction(self):
        response = client.post("/predict", json=SAMPLE_INPUT)
        data = response.json()
        if data["prediction"] == 1:
            assert data["prediction_label"] == "Disease"
        else:
            assert data["prediction_label"] == "No Disease"
