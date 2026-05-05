"""Tests for model loading and inference."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from inference import load_model, predict

SAMPLE_INPUT = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
}


class TestModelLoading:
    """Test model loading functionality."""

    def test_load_best_model(self):
        pipeline, model_name = load_model()
        assert pipeline is not None
        assert model_name == "LogisticRegression"

    def test_load_specific_model(self):
        pipeline, model_name = load_model("RandomForest")
        assert model_name == "RandomForest"
        assert pipeline is not None

    def test_load_nonexistent_model_raises(self):
        with pytest.raises(FileNotFoundError):
            load_model("NonExistentModel")


class TestPrediction:
    """Test prediction functionality."""

    def test_predict_returns_dict(self):
        result = predict(SAMPLE_INPUT)
        assert isinstance(result, dict)

    def test_predict_has_required_keys(self):
        result = predict(SAMPLE_INPUT)
        assert "prediction" in result
        assert "prediction_label" in result
        assert "confidence" in result
        assert "model_name" in result

    def test_prediction_is_binary(self):
        result = predict(SAMPLE_INPUT)
        assert result["prediction"] in [0, 1]

    def test_confidence_in_valid_range(self):
        result = predict(SAMPLE_INPUT)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_prediction_label_matches(self):
        result = predict(SAMPLE_INPUT)
        if result["prediction"] == 1:
            assert result["prediction_label"] == "Disease"
        else:
            assert result["prediction_label"] == "No Disease"

    def test_missing_features_raises(self):
        incomplete_input = {"age": 63, "sex": 1}
        with pytest.raises(ValueError, match="Missing features"):
            predict(incomplete_input)

    def test_predict_with_different_models(self):
        for model_name in ["LogisticRegression", "RandomForest", "GradientBoosting"]:
            result = predict(SAMPLE_INPUT, model_name=model_name)
            assert result["model_name"] == model_name
            assert result["prediction"] in [0, 1]
