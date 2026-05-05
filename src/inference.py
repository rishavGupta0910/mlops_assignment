"""
Standalone inference script for Heart Disease prediction.

Loads the best trained model and makes predictions on raw patient features.

Usage:
    python src/inference.py --input '{"age": 63, "sex": 1, "cp": 3, "trestbps": 145,
        "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}'
"""

import os
import sys
import json
import argparse
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
METADATA_PATH = os.path.join(MODELS_DIR, "models_metadata.json")

EXPECTED_FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


def load_model(model_name=None):
    """
    Load a trained model pipeline from pickle.

    Parameters
    ----------
    model_name : str, optional
        Name of the model to load. If None, loads the best model
        from models_metadata.json.

    Returns
    -------
    tuple
        (pipeline, model_name)
    """
    if model_name is None:
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
        model_name = metadata["best_model"]

    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    pipeline = joblib.load(model_path)
    return pipeline, model_name


def predict(input_data, model_name=None):
    """
    Make a prediction on raw patient features.

    Parameters
    ----------
    input_data : dict
        Dictionary with 13 patient features.
    model_name : str, optional
        Which model to use. Defaults to best model.

    Returns
    -------
    dict
        prediction (0 or 1), confidence (probability), model_name
    """
    # Validate input features
    missing = [f for f in EXPECTED_FEATURES if f not in input_data]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    pipeline, model_name = load_model(model_name)

    # Create DataFrame with correct column order
    df = pd.DataFrame([input_data])[EXPECTED_FEATURES]

    # Predict
    prediction = int(pipeline.predict(df)[0])
    confidence = float(pipeline.predict_proba(df)[0][prediction])

    return {
        "prediction": prediction,
        "prediction_label": "Disease" if prediction == 1 else "No Disease",
        "confidence": round(confidence, 4),
        "model_name": model_name,
    }


def main():
    parser = argparse.ArgumentParser(description="Heart Disease Prediction Inference")
    parser.add_argument("--input", type=str, required=True, help="JSON string with patient features")
    parser.add_argument(
        "--model", type=str, default=None, help="Model name (LogisticRegression, RandomForest, GradientBoosting)"
    )
    args = parser.parse_args()

    try:
        input_data = json.loads(args.input)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}")
        sys.exit(1)

    try:
        result = predict(input_data, model_name=args.model)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
