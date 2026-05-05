"""
Model training script with MLflow experiment tracking.

Trains Logistic Regression and Random Forest classifiers,
performs hyperparameter tuning, and logs everything to MLflow.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, ConfusionMatrixDisplay,
    RocCurveDisplay
)
from sklearn.pipeline import Pipeline
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess
from feature_engineering import build_preprocessor, prepare_data

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots", "mlflow")
EXPERIMENT_TRACKING_DIR = os.path.join(BASE_DIR, "experiment_tracking")
TRAINING_ARTIFACTS_DIR = os.path.join(BASE_DIR, "training_artifacts")


def evaluate_model(model, X_test, y_test):
    """Compute classification metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    return metrics, y_pred, y_proba


def plot_confusion_matrix(y_test, y_pred, model_name, output_dir):
    """Save confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, ax=ax, cmap="Blues",
        display_labels=["No Disease", "Disease"]
    )
    ax.set_title(f"Confusion Matrix - {model_name}")
    path = os.path.join(output_dir, f"confusion_matrix_{model_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_curve(model, X_test, y_test, model_name, output_dir):
    """Save ROC curve plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title(f"ROC Curve - {model_name}")
    ax.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.5)")
    ax.legend()
    path = os.path.join(output_dir, f"roc_curve_{model_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_importance(model, feature_names, model_name, output_dir):
    """Save feature importance plot (for tree-based models)."""
    if not hasattr(model, "feature_importances_"):
        return None
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices], align="center")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Feature Importance - {model_name}")
    ax.invert_yaxis()
    path = os.path.join(output_dir, f"feature_importance_{model_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def train_and_log_model(model_name, model, param_grid, X_train, X_test,
                        y_train, y_test, preprocessor):
    """
    Train a model with GridSearchCV, evaluate, and log to MLflow.
    """
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])

    # Build param grid with pipeline prefix
    pipeline_param_grid = {
        f"classifier__{k}": v for k, v in param_grid.items()
    }

    # GridSearchCV with cross-validation
    grid_search = GridSearchCV(
        pipeline, pipeline_param_grid, cv=5, scoring="roc_auc",
        n_jobs=-1, return_train_score=True
    )

    with mlflow.start_run(run_name=model_name):
        # Train
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_

        # Log parameters
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("cv_folds", 5)
        for param, value in grid_search.best_params_.items():
            clean_param = param.replace("classifier__", "")
            mlflow.log_param(f"best_{clean_param}", value)

        # Cross-validation scores
        cv_scores = cross_val_score(
            best_pipeline, X_train, y_train, cv=5, scoring="roc_auc"
        )
        mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_roc_auc_std", cv_scores.std())

        # Evaluate on test set
        metrics, y_pred, y_proba = evaluate_model(best_pipeline, X_test, y_test)
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)

        # Log GridSearchCV best score
        mlflow.log_metric("grid_search_best_score", grid_search.best_score_)

        # Generate and log plots
        artifacts_dir = os.path.join(TRAINING_ARTIFACTS_DIR, model_name)
        os.makedirs(artifacts_dir, exist_ok=True)

        cm_path = plot_confusion_matrix(y_test, y_pred, model_name, artifacts_dir)
        mlflow.log_artifact(cm_path)

        roc_path = plot_roc_curve(
            best_pipeline, X_test, y_test, model_name, artifacts_dir
        )
        mlflow.log_artifact(roc_path)

        # Feature importance for tree-based models
        feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
        fi_path = plot_feature_importance(
            best_pipeline.named_steps["classifier"],
            feature_names, model_name, artifacts_dir
        )
        if fi_path:
            mlflow.log_artifact(fi_path)

        # Log the model
        mlflow.sklearn.log_model(best_pipeline, "model")

        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"Best params: {grid_search.best_params_}")
        print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print("Test metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    return best_pipeline, metrics


def main():
    """Main training pipeline."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

    # Load and preprocess data
    df = preprocess(save=True)
    X_train, X_test, y_train, y_test = prepare_data(df)
    preprocessor = build_preprocessor()

    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

    # Set up MLflow with relative path
    os.makedirs(EXPERIMENT_TRACKING_DIR, exist_ok=True)
    mlflow.set_tracking_uri(EXPERIMENT_TRACKING_DIR)
    mlflow.set_experiment("heart-disease-classification")

    # Define models and hyperparameter grids
    models = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l1", "l2"],
                "solver": ["saga"],
            }
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5],
            }
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.01, 0.1],
            }
        },
    }

    # Train all models
    results = {}
    pipelines = {}
    best_score = 0
    best_model_name = None

    for name, config in models.items():
        pipeline, metrics = train_and_log_model(
            name, config["model"], config["params"],
            X_train, X_test, y_train, y_test, preprocessor
        )
        results[name] = metrics
        pipelines[name] = pipeline
        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_model_name = name

    # Summary
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    results_df = pd.DataFrame(results).T
    print(results_df.to_string())

    print(f"\nBest model: {best_model_name} (ROC-AUC: {best_score:.4f})")

    # Save all models
    for name, pipeline in pipelines.items():
        model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(pipeline, model_path)
        print(f"Saved {name} to {model_path}")

    # Save consolidated metadata for all models
    all_metadata = {
        "best_model": best_model_name,
        "features": list(X_train.columns),
        "models": {},
    }
    for name, metrics in results.items():
        all_metadata["models"][name] = {
            k: float(v) for k, v in metrics.items()
        }

    metadata_path = os.path.join(MODELS_DIR, "models_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(all_metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    # Export MLflow experiment summary to screenshots/mlflow/
    # This provides proof of logged params, metrics, artifacts for the assignment
    export_experiment_summary(results, models, SCREENSHOTS_DIR)

    return results


def export_experiment_summary(results, model_configs, output_dir):
    """
    Export a clean summary of all MLflow experiment runs
    (parameters, metrics, artifacts) with timestamps for submission.

    Creates a time-based subfolder inside output_dir for each training run,
    e.g. screenshots/mlflow/run_2026-05-04_13-13-25/
    """
    from datetime import datetime
    import shutil

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    folder_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(output_dir, f"run_{folder_timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # 1. Metrics comparison CSV with timestamp
    results_df = pd.DataFrame(results).T
    results_df.index.name = "model"
    results_df["run_timestamp"] = timestamp
    csv_path = os.path.join(run_dir, "experiment_metrics_comparison.csv")
    results_df.to_csv(csv_path)
    print(f"Exported metrics comparison to {csv_path}")

    # 2. Full experiment log (params + metrics + timestamps per model)
    experiment_log = {
        "experiment_name": "heart-disease-classification",
        "run_timestamp": timestamp,
        "dataset": "Heart Disease UCI (Cleveland) via ucimlrepo",
        "total_samples": 303,
        "train_test_split": "80/20 stratified",
        "runs": {},
    }
    for name, config in model_configs.items():
        experiment_log["runs"][name] = {
            "run_timestamp": timestamp,
            "parameters": {
                "model_type": name,
                "test_size": 0.2,
                "cv_folds": 5,
                "hyperparameter_grid": {
                    k: str(v) for k, v in config["params"].items()
                },
            },
            "metrics": {
                k: round(float(v), 4) for k, v in results[name].items()
            },
            "artifacts": [
                f"confusion_matrix_{name}.png",
                f"roc_curve_{name}.png",
            ]
        }
        if name in ("RandomForest", "GradientBoosting"):
            experiment_log["runs"][name]["artifacts"].append(
                f"feature_importance_{name}.png"
            )

    log_path = os.path.join(run_dir, "experiment_runs_log.json")
    with open(log_path, "w") as f:
        json.dump(experiment_log, f, indent=2)
    print(f"Exported experiment log to {log_path}")

    # 3. Copy artifact plots into the time-based run folder
    for name in model_configs:
        src_dir = os.path.join(TRAINING_ARTIFACTS_DIR, name)
        if os.path.exists(src_dir):
            for fname in os.listdir(src_dir):
                if fname.endswith(".png"):
                    shutil.copy2(
                        os.path.join(src_dir, fname),
                        os.path.join(run_dir, fname)
                    )
    print(f"Copied artifact plots to {run_dir}")


if __name__ == "__main__":
    main()
