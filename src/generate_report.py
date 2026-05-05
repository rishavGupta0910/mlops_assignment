"""
Generate the 10-page DOCX report for MLOps Assignment 1.

This script is run incrementally as each day's work completes.
It reads from models_metadata.json, screenshots, and experiment logs
to build a professional report.
"""

import os
import json
from datetime import datetime
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "report")


def add_heading_styled(doc, text, level=1):
    heading = doc.add_heading(text, level=level)
    return heading


def add_image_if_exists(doc, path, width=Inches(5.5), caption=None):
    if os.path.exists(path):
        doc.add_picture(path, width=width)
        last_paragraph = doc.paragraphs[-1]
        last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if caption:
            p = doc.add_paragraph(caption)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.runs[0].italic = True
            p.runs[0].font.size = Pt(9)
        return True
    return False


def generate_report():
    os.makedirs(REPORT_DIR, exist_ok=True)
    doc = Document()

    # ========== PAGE 1: TITLE PAGE ==========
    for _ in range(6):
        doc.add_paragraph("")

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("MLOps Assignment 1")
    run.bold = True
    run.font.size = Pt(28)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("End-to-End ML Model Development, CI/CD, and Production Deployment")
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(100, 100, 100)

    doc.add_paragraph("")

    details = doc.add_paragraph()
    details.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = details.add_run("Course: MLOps (S2-25_AMLCSZG523)")
    run.font.size = Pt(12)

    doc.add_paragraph("")

    dataset_p = doc.add_paragraph()
    dataset_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = dataset_p.add_run("Dataset: Heart Disease UCI (Cleveland)")
    run.font.size = Pt(12)

    doc.add_paragraph("")

    date_p = doc.add_paragraph()
    date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = date_p.add_run(f"Date: {datetime.now().strftime('%B %d, %Y')}")
    run.font.size = Pt(11)

    doc.add_page_break()

    # ========== PAGE 2: TABLE OF CONTENTS ==========
    add_heading_styled(doc, "Table of Contents", level=1)
    toc_items = [
        "1. Setup & Installation Instructions",
        "2. Data Acquisition & Exploratory Data Analysis",
        "3. Feature Engineering & Model Development",
        "4. Experiment Tracking (MLflow)",
        "5. Model Packaging & Reproducibility",
        "6. CI/CD Pipeline & Automated Testing",
        "7. Model Containerization (Docker)",
        "8. Production Deployment (GKE)",
        "9. Monitoring & Logging",
        "10. Architecture Diagram & Conclusion",
    ]
    for item in toc_items:
        p = doc.add_paragraph(item)
        p.paragraph_format.space_after = Pt(6)

    doc.add_page_break()

    # ========== PAGES 3-4: SETUP & ARCHITECTURE ==========
    add_heading_styled(doc, "1. Setup & Installation Instructions", level=1)

    doc.add_paragraph(
        "This project implements a complete MLOps pipeline for heart disease "
        "prediction using the Cleveland dataset from the UCI ML Repository."
    )

    add_heading_styled(doc, "Technology Stack", level=2)
    tech_items = [
        ("Language", "Python 3.11"),
        ("ML Framework", "scikit-learn"),
        ("Experiment Tracking", "MLflow (local)"),
        ("API Framework", "FastAPI"),
        ("Containerization", "Docker"),
        ("Container Registry", "GCP Artifact Registry"),
        ("Orchestration", "GKE (Google Kubernetes Engine)"),
        ("CI/CD", "GitHub Actions"),
        ("Monitoring", "Prometheus + Grafana"),
        ("Testing", "Pytest"),
    ]
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    hdr[0].text = "Component"
    hdr[1].text = "Technology"
    for comp, tech in tech_items:
        row = table.add_row().cells
        row[0].text = comp
        row[1].text = tech

    doc.add_paragraph("")

    add_heading_styled(doc, "Quick Start", level=2)
    setup_steps = [
        "1. Clone repository: git clone <repo-url>",
        "2. Create conda environment: conda create -n mlops python=3.11",
        "3. Activate: conda activate mlops",
        "4. Install dependencies: pip install -r requirements.txt",
        "5. Run training: python src/train.py",
        "6. Run API locally: uvicorn api.app:app --reload",
        "7. Run tests: pytest tests/ -v",
    ]
    for step in setup_steps:
        doc.add_paragraph(step, style="List Bullet")

    doc.add_page_break()

    # ========== PAGES 5-6: EDA & MODELLING ==========
    add_heading_styled(doc, "2. Data Acquisition & EDA", level=1)

    doc.add_paragraph(
        "The dataset is fetched directly from the UCI ML Repository using the "
        "ucimlrepo Python package (dataset ID: 45). This ensures reproducibility "
        "without shipping data files in the repository."
    )

    doc.add_paragraph(
        "The Cleveland dataset contains 303 patient records with 13 features "
        "and a target variable (num) indicating heart disease presence (0-4). "
        "We binarize the target: 0 = no disease, 1-4 = disease present."
    )

    add_heading_styled(doc, "Key EDA Findings", level=2)
    findings = [
        "Class balance: 164 no disease (54%) vs 139 disease (46%) - roughly balanced",
        "Missing values: Only 6 total (ca: 4, thal: 2) - median imputation applied",
        "Strongest predictors: thal, ca, oldpeak, exang, cp, thalach",
        "Age and cholesterol show significant overlap between classes",
        "Chest pain type 4 (asymptomatic) strongly associated with disease",
    ]
    for f in findings:
        doc.add_paragraph(f, style="List Bullet")

    # Add EDA plots
    eda_dir = os.path.join(SCREENSHOTS_DIR, "eda")
    add_image_if_exists(
        doc,
        os.path.join(eda_dir, "class_balance.png"),
        width=Inches(5),
        caption="Figure 1: Class distribution (original and binary)",
    )
    add_image_if_exists(
        doc,
        os.path.join(eda_dir, "correlation_heatmap.png"),
        width=Inches(5),
        caption="Figure 2: Feature correlation heatmap",
    )

    doc.add_page_break()

    add_image_if_exists(
        doc,
        os.path.join(eda_dir, "feature_distributions.png"),
        width=Inches(5.5),
        caption="Figure 3: Feature distributions by target class",
    )
    add_image_if_exists(
        doc,
        os.path.join(eda_dir, "boxplots_by_target.png"),
        width=Inches(5.5),
        caption="Figure 4: Numerical features box plots",
    )

    doc.add_page_break()

    # ========== PAGE 7: MODEL DEVELOPMENT & EXPERIMENT TRACKING ==========
    add_heading_styled(doc, "3. Feature Engineering & Model Development", level=1)

    doc.add_paragraph(
        "Features are preprocessed using a scikit-learn ColumnTransformer pipeline: "
        "StandardScaler for 5 numerical features (age, trestbps, chol, thalach, oldpeak) "
        "and OneHotEncoder for 8 categorical features (sex, cp, fbs, restecg, exang, "
        "slope, ca, thal). This produces 20 transformed features."
    )

    add_heading_styled(doc, "Models Trained", level=2)
    doc.add_paragraph(
        "Three classifiers were trained with hyperparameter tuning via "
        "5-fold stratified cross-validation using GridSearchCV:"
    )

    # Load metrics from metadata
    metadata_path = os.path.join(MODELS_DIR, "models_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)

        table = doc.add_table(rows=1, cols=6)
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        hdr = table.rows[0].cells
        for i, h in enumerate(["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]):
            hdr[i].text = h

        for name, metrics in metadata["models"].items():
            row = table.add_row().cells
            row[0].text = name
            row[1].text = f"{metrics['accuracy']:.4f}"
            row[2].text = f"{metrics['precision']:.4f}"
            row[3].text = f"{metrics['recall']:.4f}"
            row[4].text = f"{metrics['f1']:.4f}"
            row[5].text = f"{metrics['roc_auc']:.4f}"

        doc.add_paragraph("")
        doc.add_paragraph(
            f"Best model: {metadata['best_model']} with "
            f"ROC-AUC = {metadata['models'][metadata['best_model']]['roc_auc']:.4f}"
        )

    add_image_if_exists(
        doc,
        os.path.join(eda_dir, "roc_curves_comparison.png"),
        width=Inches(4.5),
        caption="Figure 5: ROC curves for all models",
    )
    add_image_if_exists(
        doc, os.path.join(eda_dir, "confusion_matrices.png"), width=Inches(5.5), caption="Figure 6: Confusion matrices"
    )

    doc.add_page_break()

    # ========== PAGE 8: EXPERIMENT TRACKING ==========
    add_heading_styled(doc, "4. Experiment Tracking (MLflow)", level=1)

    doc.add_paragraph(
        "MLflow is integrated into the training pipeline (src/train.py) to track "
        "all experiment runs. For each model, the following are logged:"
    )
    logged_items = [
        "Parameters: model type, hyperparameter grid, best hyperparameters, test_size, cv_folds",
        "Metrics: accuracy, precision, recall, F1, ROC-AUC (test + cross-validation)",
        "Artifacts: trained model (sklearn), confusion matrix plot, ROC curve plot, feature importance plot",
    ]
    for item in logged_items:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_paragraph(
        "The MLflow tracking data is stored in experiment_tracking/ and can be "
        "viewed by running: mlflow ui --backend-store-uri experiment_tracking/"
    )

    add_heading_styled(doc, "Experiment Tracking Directory Structure", level=2)
    doc.add_paragraph(
        "Since MLflow uses internally generated IDs for its directory structure "
        "(not human-readable), we also export a clean, time-based summary after "
        "each training run to screenshots/mlflow/. Each run creates a timestamped "
        "subfolder (e.g., run_2026-05-04_13-20-32/) containing:"
    )
    export_items = [
        "experiment_metrics_comparison.csv - side-by-side comparison of all model metrics with run timestamp",
        "experiment_runs_log.json - full experiment log including hyperparameter grids, best parameters, and artifact references",
        "Confusion matrix, ROC curve, and feature importance plots for all models",
    ]
    for item in export_items:
        doc.add_paragraph(item, style="List Bullet")

    add_heading_styled(doc, "Training Artifacts", level=2)
    doc.add_paragraph(
        "The training_artifacts/ directory stores the raw plots (confusion matrices, "
        "ROC curves, feature importance charts) generated during each model's training. "
        "These are organized by model name (e.g., training_artifacts/LogisticRegression/). "
        "MLflow logs these as artifacts within each run, and they are also copied into "
        "the time-based screenshots/mlflow/ folders for easy reference."
    )

    # Add MLflow UI screenshots
    mlflow_dir = os.path.join(SCREENSHOTS_DIR, "mlflow")
    add_image_if_exists(
        doc,
        os.path.join(mlflow_dir, "training_runs.png"),
        width=Inches(5.5),
        caption="Figure 7: MLflow UI - All experiment runs",
    )
    add_image_if_exists(
        doc,
        os.path.join(mlflow_dir, "logistic_regression.png"),
        width=Inches(5.5),
        caption="Figure 8: MLflow UI - Logistic Regression run details",
    )

    doc.add_page_break()

    # ========== MODEL PACKAGING & REPRODUCIBILITY ==========
    add_heading_styled(doc, "5. Model Packaging & Reproducibility", level=1)

    doc.add_paragraph(
        "All three trained models are serialized as full scikit-learn pipelines "
        "(ColumnTransformer + Classifier) using joblib pickle format. This means "
        "the model file contains both the preprocessing and classification logic, "
        "so raw features can be fed directly without separate preprocessing steps."
    )

    add_heading_styled(doc, "Inference Script", level=2)
    doc.add_paragraph(
        "A standalone inference script (src/inference.py) allows command-line "
        "predictions without running the API server:"
    )
    doc.add_paragraph(
        'python src/inference.py --input \'{"age": 63, "sex": 1, "cp": 3, ...}\'',
        style="List Bullet",
    )
    doc.add_paragraph(
        "The script automatically loads the best model (from models_metadata.json) "
        "and returns the prediction, label, confidence score, and model name."
    )

    add_heading_styled(doc, "Model Artifacts", level=2)
    model_files = [
        "models/LogisticRegression.pkl - Best model (ROC-AUC: 0.9643)",
        "models/RandomForest.pkl - Second best (ROC-AUC: 0.9556)",
        "models/GradientBoosting.pkl - Third (ROC-AUC: 0.9177)",
        "models/models_metadata.json - Metrics, best model indicator, feature list",
    ]
    for item in model_files:
        doc.add_paragraph(item, style="List Bullet")

    add_heading_styled(doc, "Unit Tests", level=2)
    doc.add_paragraph("32 unit tests cover the full pipeline using Pytest:")
    test_items = [
        "test_data_loader.py (5 tests) - data shape, columns, target values",
        "test_preprocessing.py (8 tests) - missing value handling, binarization",
        "test_model.py (10 tests) - model loading, prediction format, all 3 models",
        "test_api.py (9 tests) - API endpoints, validation, error handling",
    ]
    for item in test_items:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_page_break()

    # ========== MODEL CONTAINERIZATION ==========
    add_heading_styled(doc, "6. Model Containerization (Docker)", level=1)

    add_heading_styled(doc, "FastAPI Application", level=2)
    doc.add_paragraph("The prediction API is built with FastAPI (api/app.py) and exposes " "the following endpoints:")
    endpoints = [
        "GET /health - Returns model status, confirms model is loaded and ready",
        "POST /predict - Accepts 13 patient features (JSON), returns prediction (0/1), "
        "confidence score, prediction label, and model name",
        "GET /metrics - Prometheus-format metrics (request count, latency histogram, "
        "auto-instrumented via prometheus-fastapi-instrumentator)",
    ]
    for ep in endpoints:
        doc.add_paragraph(ep, style="List Bullet")

    doc.add_paragraph(
        "Input validation uses Pydantic models with range constraints on all 13 "
        "features (e.g., age >= 0, sex in [0,1], cp in [0,3]). Invalid inputs "
        "return HTTP 422 with detailed error messages."
    )

    add_heading_styled(doc, "Docker Configuration", level=2)
    doc.add_paragraph(
        "The Dockerfile uses python:3.11-slim as the base image, installs pinned "
        "dependencies from requirements.txt, copies the source code, API, and model "
        "artifacts, and runs uvicorn on port 8000."
    )
    doc.add_paragraph("Build and run commands:")
    docker_cmds = [
        "docker build -t heart-disease-api .",
        "docker run -p 8000:8000 heart-disease-api",
    ]
    for cmd in docker_cmds:
        doc.add_paragraph(cmd, style="List Bullet")

    doc.add_paragraph(
        "The containerized API was tested locally with curl, confirming both "
        "/health and /predict endpoints return correct responses."
    )

    doc.add_page_break()

    # ========== CI/CD PIPELINE ==========
    add_heading_styled(doc, "7. CI/CD Pipeline & Automated Testing", level=1)

    doc.add_paragraph(
        "A GitHub Actions CI/CD pipeline (.github/workflows/ci-cd.yml) automates "
        "the full lifecycle from code push to production deployment."
    )

    add_heading_styled(doc, "Pipeline Stages", level=2)
    stages = [
        "Lint - flake8 (style checks) + black --check (formatting verification)",
        "Test - install dependencies, train model (fresh), run pytest (32 tests)",
        "Build & Push - train model, build Docker image, push to GCP Artifact Registry",
        "Deploy - get GKE credentials, kubectl set image, verify rollout status",
    ]
    for stage in stages:
        doc.add_paragraph(stage, style="List Bullet")

    doc.add_paragraph(
        "The pipeline triggers on push to main and on pull requests. Build & Push "
        "and Deploy stages only run on main branch pushes. Each stage depends on "
        "the previous one, ensuring failures are caught early."
    )

    add_heading_styled(doc, "GitHub Secrets", level=2)
    secrets = [
        "GCP_SA_KEY - Service account JSON key for GCP authentication",
        "GCP_PROJECT_ID - mlops-personal-lab",
        "GKE_CLUSTER - mlops-cluster",
        "GKE_ZONE - us-central1-a",
    ]
    for s in secrets:
        doc.add_paragraph(s, style="List Bullet")

    doc.add_page_break()

    # ========== PRODUCTION DEPLOYMENT ==========
    add_heading_styled(doc, "8. Production Deployment (GKE)", level=1)

    doc.add_paragraph(
        "The model API is deployed to Google Kubernetes Engine (GKE) using a "
        "zonal Standard cluster with 1 e2-small node."
    )

    add_heading_styled(doc, "GCP Infrastructure", level=2)
    infra_items = [
        "Project: mlops-personal-lab",
        "Cluster: mlops-cluster (us-central1-a, 1 node, e2-small)",
        "Container Registry: us-central1-docker.pkg.dev/mlops-personal-lab/mlops-repo",
        "Service Account: github-actions-sa (Artifact Registry Writer + GKE Developer)",
    ]
    for item in infra_items:
        doc.add_paragraph(item, style="List Bullet")

    add_heading_styled(doc, "Kubernetes Configuration", level=2)
    doc.add_paragraph(
        "The deployment uses 1 replica with resource requests of 100m CPU / 128Mi memory "
        "and limits of 250m CPU / 256Mi memory. Liveness and readiness probes hit the "
        "/health endpoint to ensure the pod is serving traffic correctly."
    )
    doc.add_paragraph(
        "A LoadBalancer Service exposes the API on port 80 (forwarding to container port 8000), "
        "provisioning a public IP address automatically."
    )

    add_heading_styled(doc, "Live Endpoint", level=2)
    doc.add_paragraph("The API is accessible at: http://34.60.20.112")
    doc.add_paragraph("Verified endpoints:")
    verified = [
        "GET /health - returns model status and readiness",
        "POST /predict - accepts patient features, returns prediction with confidence",
        "GET /metrics - Prometheus-format metrics for monitoring",
    ]
    for v in verified:
        doc.add_paragraph(v, style="List Bullet")

    doc.add_page_break()

    # ========== MONITORING (Day 4 placeholder) ==========
    add_heading_styled(doc, "9. Monitoring & Logging", level=1)
    doc.add_paragraph("[To be completed on Day 4 - Prometheus metrics, Grafana dashboard, " "API request logging]")

    doc.add_page_break()

    add_heading_styled(doc, "10. Architecture Diagram & Conclusion", level=1)
    doc.add_paragraph(
        "[To be added - End-to-end pipeline diagram: Data -> Preprocessing -> "
        "Training (MLflow) -> Model Packaging -> Docker -> GKE -> API -> Monitoring]"
    )
    doc.add_paragraph("")
    doc.add_paragraph(
        "[To be completed on Day 4 - Summary of findings, lessons learned, " "and link to code repository]"
    )

    # Save
    output_path = os.path.join(REPORT_DIR, "MLOps_Assignment1_Report.docx")
    doc.save(output_path)
    print(f"Report saved to {output_path}")
    return output_path


if __name__ == "__main__":
    generate_report()
