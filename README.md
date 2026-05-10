# Heart Disease Prediction - MLOps Pipeline

End-to-end ML pipeline for heart disease prediction using the UCI Cleveland dataset. Includes EDA, model training with MLflow tracking, FastAPI serving, Docker containerization, CI/CD via GitHub Actions, GKE deployment, and Prometheus + Grafana monitoring.

## Project Structure

```
mlops_assignment/
├── api/                    # FastAPI application
├── deployment/
│   ├── k8s/               # Kubernetes manifests
│   └── monitoring/        # Prometheus + Grafana manifests
├── experiment_tracking/    # MLflow tracking store
├── models/                 # Trained model pickles
├── notebooks/              # EDA and training notebooks
├── screenshots/            # Evidence for submission
├── scripts/                # Utility scripts (traffic generation)
├── src/                    # Source code (training, inference, preprocessing)
├── tests/                  # Unit tests (pytest)
├── training_artifacts/     # Training plots per model
├── Dockerfile
├── requirements.txt
└── .github/workflows/      # CI/CD pipeline
```

## Quick Start

### 1. Environment Setup

```bash
conda create -n mlops python=3.11 -y
conda activate mlops
pip install -r requirements.txt
```

### 2. Run Training

```bash
python src/train.py
```

This trains 3 models (Logistic Regression, Random Forest, Gradient Boosting), logs to MLflow, and saves pickles to `models/`.

### 3. View MLflow UI

```bash
mlflow ui --backend-store-uri experiment_tracking/
# Open http://localhost:5000
```

### 4. Run API Locally

```bash
uvicorn api.app:app --reload
# Open http://localhost:8000/docs for Swagger UI
```

### 5. Run Tests

```bash
pytest tests/ -v
```

### 6. Docker

```bash
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check - model status |
| `/predict` | POST | Make prediction from patient features |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | Swagger UI documentation |

### Example Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}'
```

Response:
```json
{
  "prediction": 0,
  "prediction_label": "No Disease",
  "confidence": 0.8682,
  "model_name": "LogisticRegression",
  "timestamp": "2026-05-05T03:42:01.659148"
}
```

## GKE Deployment

### Deploy to GKE

```bash
# Authenticate
gcloud container clusters get-credentials mlops-cluster --zone=us-central1-a --project=mlops-personal-lab

# Apply manifests
kubectl apply -f deployment/k8s/

# Check status
kubectl get pods
kubectl get svc heart-disease-api
```

### Deploy Monitoring Stack

```bash
kubectl apply -f deployment/monitoring/
```

### Access Grafana Dashboard

```bash
# Port-forward Grafana to localhost
kubectl port-forward svc/grafana 3000:3000

# Open in browser
open http://localhost:3000
```

- **Login**: admin / admin (skip password change)
- Navigate to **Dashboards → Heart Disease API Monitoring**
- The dashboard includes:
  - **ML Metrics**: Prediction distribution, confidence scores, active model info
  - **API Metrics**: Request rate, latency percentiles, HTTP status codes, error rate

### Access Prometheus

```bash
kubectl port-forward svc/prometheus 9090:9090

# Open http://localhost:9090
```

Useful queries:
- `ml_predictions_total` — Total predictions by class
- `ml_prediction_confidence_bucket` — Confidence score distribution
- `rate(http_requests_total[1m])` — Request rate

### Generate Test Traffic

```bash
./scripts/generate_traffic.sh http://34.60.20.112 50
```

This sends randomized prediction requests to populate the monitoring dashboards.

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci-cd.yml`):

1. **Lint** - flake8 + black formatting check
2. **Test** - train models + pytest (32 tests)
3. **Build & Push** - Docker image to GCP Artifact Registry
4. **Deploy** - Rolling update to GKE

Triggers on push to `main` and pull requests.

## Models

| Model | ROC-AUC | Accuracy |
|-------|---------|----------|
| **Logistic Regression** | 0.9643 | 0.8852 |
| Random Forest | 0.9556 | 0.8852 |
| Gradient Boosting | 0.9177 | 0.8361 |

Best model: Logistic Regression (selected by ROC-AUC).

## Dataset

Heart Disease UCI (Cleveland) - 303 samples, 13 features, binary target.
Fetched via `ucimlrepo` package (dataset ID: 45).

## Cost Control

After the assignment, scale down or delete resources:

```bash
# Scale nodes to zero
gcloud container clusters resize mlops-cluster --zone=us-central1-a --node-pool=default-pool --num-nodes=0 --quiet

# Or delete cluster entirely
gcloud container clusters delete mlops-cluster --zone=us-central1-a --quiet
```

