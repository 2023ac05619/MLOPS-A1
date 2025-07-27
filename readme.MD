# Machine Learning Operations (MLOps) Pipeline

This project demonstrates a complete MLOps pipeline for the Iris dataset. It features automated CI/CD, model training, experiment tracking, API deployment, and monitoring—all designed to run offline and locally. The goal is to provide a robust, reproducible workflow for developing and deploying machine learning models.

---

## Part 1: Repository and Data Versioning

### Local Git Repository Setup

Initialize Git repository with proper configuration:

```bash
git init
```

### Dataset Loading and Preprocessing

* Uses the Iris dataset from scikit-learn for classification tasks
* Implements data preprocessing with pandas and scikit-learn
* Includes train/test splits and feature scaling

### Data Version Control (DVC)

Initialize DVC for data tracking:

```bash
dvc init
```

Track datasets:

```bash
dvc add data/your_dataset.csv
git add .dvc/ data/your_dataset.csv.dvc
git commit -m "Added and tracked dataset"
```

### Directory Structure

```
your_project/
├── data/
├── models/
├── src/
├── logs/
├── api/
├── mlruns/
└── Dockerfile
```

---

## Part 2: Model Development & Experiment Tracking

### Model Training

Implemented multiple machine learning models:

* Logistic Regression 
* Random Forest 
* Decision Tree
* Linear Regression

All models use scikit-learn with appropriate data splitting and evaluation metrics.

### MLflow Experiment Tracking

Local MLflow tracking implementation:

```python
mlflow.set_tracking_uri("file:///absolute/path/to/mlruns")
mlflow.start_run()
mlflow.log_param("model", "RandomForest")
mlflow.log_metric("accuracy", 0.92)
mlflow.sklearn.log_model(model, "model")
mlflow.end_run()
```

### Model Registry

Local model registration:

```python
mlflow.register_model("runs:/<RUN_ID>/model", "BestLocalModel")
```

---

## Part 3: API & Docker Packaging

### RESTful API Implementation

Flask-based API with endpoints for prediction, health checks, metrics, and history.

### Docker Containerization

Production-ready Dockerfile:

```Dockerfile
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "api/app.py"]
```

Build and run:

```bash
docker build -t offline-ml-api .
docker run -p 5000:5000 offline-ml-api
```

---

## Part 4: Local CI/CD Automation

### Automated Testing and Linting

Comprehensive CI pipeline (`ci.sh`):

```bash
#!/bin/bash
flake8 src/
pytest tests/
```

### Docker Image Build

```bash
docker build -t offline-ml-api .
```

### Local Deployment

Automated deployment script:

```bash
# deploy.sh
docker stop offline-ml-api || true
docker rm offline-ml-api || true
docker run -d -p 5000:5000 --name offline-ml-api offline-ml-api
```

---

## Part 5: Logging and Monitoring

### Prediction Logging

Local file and database logging:

```python
with open("logs/requests.log", "a") as f:
    f.write(json.dumps({"input": input_data, "prediction": output}) + "\n")
```

### SQLite Database Integration

Uses Python's `sqlite3` module for prediction storage and querying.

### Metrics Endpoint

API endpoint `/metrics` provides:

* Request count
* Average latency  
* Model usage statistics

---

## Key Features

### Input Validation using Pydantic

Robust input validation implementation:

```python
from pydantic import BaseModel, Field, validator

class PredictionRequest(BaseModel):
    features: List[float] = Field(..., min_items=4, max_items=4)
```

### Prometheus Integration & Dashboard

Metrics collection and monitoring:

```python
from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter('prediction_requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('prediction_request_duration_seconds', 'Latency')
```

### Model Re-training Trigger

Automated model retraining capability:

```python
@app.route('/add_training_data', methods=['POST'])
def add_training_data():
    # Store new sample and trigger retraining if threshold reached
```

---

## Implementation Summary

| Feature                | Status      | Notes                                 |
|------------------------|-------------|----------------------------------------|
| Git Repository         | Complete    | Version control with Git               |
| Data Versioning        | Complete    | DVC for local dataset tracking         |
| Experiment Logging     | Complete    | MLflow for experiment tracking         |
| API & Docker           | Complete    | Flask API, Dockerized for deployment   |
| CI/CD                  | Complete    | Automated scripts for build/test/deploy|
| Logging/Monitoring     | Complete    | Logs and metrics with SQLite, Prometheus|
| Input Validation       | Complete    | Pydantic schemas for API validation    |
| Metrics Dashboard      | Complete    | Real-time metrics and dashboard        |
| Model Retraining       | Complete    | Supports retraining on new data        |

---

## Getting Started

1. **Clone the repository and install dependencies:**
   ```bash
   git clone <repo-url>
   cd <project-folder>
   pip install -r requirements.txt
   ```

2. **Run the data preprocessing and model training scripts:**
   ```bash
   python src/data_preprocessing_simple.py
   python src/model_training_mlflow.py
   ```

3. **Start the API:**
   ```bash
   python api/app.py
   # or use Docker:
   ./deploy.sh
   ```

---

## API Endpoints

The API exposes several endpoints for interacting with the model and monitoring the service:

- **GET /health**
  - Quick health check to see if the service is running.
  - Example response: `{ "status": "healthy", "timestamp": "..." }`

- **POST /predict**
  - Make a prediction by sending a JSON payload with your features.
  - Example input: `{ "features": [5.1, 3.5, 1.4, 0.2] }`
  - Example output: `{ "prediction": 0, "prediction_name": "setosa", "confidence": 0.98, ... }`

- **POST /add_training_data**
  - Add a new training sample to the system.
  - Example input: `{ "features": [4.9, 3.0, 1.4, 0.2], "target": 0 }`

- **POST /trigger_retrain**
  - Manually trigger model retraining (useful after adding new data).
  - Example input: `{ "force_retrain": true }`

- **GET /metrics**
  - Get system and model metrics, such as request count and average latency.

- **GET /prometheus_metrics**
  - Prometheus-compatible metrics for monitoring tools.

- **GET /dashboard**
  - A simple web dashboard for real-time monitoring (open in your browser).

- **GET /predictions/history**
  - Retrieve recent prediction history.
  - Example output: `{ "history": [...], "count": 10 }`

---

## API Usage Examples
```bash
# Health check
curl http://localhost:5000/health

# Make prediction with validation
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

# Add training data
curl -X POST http://localhost:5000/add_training_data \
  -H "Content-Type: application/json" \
  -d '{"features": [4.9, 3.0, 1.4, 0.2], "target": 0}'

# View monitoring dashboard
open http://localhost:5000/dashboard

# Get Prometheus metrics
curl http://localhost:5000/prometheus_metrics

# Trigger manual retraining
curl -X POST http://localhost:5000/trigger_retrain \
  -H "Content-Type: application/json" \
  -d '{"force_retrain": true}'
```

### Available Commands
```bash
# Run enhanced tests
source venv/bin/activate && python -m pytest tests/ -v

# Start enhanced API
source venv/bin/activate && python api/app_enhanced.py

# Run CI pipeline
./ci.sh

# Deploy with Docker
./deploy.sh
```

### Technical Achievements
- 96.67% accuracy on Random Forest model
- Fully offline - no external dependencies or cloud services
- Production-ready input validation with Pydantic
- Real-time monitoring with Prometheus and custom dashboard
- Intelligent retraining with automatic threshold-based triggers
- Complete logging system with both file and database storage
- Containerized deployment ready with Docker
- Automated CI/CD pipeline with local shell scripts
- Comprehensive API with 8 endpoints covering all functionality