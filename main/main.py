import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from prometheus_flask_exporter import PrometheusMetrics
from pydantic import BaseModel, ValidationError
from datetime import datetime
from src.model_training import train_model
from db.db import init_db, get_recent_predictions
from src.mlflow import demonstrate_mlflow_tracking, show_detailed_run_info
from api import app
from api import health_check, predict_route, add_training_data_route, trigger_retrain_route
from api import metrics_route, prometheus_metrics_route, dashboard_route, predictions_history_route
from api import setup_routes



if __name__ == '__main__':
    # Initialize database
    init_db()
    
    # Load model and metadata
    if not load_model_and_metadata():
        print("[ERROR] Failed to load model. Please train the model first.")
        exit(1)
    
    print("[INFO] Starting MLflow Demonstration...")
    
    # Main demonstration
    demonstrate_mlflow_tracking()
    
    # Detailed run info
    show_detailed_run_info()
    
    print("\n" + "=" * 60)
    print("[INFO] MLflow Demonstration Complete!")
    print("\n[INFO] To view the MLflow UI:")
    print("   1. Run: mlflow ui")
    print("   2. Open: http://localhost:5000")
    print("\n[INFO] Key MLflow Features Demonstrated:")
    print("   [INFO] Experiment tracking")
    print("   [INFO] Parameter logging")
    print("   [INFO] Metrics logging")
    print("   [INFO] Model logging and registration")
    print("   [INFO] Artifact storage")
    print("   [INFO] Model versioning")
    print("   [INFO] Run comparison") 
    
    # Start Flask app
    app = Flask(__name__)
    CORS(app)
    metrics = PrometheusMetrics(app)    

    print("[INFO] Starting Enhanced Flask API server...")
    print("[INFO] API endpoints available:")
    print("  - GET  /              : API documentation")
    print("  - GET  /health        : Health check") 
    print("  - POST /predict       : Make predictions (with Pydantic validation)")
    print("  - POST /add_training_data : Add new training sample")
    print("  - POST /trigger_retrain   : Manually trigger retraining")
    print("  - GET  /metrics       : System metrics")
    print("  - GET  /prometheus_metrics : Prometheus metrics")
    print("  - GET  /dashboard     : Monitoring dashboard")
    print("  - GET  /predictions/history : Recent predictions")
    
    app.run(host='0.0.0.0', port=5000, debug=False) 