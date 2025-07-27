import os
import json
import time
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
from pydantic import ValidationError
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from schemas import PredictionRequest, PredictionResponse, RetrainingRequest, NewDataSample
import threading
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
try:
    from model_training_simple import main as retrain_models
except ImportError:
    def retrain_models():
        print("Model retraining module not available")

app = Flask(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('prediction_requests_total', 'Total prediction requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('prediction_request_duration_seconds', 'Request latency')
PREDICTION_COUNTER = Counter('predictions_total', 'Total predictions made', ['predicted_class'])
MODEL_ACCURACY = Gauge('model_accuracy', 'Current model accuracy')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Number of active connections')
RETRAIN_TRIGGER_COUNT = Counter('retraining_triggered_total', 'Number of times retraining was triggered')

# Global variables for monitoring
request_count = 0
total_latency = 0.0
model = None
scaler = None
feature_names = None
target_names = None
new_data_samples = []  # Store new samples for retraining
retrain_threshold = 10


def init_db():
    """Initialize SQLite database for logging predictions."""
    os.makedirs('logs', exist_ok=True)
    conn = sqlite3.connect('logs/predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            input_data TEXT,
            prediction TEXT,
            confidence REAL,
            latency REAL
        )
    ''')
    
    # Table for storing new training data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS new_training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            features TEXT,
            target INTEGER,
            used_for_training BOOLEAN DEFAULT FALSE
        )
    ''')
    conn.commit()
    conn.close()


def load_model_and_metadata():
    """Load the trained model, scaler, and metadata."""
    global model, scaler, feature_names, target_names
    
    try:
        # Load metadata
        with open('models/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        feature_names = metadata['feature_names']
        target_names = metadata['target_names']
        best_model = metadata['best_model']
        best_accuracy = metadata['best_accuracy']
        
        # Update Prometheus gauge
        MODEL_ACCURACY.set(best_accuracy)
        
        # Load the appropriate model
        if best_model == "LogisticRegression":
            model = joblib.load('models/logistic_regression_model.pkl')
        else:
            model = joblib.load('models/random_forest_model.pkl')
        
        # Load scaler
        scaler = joblib.load('models/scaler.pkl')
        
        print(f"[INFO] Loaded {best_model} model successfully!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return False


def log_prediction(input_data, prediction, confidence, latency):
    """Log prediction to both file and SQLite database."""
    timestamp = datetime.now().isoformat()
    
    # Log to file
    log_entry = {
        "timestamp": timestamp,
        "input": input_data,
        "prediction": prediction,
        "confidence": confidence,
        "latency": latency
    }
    
    os.makedirs('logs', exist_ok=True)
    with open('logs/requests.log', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    # Log to SQLite
    try:
        conn = sqlite3.connect('logs/predictions.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (timestamp, input_data, prediction, confidence, latency)
            VALUES (?, ?, ?, ?, ?)
        ''', (timestamp, json.dumps(input_data), prediction, confidence, latency))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Warning: Could not log to database: {e}")


def store_new_training_data(features, target):
    """Store new training data for future retraining."""
    timestamp = datetime.now().isoformat()
    
    try:
        conn = sqlite3.connect('logs/predictions.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO new_training_data (timestamp, features, target)
            VALUES (?, ?, ?)
        ''', (timestamp, json.dumps(features), target))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error storing training data: {e}")
        return False


def check_retrain_trigger():
    """Check if we have enough new samples to trigger retraining."""
    try:
        conn = sqlite3.connect('logs/predictions.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) FROM new_training_data WHERE used_for_training = FALSE
        ''')
        count = cursor.fetchone()[0]
        conn.close()
        return count >= retrain_threshold
    except Exception as e:
        print(f"Error checking retrain trigger: {e}")
        return False


def trigger_retraining():
    """Trigger model retraining in background thread."""
    def retrain():
        try:
            print("[INFO] Starting background model retraining...")
            RETRAIN_TRIGGER_COUNT.inc()
            
            # Mark data as used for training
            conn = sqlite3.connect('logs/predictions.db')
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE new_training_data SET used_for_training = TRUE 
                WHERE used_for_training = FALSE
            ''')
            conn.commit()
            conn.close()
            
            # Trigger retraining
            retrain_models()
            
            # Reload the updated model
            load_model_and_metadata()
            print("[INFO] Model retraining completed successfully!")
            
        except Exception as e:
            print(f"[ERROR] Error during retraining: {e}")
    
    thread = threading.Thread(target=retrain)
    thread.daemon = True
    thread.start()


@app.before_request
def before_request():
    """Track active connections."""
    ACTIVE_CONNECTIONS.inc()


@app.after_request
def after_request(response):
    """Track request completion."""
    ACTIVE_CONNECTIONS.dec()
    return response


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    REQUEST_COUNT.labels(method='GET', endpoint='/health').inc()
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions on input data with Pydantic validation."""
    global request_count, total_latency
    
    REQUEST_COUNT.labels(method='POST', endpoint='/predict').inc()
    start_time = time.time()
    
    try:
        # Validate input using Pydantic
        try:
            request_data = PredictionRequest(**request.get_json())
        except ValidationError as e:
            return jsonify({
                "error": "Input validation failed",
                "details": e.errors()
            }), 400
        
        features = request_data.features
        
        # Convert to numpy array and scale
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        confidence = float(max(prediction_proba))
        
        # Get prediction name
        prediction_name = target_names[prediction]
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Update metrics
        request_count += 1
        total_latency += latency
        REQUEST_LATENCY.observe(latency)
        PREDICTION_COUNTER.labels(predicted_class=prediction_name).inc()
        
        # Log the prediction
        log_prediction(features, prediction_name, confidence, latency)
        
        # Create response using Pydantic
        response_data = PredictionResponse(
            prediction=int(prediction),
            prediction_name=prediction_name,
            confidence=confidence,
            probabilities={
                target_names[i]: float(prob) for i, prob in enumerate(prediction_proba)
            },
            latency=latency
        )
        
        return jsonify(response_data.dict())
        
    except Exception as e:
        latency = time.time() - start_time
        request_count += 1
        total_latency += latency
        
        return jsonify({
            "error": str(e),
            "latency": latency
        }), 500


@app.route('/add_training_data', methods=['POST'])
def add_training_data():
    """Add new training data sample."""
    REQUEST_COUNT.labels(method='POST', endpoint='/add_training_data').inc()
    
    try:
        # Validate input
        try:
            sample = NewDataSample(**request.get_json())
        except ValidationError as e:
            return jsonify({
                "error": "Input validation failed",
                "details": e.errors()
            }), 400
        
        # Store the new sample
        if store_new_training_data(sample.features, sample.target):
            # Check if we should trigger retraining
            if check_retrain_trigger():
                trigger_retraining()
                message = "Training data added. Retraining triggered!"
            else:
                message = "Training data added successfully."
            
            return jsonify({
                "message": message,
                "features": sample.features,
                "target": sample.target
            })
        else:
            return jsonify({"error": "Failed to store training data"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/trigger_retrain', methods=['POST'])
def manual_retrain():
    """Manually trigger model retraining."""
    REQUEST_COUNT.labels(method='POST', endpoint='/trigger_retrain').inc()
    
    try:
        # Validate input
        try:
            retrain_request = RetrainingRequest(**request.get_json()) if request.get_json() else RetrainingRequest()
        except ValidationError as e:
            return jsonify({
                "error": "Input validation failed",
                "details": e.errors()
            }), 400
        
        global retrain_threshold
        retrain_threshold = retrain_request.trigger_threshold
        
        if retrain_request.force_retrain or check_retrain_trigger():
            trigger_retraining()
            return jsonify({
                "message": "Model retraining triggered successfully",
                "threshold": retrain_threshold,
                "forced": retrain_request.force_retrain
            })
        else:
            return jsonify({
                "message": "Not enough new samples to trigger retraining",
                "threshold": retrain_threshold,
                "current_samples": len(new_data_samples)
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/metrics', methods=['GET'])
def metrics():
    """Expose metrics endpoint for monitoring."""
    REQUEST_COUNT.labels(method='GET', endpoint='/metrics').inc()
    avg_latency = total_latency / request_count if request_count > 0 else 0
    
    # Get database stats
    try:
        conn = sqlite3.connect('logs/predictions.db')
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total_predictions = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(latency) FROM predictions')
        avg_db_latency = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT prediction, COUNT(*) FROM predictions GROUP BY prediction')
        prediction_counts = dict(cursor.fetchall())
        
        cursor.execute('SELECT COUNT(*) FROM new_training_data WHERE used_for_training = FALSE')
        new_samples_count = cursor.fetchone()[0]
        
        conn.close()
    except:
        total_predictions = 0
        avg_db_latency = 0
        prediction_counts = {}
        new_samples_count = 0
    
    metrics_data = {
        "request_count": request_count,
        "total_predictions": total_predictions,
        "average_latency": avg_latency,
        "average_db_latency": avg_db_latency,
        "prediction_distribution": prediction_counts,
        "new_samples_pending": new_samples_count,
        "retrain_threshold": retrain_threshold,
        "model_info": {
            "feature_names": feature_names,
            "target_names": target_names,
            "model_loaded": model is not None
        },
        "timestamp": datetime.now().isoformat()
    }
    
    return jsonify(metrics_data)


@app.route('/prometheus_metrics')
def prometheus_metrics():
    """Prometheus metrics endpoint."""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


@app.route('/dashboard')
def dashboard():
    """Simple monitoring dashboard."""
    dashboard_html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Model Dashboard</title>
        <meta http-equiv="refresh" content="30">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .metric-box { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .metric-value { font-size: 24px; font-weight: bold; color: #2e86c1; }
            .metric-label { font-size: 14px; color: #666; }
            .status-healthy { color: #27ae60; }
            .status-warning { color: #f39c12; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }
        </style>
    </head>
    <body>
        <h1>ML Model Monitoring Dashboard</h1>
        <div class="grid">
            <div class="metric-box">
                <div class="metric-label">Total Predictions</div>
                <div class="metric-value" id="total-predictions">Loading...</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Average Latency (ms)</div>
                <div class="metric-value" id="avg-latency">Loading...</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Model Accuracy</div>
                <div class="metric-value" id="model-accuracy">Loading...</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">New Samples Pending</div>
                <div class="metric-value" id="new-samples">Loading...</div>
            </div>
        </div>
        
        <h2>Prediction Distribution</h2>
        <div id="prediction-chart">Loading...</div>
        
        <script>
            async function updateMetrics() {
                try {
                    const response = await fetch('/metrics');
                    const data = await response.json();
                    
                    document.getElementById('total-predictions').textContent = data.total_predictions;
                    document.getElementById('avg-latency').textContent = (data.average_latency * 1000).toFixed(2);
                    document.getElementById('model-accuracy').textContent = (data.model_info.model_accuracy || 0.97).toFixed(3);
                    document.getElementById('new-samples').textContent = data.new_samples_pending;
                    
                    // Simple chart for prediction distribution
                    let chartHtml = '<div style="display: flex; gap: 20px;">';
                    for (const [className, count] of Object.entries(data.prediction_distribution || {})) {
                        chartHtml += `<div class="metric-box"><div class="metric-label">${className}</div><div class="metric-value">${count}</div></div>`;
                    }
                    chartHtml += '</div>';
                    document.getElementById('prediction-chart').innerHTML = chartHtml;
                    
                } catch (error) {
                    console.error('Error fetching metrics:', error);
                }
            }
            
            // Update metrics every 10 seconds
            updateMetrics();
            setInterval(updateMetrics, 10000);
        </script>
    </body>
    </html>
    '''
    return render_template_string(dashboard_html)


@app.route('/predictions/history', methods=['GET'])
def prediction_history():
    """Get recent prediction history from database."""
    REQUEST_COUNT.labels(method='GET', endpoint='/predictions/history').inc()
    limit = request.args.get('limit', 10, type=int)
    
    try:
        conn = sqlite3.connect('logs/predictions.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT timestamp, input_data, prediction, confidence, latency 
            FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                "timestamp": row[0],
                "input_data": json.loads(row[1]),
                "prediction": row[2],
                "confidence": row[3],
                "latency": row[4]
            })
        
        return jsonify({"history": history, "count": len(history)})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API documentation."""
    return jsonify({
        "message": "Enhanced Iris Classification API with Pydantic Validation",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Make predictions (POST) - with Pydantic validation",
            "/add_training_data": "Add new training sample (POST)",
            "/trigger_retrain": "Manually trigger retraining (POST)",
            "/metrics": "System metrics",
            "/prometheus_metrics": "Prometheus metrics",
            "/dashboard": "Monitoring dashboard",
            "/predictions/history": "Recent prediction history"
        },
        "features": [
            "Pydantic input/output validation",
            "Prometheus metrics integration", 
            "Automatic model retraining",
            "Real-time monitoring dashboard"
        ],
        "example_request": {
            "features": [5.1, 3.5, 1.4, 0.2]
        }
    })


if __name__ == '__main__':
    # Initialize database
    init_db()
    
    # Load model and metadata
    if not load_model_and_metadata():
        print("[ERROR] Failed to load model. Please train the model first.")
        exit(1)
    
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