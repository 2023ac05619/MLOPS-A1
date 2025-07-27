import os
import json
import time
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Global variables for monitoring
request_count = 0
total_latency = 0.0
model = None
scaler = None
feature_names = None
target_names = None

# Initialize SQLite database for logging
def init_db():
    """Initialize SQLite database for logging predictions."""
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

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions on input data."""
    global request_count, total_latency
    
    start_time = time.time()
    
    try:
        # Get input data
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({
                "error": "Invalid input. Expected JSON with 'features' key."
            }), 400
        
        features = data['features']
        
        # Validate input features
        if len(features) != 4:
            return jsonify({
                "error": f"Expected 4 features, got {len(features)}"
            }), 400
        
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
        
        # Update global metrics
        request_count += 1
        total_latency += latency
        
        # Log the prediction
        log_prediction(features, prediction_name, confidence, latency)
        
        response = {
            "prediction": int(prediction),
            "prediction_name": prediction_name,
            "confidence": confidence,
            "probabilities": {
                target_names[i]: float(prob) for i, prob in enumerate(prediction_proba)
            },
            "latency": latency
        }
        
        return jsonify(response)
        
    except Exception as e:
        latency = time.time() - start_time
        request_count += 1
        total_latency += latency
        
        return jsonify({
            "error": str(e),
            "latency": latency
        }), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Expose metrics endpoint for monitoring."""
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
        
        conn.close()
    except:
        total_predictions = 0
        avg_db_latency = 0
        prediction_counts = {}
    
    metrics_data = {
        "request_count": request_count,
        "total_predictions": total_predictions,
        "average_latency": avg_latency,
        "average_db_latency": avg_db_latency,
        "prediction_distribution": prediction_counts,
        "model_info": {
            "feature_names": feature_names,
            "target_names": target_names,
            "model_loaded": model is not None
        },
        "timestamp": datetime.now().isoformat()
    }
    
    return jsonify(metrics_data)

@app.route('/predictions/history', methods=['GET'])
def prediction_history():
    """Get recent prediction history from database."""
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
        "message": "Iris Classification API",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Make predictions (POST)",
            "/metrics": "System metrics",
            "/predictions/history": "Recent prediction history"
        },
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

    print("[INFO] Starting Flask API server...")
    print("[INFO] API endpoints available:")
    print("  - GET  /           : API documentation")
    print("  - GET  /health     : Health check")
    print("  - POST /predict    : Make predictions")
    print("  - GET  /metrics    : System metrics")
    print("  - GET  /predictions/history : Recent predictions")
    
    app.run(host='0.0.0.0', port=5000, debug=False) 