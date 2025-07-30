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


# Add the src directory to the Python path
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


def log_prediction(input_data, prediction, confidence, latency):
    """Log prediction to both file and SQLite database."""
    
    # Get the current timestamp in ISO format
    timestamp = datetime.now().isoformat()
    
    print(f"[INFO] Logging prediction with timestamp {timestamp}")
    
    # Create a log entry as a dictionary
    log_entry = {
        "timestamp": timestamp,
        "input": input_data,
        "prediction": prediction,
        "confidence": confidence,
        "latency": latency
    }
    
    print(f"[INFO] Logging to file: {log_entry}")
    
    # Create a directory for logging if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Append the log entry to a file named 'requests.log' in the 'logs' directory
    with open('logs/requests.log', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    # Log to SQLite database
    try:
        # Connect to the database
        conn = sqlite3.connect('logs/predictions.db')
        
        # Create a cursor object to execute SQL queries
        cursor = conn.cursor()
        
        # Execute an SQL query to insert the log entry into the 'predictions' table
        cursor.execute('''
            INSERT INTO predictions (timestamp, input_data, prediction, confidence, latency)
            VALUES (?, ?, ?, ?, ?)
        ''', (timestamp, json.dumps(input_data), prediction, confidence, latency))
        
        # Commit the database changes
        conn.commit()
        
        # Close the database connection
        conn.close()
    except Exception as e:
        print(f"[ERROR] Could not log to database: {e}")


def store_new_training_data(features, target):
    timestamp = datetime.now().isoformat()
    
    print(f"[INFO] Storing new training data with timestamp {timestamp}")
    
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect('logs/predictions.db')
        
        # Create a cursor to execute SQL queries
        cursor = conn.cursor()
        
        # Execute an SQL query to insert the new training data into the 'new_training_data' table
        cursor.execute('''
            INSERT INTO new_training_data (timestamp, features, target)
            VALUES (?, ?, ?)
        ''', (timestamp, json.dumps(features), target))
        
        # Commit the database changes
        conn.commit()
        
        # Close the database connection
        conn.close()
        
        print("[INFO] Stored new training data successfully!")
        
        # Return True to indicate that the data was stored successfully
        return True
    except Exception as e:
        # Print an error message if an exception occurred
        print(f"[ERROR] Error storing training data: {e}")
        
        # Return False to indicate that the data was not stored successfully
        return False


def check_retrain_trigger():
    """Check if we have enough new samples to trigger retraining."""
    
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect('logs/predictions.db')
        
        # Create a cursor object to execute SQL queries
        cursor = conn.cursor()
        
        # Execute an SQL query to count the number of new training samples
        # that have not been used for training yet
        cursor.execute('''
            SELECT COUNT(*) FROM new_training_data WHERE used_for_training = FALSE
        ''')
        
        # Fetch the result of the query, which is the count of new samples
        count = cursor.fetchone()[0]
        
        # Close the database connection
        conn.close()
        
        # Return True if the number of new samples meets or exceeds the retrain threshold
        return count >= retrain_threshold
    
    except Exception as e:
        # Print an error message if an exception occurred
        print(f"Error checking retrain trigger: {e}")
        
        # Return False to indicate that we could not check the retrain trigger
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


