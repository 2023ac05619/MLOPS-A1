import os
import sqlite3
import json
import joblib
from datetime import datetime
from prometheus_flask_exporter import PrometheusMetrics 

def init_db():
    """Initialize SQLite database for logging predictions."""
    
    # Check if database file exists
    db_path = 'logs/predictions.db'
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # If database file doesn't exist, create new database
    if not os.path.isfile(db_path):
        print("[INFO] Database file does not exist, creating new database...")
        
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create 'predictions' table
        print("[INFO] Creating 'predictions' table...")
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
        
        # Create 'new_training_data' table
        print("[INFO] Creating 'new_training_data' table...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS new_training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                features TEXT,
                target INTEGER,
                used_for_training BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Commit database changes
        print("[INFO] Committing database changes...")
        conn.commit()
        
        # Close database connection
        print("[INFO] Closing database connection...")
        conn.close()
    else:
        print("[INFO] Using existing database file...")


def load_model_and_metadata():
    global model, scaler, feature_names, target_names
    
    print("[INFO] Loading model and metadata...")
    
    try:
        # Load metadata
        print("[INFO] Loading metadata...")
        with open('models/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print(f"[INFO] Loaded metadata: {metadata}")
        
        # Extract feature names, target names, best model, and best accuracy
        feature_names = metadata['feature_names']
        target_names = metadata['target_names']
        best_model = metadata['best_model']
        best_accuracy = metadata['best_accuracy']
        
        print(f"[INFO] Best model: {best_model} with accuracy {best_accuracy}")
        
        # Update Prometheus gauge
        print("[INFO] Updating Prometheus gauge...")
        MODEL_ACCURACY.set(best_accuracy)
        
        # Load the appropriate model
        if best_model == "LogisticRegression":
            print("[INFO] Loading LogisticRegression model")
            model = joblib.load('models/logistic_regression_model.pkl')
        else:
            print("[INFO] Loading RandomForest model")
            model = joblib.load('models/random_forest_model.pkl')
        
        # Load scaler
        print("[INFO] Loading scaler")
        scaler = joblib.load('models/scaler.pkl')
        
        print("[INFO] Loaded model and metadata successfully!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return False
