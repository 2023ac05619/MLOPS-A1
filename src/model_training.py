import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from data_preprocessing_simple import load_and_preprocess_data
import joblib

# Set MLflow tracking URI to local directory (one level up from src)
mlflow.set_tracking_uri(f"file://{os.path.abspath('../mlruns')}")

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Train Logistic Regression model with MLflow tracking.
    
    Returns:
        tuple: (model, accuracy, run_id)
    """
    with mlflow.start_run(run_name="Logistic_Regression") as run:
        # Define model parameters
        C = 1.0
        max_iter = 1000
        
        # Log parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("solver", "lbfgs")
        
        # Train the model
        model = LogisticRegression(C=C, max_iter=max_iter, solver='lbfgs', random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("test_samples", len(y_test))
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally (create directory if needed)
        os.makedirs('../models', exist_ok=True)
        joblib.dump(model, '../models/logistic_regression_model.pkl')
        
        print(f"[INFO] Logistic Regression trained!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Run ID: {run.info.run_id}")
        
        return model, accuracy, run.info.run_id

def train_random_forest(X_train, X_test, y_train, y_test):
    """
    Train Random Forest model with MLflow tracking.
    
    Returns:
        tuple: (model, accuracy, run_id)
    """
    with mlflow.start_run(run_name="Random_Forest") as run:
        # Define model parameters
        n_estimators = 100
        max_depth = 3
        
        # Log parameters
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", 42)
        
        # Train the model
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("test_samples", len(y_test))
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally
        joblib.dump(model, '../models/random_forest_model.pkl')
        
        print(f"[INFO] Random Forest trained!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Run ID: {run.info.run_id}")
        
        return model, accuracy, run.info.run_id

def register_best_model(lr_accuracy, lr_run_id, rf_accuracy, rf_run_id):
    """
    Register the best performing model.
    
    Args:
        lr_accuracy: Logistic Regression accuracy
        lr_run_id: Logistic Regression run ID
        rf_accuracy: Random Forest accuracy  
        rf_run_id: Random Forest run ID
    """
    if lr_accuracy > rf_accuracy:
        best_run_id = lr_run_id
        best_model_name = "LogisticRegression"
        best_accuracy = lr_accuracy
    else:
        best_run_id = rf_run_id
        best_model_name = "RandomForest"
        best_accuracy = rf_accuracy
    
    # Register the best model
    model_uri = f"runs:/{best_run_id}/model"
    registered_model = mlflow.register_model(model_uri, "BestLocalModel")
    
    print(f"[INFO] Best model registered: {best_model_name}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    print(f"Model version: {registered_model.version}")
    
    return best_run_id, best_model_name, best_accuracy

def main():
    """
    Main function to run the complete model training pipeline.
    """
    print("Starting model training pipeline...")
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, feature_names, target_names, scaler = load_and_preprocess_data()
    
    # Save the scaler for later use in API (create directory if needed)
    os.makedirs('../models', exist_ok=True)
    joblib.dump(scaler, '../models/scaler.pkl')
    
    # Train both models
    lr_model, lr_accuracy, lr_run_id = train_logistic_regression(X_train, X_test, y_train, y_test)
    rf_model, rf_accuracy, rf_run_id = train_random_forest(X_train, X_test, y_train, y_test)
    
    # Register the best model
    best_run_id, best_model_name, best_accuracy = register_best_model(
        lr_accuracy, lr_run_id, rf_accuracy, rf_run_id
    )
    
    # Save metadata
    metadata = {
        'feature_names': feature_names.tolist(),
        'target_names': target_names.tolist(),
        'best_model': best_model_name,
        'best_accuracy': best_accuracy,
        'best_run_id': best_run_id
    }
    
    import json
    with open('../models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("[INFO] Model training pipeline completed!")

if __name__ == "__main__":
    main() 