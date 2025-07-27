import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from data_preprocessing_simple import load_and_preprocess_data
import joblib
import json

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Train Logistic Regression model.
    
    Returns:
        tuple: (model, accuracy)
    """
    print("Training Logistic Regression...")
    
    # Define model parameters
    model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save model locally
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, '../models/logistic_regression_model.pkl')
    
    print(f"[INFO] Logistic Regression trained!")
    print(f"Accuracy: {accuracy:.4f}")
    
    return model, accuracy

def train_random_forest(X_train, X_test, y_train, y_test):
    """
    Train Random Forest model.
    
    Returns:
        tuple: (model, accuracy)
    """
    print("Training Random Forest...")
    
    # Define model parameters
    model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save model locally
    joblib.dump(model, '../models/random_forest_model.pkl')
    
    print(f"[INFO] Random Forest trained!")
    print(f"Accuracy: {accuracy:.4f}")
    
    return model, accuracy

def main():
    """
    Main function to run the complete model training pipeline.
    """
    print("Starting simplified model training pipeline...")
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, feature_names, target_names, scaler = load_and_preprocess_data()
    
    # Save the scaler for later use in API
    os.makedirs('../models', exist_ok=True)
    joblib.dump(scaler, '../models/scaler.pkl')
    
    # Train both models
    lr_model, lr_accuracy = train_logistic_regression(X_train, X_test, y_train, y_test)
    rf_model, rf_accuracy = train_random_forest(X_train, X_test, y_train, y_test)
    
    # Determine best model
    if lr_accuracy > rf_accuracy:
        best_model_name = "LogisticRegression"
        best_accuracy = lr_accuracy
        print(f"[INFO] Best model: {best_model_name} (accuracy: {best_accuracy:.4f})")
    else:
        best_model_name = "RandomForest"
        best_accuracy = rf_accuracy
        print(f"[INFO] Best model: {best_model_name} (accuracy: {best_accuracy:.4f})")
    
    # Save metadata
    metadata = {
        'feature_names': list(feature_names),
        'target_names': list(target_names),
        'best_model': best_model_name,
        'best_accuracy': best_accuracy,
        'logistic_regression_accuracy': lr_accuracy,
        'random_forest_accuracy': rf_accuracy
    }
    
    with open('../models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("[INFO] Model training pipeline completed!")
    print(f"Models saved in: {os.path.abspath('../models')}")

if __name__ == "__main__":
    main() 