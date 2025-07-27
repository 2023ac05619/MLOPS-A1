import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
from data_preprocessing_simple import load_and_preprocess_data
import joblib
import json
from datetime import datetime

# Set MLflow tracking URI to local directory
mlflow.set_tracking_uri(f"file://{os.path.abspath('../mlruns')}")

def train_logistic_regression(X_train, X_test, y_train, y_test, feature_names, target_names):
    """
    Train Logistic Regression model with comprehensive MLflow tracking.
    
    Returns:
        tuple: (model, accuracy, run_id)
    """
    with mlflow.start_run(run_name="Logistic_Regression_Iris") as run:
        # Define model parameters
        C = 1.0
        max_iter = 1000
        solver = 'lbfgs'
        
        # Log parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("solver", solver)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("dataset", "Iris")
        mlflow.log_param("features", len(feature_names))
        mlflow.log_param("classes", len(target_names))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        
        # Train the model
        model = LogisticRegression(C=C, max_iter=max_iter, solver=solver, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("test_samples", len(y_test))
        
        # Log per-class metrics
        for i, class_name in enumerate(target_names):
            class_mask = y_test == i
            if np.any(class_mask):
                class_accuracy = accuracy_score(y_test[class_mask], y_pred[class_mask])
                # Sanitize class name for MLflow
                sanitized_class = class_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                mlflow.log_metric(f"accuracy_{sanitized_class}", class_accuracy)
        
        # Log confusion matrix as artifact
        cm = confusion_matrix(y_test, y_pred)
        cm_text = f"Confusion Matrix:\n{cm}\n\nClassification Report:\n{classification_report(y_test, y_pred, target_names=target_names)}"
        
        # Save artifacts
        os.makedirs("../temp_artifacts", exist_ok=True)
        with open("../temp_artifacts/logistic_regression_report.txt", "w") as f:
            f.write(cm_text)
        mlflow.log_artifact("../temp_artifacts/logistic_regression_report.txt")
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "model",
            signature=mlflow.models.infer_signature(X_train, y_pred),
            input_example=X_train[:5]
        )
        
        # Save model locally
        os.makedirs('../models', exist_ok=True)
        joblib.dump(model, '../models/logistic_regression_model.pkl')
        
        print(f"[INFO] Logistic Regression trained!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"MLflow Run ID: {run.info.run_id}")
        
        return model, accuracy, run.info.run_id

def train_random_forest(X_train, X_test, y_train, y_test, feature_names, target_names):
    """
    Train Random Forest model with comprehensive MLflow tracking.
    
    Returns:
        tuple: (model, accuracy, run_id)
    """
    with mlflow.start_run(run_name="Random_Forest_Iris") as run:
        # Define model parameters
        n_estimators = 100
        max_depth = 3
        min_samples_split = 2
        min_samples_leaf = 1
        
        # Log parameters
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("dataset", "Iris")
        mlflow.log_param("features", len(feature_names))
        mlflow.log_param("classes", len(target_names))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        
        # Train the model
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("test_samples", len(y_test))
        
        # Log feature importance
        feature_importance = model.feature_importances_
        for i, (feature, importance) in enumerate(zip(feature_names, feature_importance)):
            # Sanitize feature name for MLflow (remove special characters)
            sanitized_feature = feature.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            mlflow.log_metric(f"feature_importance_{sanitized_feature}", importance)
        
        # Log per-class metrics
        for i, class_name in enumerate(target_names):
            class_mask = y_test == i
            if np.any(class_mask):
                class_accuracy = accuracy_score(y_test[class_mask], y_pred[class_mask])
                # Sanitize class name for MLflow
                sanitized_class = class_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                mlflow.log_metric(f"accuracy_{sanitized_class}", class_accuracy)
        
        # Log confusion matrix and feature importance as artifacts
        cm = confusion_matrix(y_test, y_pred)
        report_text = f"Confusion Matrix:\n{cm}\n\nClassification Report:\n{classification_report(y_test, y_pred, target_names=target_names)}\n\nFeature Importance:\n"
        for feature, importance in zip(feature_names, feature_importance):
            report_text += f"{feature}: {importance:.4f}\n"
        
        # Save artifacts
        os.makedirs("../temp_artifacts", exist_ok=True)
        with open("../temp_artifacts/random_forest_report.txt", "w") as f:
            f.write(report_text)
        mlflow.log_artifact("../temp_artifacts/random_forest_report.txt")
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "model",
            signature=mlflow.models.infer_signature(X_train, y_pred),
            input_example=X_train[:5]
        )
        
        # Save model locally
        joblib.dump(model, '../models/random_forest_model.pkl')
        
        print(f"[INFO] Random Forest trained!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"MLflow Run ID: {run.info.run_id}")
        
        return model, accuracy, run.info.run_id

def register_best_model(lr_accuracy, lr_run_id, rf_accuracy, rf_run_id):
    """
    Register the best performing model in MLflow Model Registry.
    
    Args:
        lr_accuracy: Logistic Regression accuracy
        lr_run_id: Logistic Regression run ID
        rf_accuracy: Random Forest accuracy  
        rf_run_id: Random Forest run ID
    
    Returns:
        tuple: (best_run_id, best_model_name, best_accuracy)
    """
    if lr_accuracy > rf_accuracy:
        best_run_id = lr_run_id
        best_model_name = "LogisticRegression"
        best_accuracy = lr_accuracy
    else:
        best_run_id = rf_run_id
        best_model_name = "RandomForest"
        best_accuracy = rf_accuracy
    
    try:
        # Register the best model
        model_uri = f"runs:/{best_run_id}/model"
        registered_model = mlflow.register_model(model_uri, "BestIrisModel")
        
        # Add model version description
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        client.update_model_version(
            name="BestIrisModel",
            version=registered_model.version,
            description=f"Best performing {best_model_name} model with {best_accuracy:.4f} accuracy. Trained on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        print(f"[INFO] Best model registered in MLflow Model Registry!")
        print(f"Model Name: BestIrisModel")
        print(f"Model Type: {best_model_name}")
        print(f"Best Accuracy: {best_accuracy:.4f}")
        print(f"Model Version: {registered_model.version}")
        print(f"Model URI: {model_uri}")
        
    except Exception as e:
        print(f"[WARNING] Model registration failed: {e}")
        print("Model saved locally but not registered in MLflow")
    
    return best_run_id, best_model_name, best_accuracy

def create_experiment_summary():
    """Create a summary of all experiments."""
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        # Get current experiment
        experiment = client.get_experiment_by_name("Default")
        if experiment:
            runs = client.search_runs(experiment_ids=[experiment.experiment_id])
            
            print("\n[INFO] Experiment Summary:")
            print("=" * 50)
            for run in runs:
                print(f"Run ID: {run.info.run_id}")
                print(f"Run Name: {run.data.tags.get('mlflow.runName', 'Unknown')}")
                print(f"Model Type: {run.data.params.get('model_type', 'Unknown')}")
                print(f"Accuracy: {run.data.metrics.get('accuracy', 'Unknown')}")
                print(f"Status: {run.info.status}")
                print("-" * 30)
    except Exception as e:
        print(f"[WARNING] Could not create experiment summary: {e}")

def main():
    """
    Main function to run the complete MLflow-enabled model training pipeline.
    """
    print("[INFO] Starting MLflow-enabled model training pipeline...")
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    
    try:
        # Create or get experiment
        experiment_name = "Iris_Classification_Experiment"
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except:
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
        mlflow.set_experiment(experiment_name)
        print(f"[INFO] Using MLflow experiment: {experiment_name}")
        
        # Load and preprocess data
        X_train, X_test, y_train, y_test, feature_names, target_names, scaler = load_and_preprocess_data()
        
        # Save the scaler for later use in API
        os.makedirs('../models', exist_ok=True)
        joblib.dump(scaler, '../models/scaler.pkl')
        
        # Train both models with MLflow tracking
        print("\n[INFO] Training models with MLflow tracking...")
        lr_model, lr_accuracy, lr_run_id = train_logistic_regression(
            X_train, X_test, y_train, y_test, feature_names, target_names
        )
        
        rf_model, rf_accuracy, rf_run_id = train_random_forest(
            X_train, X_test, y_train, y_test, feature_names, target_names
        )
        
        # Register the best model
        print("\n[INFO] Registering best model...")
        best_run_id, best_model_name, best_accuracy = register_best_model(
            lr_accuracy, lr_run_id, rf_accuracy, rf_run_id
        )
        
        # Save metadata
        metadata = {
            'feature_names': list(feature_names),
            'target_names': list(target_names),
            'best_model': best_model_name,
            'best_accuracy': best_accuracy,
            'best_run_id': best_run_id,
            'logistic_regression_accuracy': lr_accuracy,
            'logistic_regression_run_id': lr_run_id,
            'random_forest_accuracy': rf_accuracy,
            'random_forest_run_id': rf_run_id,
            'experiment_name': experiment_name,
            'mlflow_tracking_uri': mlflow.get_tracking_uri(),
            'trained_timestamp': datetime.now().isoformat()
        }
        
        with open('../models/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create experiment summary
        create_experiment_summary()
        
        # Clean up temporary artifacts
        import shutil
        if os.path.exists("../temp_artifacts"):
            shutil.rmtree("../temp_artifacts")
        
        print("\n[INFO] MLflow-enabled model training pipeline completed!")
        print(f"[INFO] Models saved in: {os.path.abspath('../models')}")
        print(f"[INFO] MLflow UI: Run 'mlflow ui' in the project root to view experiments")
        print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
        
    except Exception as e:
        print(f"[ERROR] Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main() 