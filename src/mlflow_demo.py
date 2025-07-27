#!/usr/bin/env python3
"""
MLflow Experiment Tracking Demonstration Script

This script demonstrates how to:
1. Query MLflow experiments
2. Retrieve run information
3. Load models from MLflow
4. Access tracked parameters and metrics
"""

import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import json

# Set MLflow tracking URI
mlflow.set_tracking_uri(f"file://{os.path.abspath('../mlruns')}")

def demonstrate_mlflow_tracking():
    """Demonstrate MLflow experiment tracking capabilities."""
    print("[INFO] MLflow Experiment Tracking Demonstration")
    print("=" * 60)
    
    client = MlflowClient()
    
    # 1. List all experiments
    print("\n[INFO] Available Experiments:")
    experiments = client.search_experiments()
    for exp in experiments:
        print(f"  - {exp.name} (ID: {exp.experiment_id})")
    
    # 2. Get runs from Iris Classification Experiment
    try:
        iris_experiment = client.get_experiment_by_name("Iris_Classification_Experiment")
        if iris_experiment:
            print(f"\n[INFO] Iris Classification Experiment (ID: {iris_experiment.experiment_id})")
            runs = client.search_runs(experiment_ids=[iris_experiment.experiment_id])
            
            print(f"\n[INFO] Found {len(runs)} runs:")
            
            run_data = []
            for run in runs:
                run_info = {
                    'run_id': run.info.run_id,
                    'run_name': run.data.tags.get('mlflow.runName', 'Unknown'),
                    'model_type': run.data.params.get('model_type', 'Unknown'),
                    'status': run.info.status,
                    'start_time': run.info.start_time,
                }
                
                # Add metrics
                for metric_name, metric_value in run.data.metrics.items():
                    run_info[metric_name] = metric_value
                
                # Add key parameters
                for param_name, param_value in run.data.params.items():
                    if param_name in ['C', 'n_estimators', 'max_depth']:
                        run_info[param_name] = param_value
                
                run_data.append(run_info)
            
            # Display as DataFrame for better formatting
            df = pd.DataFrame(run_data)
            print(df.to_string(index=False))
            
    except Exception as e:
        print(f"Error accessing experiment: {e}")
    
    # 3. Demonstrate model loading from MLflow
    print("\n[INFO] Model Loading from MLflow:")
    try:
        # Load metadata to get best model run ID
        with open('../models/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        best_run_id = metadata['best_run_id']
        best_model_type = metadata['best_model']
        
        print(f"[INFO] Best Model: {best_model_type}")
        print(f"[INFO] Best Run ID: {best_run_id}")
        print(f"[INFO] Best Accuracy: {metadata['best_accuracy']:.4f}")
        
        # Load model from MLflow
        model_uri = f"runs:/{best_run_id}/model"
        try:
            loaded_model = mlflow.sklearn.load_model(model_uri)
            print(f"[INFO] Successfully loaded model from MLflow")
            print(f"[INFO] Model type: {type(loaded_model).__name__}")
            
            # Show model parameters
            if hasattr(loaded_model, 'get_params'):
                params = loaded_model.get_params()
                print(f"[INFO] Model parameters: {params}")
                
        except Exception as e:
            print(f"[WARNING] Could not load model from MLflow: {e}")
            print("Loading from local file instead...")
            import joblib
            if best_model_type == "RandomForest":
                loaded_model = joblib.load('../models/random_forest_model.pkl')
            else:
                loaded_model = joblib.load('../models/logistic_regression_model.pkl')
            print(f"[INFO] Loaded {best_model_type} from local file")
            
    except Exception as e:
        print(f"Error in model loading demonstration: {e}")
    
    # 4. Show registered models
    print("\n[INFO] Registered Models:")
    try:
        registered_models = client.search_registered_models()
        for model in registered_models:
            print(f"  Model: {model.name}")
            latest_version = client.get_latest_versions(model.name)[0]
            print(f"    Latest Version: {latest_version.version}")
            print(f"    Description: {latest_version.description}")
            print(f"    Stage: {latest_version.current_stage}")
    except Exception as e:
        print(f"No registered models found or error: {e}")
    
    # 5. Show experiment summary
    print("\n[INFO] Experiment Summary:")
    print(f"[INFO] MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"[INFO] Total Experiments: {len(experiments)}")
    if iris_experiment:
        print(f"[INFO] Total Runs in Iris Experiment: {len(runs)}")
        
        # Show best metrics
        best_accuracy = max([run.data.metrics.get('accuracy', 0) for run in runs])
        best_f1 = max([run.data.metrics.get('f1_score', 0) for run in runs])
        
        print(f"[INFO] Best Accuracy Achieved: {best_accuracy:.4f}")
        print(f"[INFO] Best F1-Score Achieved: {best_f1:.4f}")

def show_detailed_run_info(run_id=None):
    """Show detailed information about a specific run."""
    client = MlflowClient()
    
    if not run_id:
        # Use best model run ID from metadata
        try:
            with open('../models/metadata.json', 'r') as f:
                metadata = json.load(f)
            run_id = metadata['best_run_id']
        except:
            print("No run ID provided and couldn't load from metadata")
            return
    
    print(f"\n[INFO] Detailed Run Information (ID: {run_id})")
    print("=" * 60)
    
    try:
        run = client.get_run(run_id)
        
        print(f"[INFO] Run Name: {run.data.tags.get('mlflow.runName', 'Unknown')}")
        print(f"[INFO] Status: {run.info.status}")
        print(f"[INFO] Start Time: {run.info.start_time}")
        print(f"[INFO] End Time: {run.info.end_time}")
        
        print("\n[INFO] Parameters:")
        for key, value in run.data.params.items():
            print(f"  {key}: {value}")
        
        print("\n[INFO] Metrics:")
        for key, value in run.data.metrics.items():
            print(f"  {key}: {value:.4f}")
        
        print("\n[INFO] Tags:")
        for key, value in run.data.tags.items():
            print(f"  {key}: {value}")
        
        print("\n[INFO] Artifacts:")
        artifacts = client.list_artifacts(run_id)
        for artifact in artifacts:
            print(f"  {artifact.path} ({artifact.file_size} bytes)")
            
    except Exception as e:
        print(f"Error retrieving run information: {e}")

if __name__ == "__main__":
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