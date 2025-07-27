import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import numpy as np

def load_and_preprocess_data():
    """
    Load the Iris dataset from scikit-learn and preprocess it.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names, target_names)
    """
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Create a DataFrame for better data handling
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Apply standard scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the dataset to CSV for DVC tracking
    os.makedirs('data', exist_ok=True)
    
    # Save original dataset
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df['target_name'] = [target_names[i] for i in y]
    df.to_csv('data/iris_dataset.csv', index=False)
    
    # Save train/test splits
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df['target'] = y_train
    train_df.to_csv('data/iris_train.csv', index=False)
    
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['target'] = y_test
    test_df.to_csv('data/iris_test.csv', index=False)
    
    print("[INFO] Dataset loaded and preprocessed successfully!")
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    print(f"Feature names: {feature_names}")
    print(f"Target names: {target_names}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, target_names, scaler

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names, target_names, scaler = load_and_preprocess_data() 