import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import json

def load_and_preprocess_data():
    """
    Load the Iris dataset from scikit-learn and preprocess it.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names, target_names)
    """
    print("Loading Iris dataset...")
    
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Get metadata
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Target classes: {target_names}")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Apply standard scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create data directory and save datasets as CSV
    os.makedirs('data', exist_ok=True)
    
    # Save original dataset as simple CSV
    with open('data/iris_dataset.csv', 'w') as f:
        # Write header
        f.write(','.join(feature_names) + ',target,target_name\n')
        # Write data
        for i in range(len(X)):
            row = ','.join([str(x) for x in X[i]]) + f',{y[i]},{target_names[y[i]]}\n'
            f.write(row)
    
    # Save train/test splits
    with open('data/iris_train.csv', 'w') as f:
        f.write(','.join(feature_names) + ',target\n')
        for i in range(len(X_train)):
            row = ','.join([str(x) for x in X_train[i]]) + f',{y_train[i]}\n'
            f.write(row)
    
    with open('data/iris_test.csv', 'w') as f:
        f.write(','.join(feature_names) + ',target\n')
        for i in range(len(X_test)):
            row = ','.join([str(x) for x in X_test[i]]) + f',{y_test[i]}\n'
            f.write(row)
    
    print("[INFO] Dataset loaded and preprocessed successfully!")
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, target_names, scaler

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names, target_names, scaler = load_and_preprocess_data() 