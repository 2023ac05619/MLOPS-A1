import pytest
import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocess_data import load_and_preprocess_data

def test_load_and_preprocess_data():
    """Test data loading and preprocessing function."""
    try:
        X_train, X_test, y_train, y_test, feature_names, target_names, scaler = load_and_preprocess_data()
        
        # Test shapes
        assert X_train.shape[1] == 4  # 4 features for Iris
        assert X_test.shape[1] == 4
        assert len(feature_names) == 4
        assert len(target_names) == 3  # 3 classes for Iris
        
        # Test that we have some data
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]
        
        # Test scaling (scaled data should have approximately zero mean)
        assert abs(np.mean(X_train)) < 0.1
        
        # Test that files are created
        assert os.path.exists('data/iris_dataset.csv')
        assert os.path.exists('data/iris_train.csv')
        assert os.path.exists('data/iris_test.csv')
        
        print("✅ Data preprocessing test passed!")
        
    except Exception as e:
        pytest.fail(f"Data preprocessing failed: {e}")

def test_iris_dataset_integrity():
    """Test that the saved Iris dataset has the correct properties."""
    if os.path.exists('data/iris_dataset.csv'):
        df = pd.read_csv('data/iris_dataset.csv')
        
        # Test dataset properties
        assert len(df) == 150  # Iris has 150 samples
        assert len(df.columns) == 6  # 4 features + target + target_name
        assert df['target'].nunique() == 3  # 3 classes
        assert set(df['target_name'].unique()) == {'setosa', 'versicolor', 'virginica'}
        
        print("✅ Iris dataset integrity test passed!")

def test_train_test_split_integrity():
    """Test that train/test splits maintain class distribution."""
    if os.path.exists('data/iris_train.csv') and os.path.exists('data/iris_test.csv'):
        train_df = pd.read_csv('data/iris_train.csv')
        test_df = pd.read_csv('data/iris_test.csv')
        
        # Test that both datasets have all classes
        assert train_df['target'].nunique() == 3
        assert test_df['target'].nunique() == 3
        
        # Test approximate split ratio (80/20)
        total_samples = len(train_df) + len(test_df)
        train_ratio = len(train_df) / total_samples
        assert 0.75 <= train_ratio <= 0.85  # Should be around 0.8
        
        print("✅ Train/test split integrity test passed!")

if __name__ == '__main__':
    pytest.main([__file__]) 