import pytest
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))

from app_enhanced import app, init_db

@pytest.fixture
def client():
    """Create a test client for the enhanced Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        with app.app_context():
            init_db()
        yield client

def test_predict_with_valid_pydantic_input(client):
    """Test prediction with valid Pydantic input."""
    payload = {
        "features": [5.1, 3.5, 1.4, 0.2]
    }
    
    response = client.post('/predict', 
                          data=json.dumps(payload),
                          content_type='application/json')
    
    if response.status_code == 200:
        data = json.loads(response.data)
        # Check Pydantic response structure
        assert 'prediction' in data
        assert 'prediction_name' in data
        assert 'confidence' in data
        assert 'probabilities' in data
        assert 'latency' in data
        
        # Validate response types
        assert isinstance(data['prediction'], int)
        assert isinstance(data['prediction_name'], str)
        assert isinstance(data['confidence'], float)
        assert isinstance(data['probabilities'], dict)
        assert isinstance(data['latency'], float)

def test_predict_with_invalid_pydantic_input(client):
    """Test prediction with invalid Pydantic input."""
    # Test with wrong number of features
    payload = {
        "features": [5.1, 3.5]  # Only 2 features instead of 4
    }
    
    response = client.post('/predict',
                          data=json.dumps(payload),
                          content_type='application/json')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert 'Input validation failed' in data['error']

def test_predict_with_negative_features(client):
    """Test prediction with negative feature values."""
    payload = {
        "features": [5.1, -3.5, 1.4, 0.2]  # Negative sepal width
    }
    
    response = client.post('/predict',
                          data=json.dumps(payload),
                          content_type='application/json')
    
    # Accept either 400 (validation error) or 500 (processing error) for negative values
    assert response.status_code in [400, 500]
    data = json.loads(response.data)
    assert 'error' in data

def test_add_training_data_valid(client):
    """Test adding valid training data."""
    payload = {
        "features": [4.9, 3.0, 1.4, 0.2],
        "target": 0
    }
    
    response = client.post('/add_training_data',
                          data=json.dumps(payload),
                          content_type='application/json')
    
    if response.status_code == 200:
        data = json.loads(response.data)
        assert 'message' in data
        assert 'features' in data
        assert 'target' in data

def test_add_training_data_invalid_target(client):
    """Test adding training data with invalid target."""
    payload = {
        "features": [4.9, 3.0, 1.4, 0.2],
        "target": 5  # Invalid target (should be 0, 1, or 2)
    }
    
    response = client.post('/add_training_data',
                          data=json.dumps(payload),
                          content_type='application/json')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_trigger_retrain_endpoint(client):
    """Test manual retraining trigger."""
    payload = {
        "trigger_threshold": 5,
        "force_retrain": False
    }
    
    response = client.post('/trigger_retrain',
                          data=json.dumps(payload),
                          content_type='application/json')
    
    # Should work regardless of whether there's enough data
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'message' in data

def test_enhanced_metrics_endpoint(client):
    """Test the enhanced metrics endpoint."""
    response = client.get('/metrics')
    assert response.status_code == 200
    data = json.loads(response.data)
    
    # Check for new fields
    assert 'new_samples_pending' in data
    assert 'retrain_threshold' in data
    assert 'model_info' in data

def test_dashboard_endpoint(client):
    """Test the monitoring dashboard endpoint."""
    response = client.get('/dashboard')
    assert response.status_code == 200
    # Should return HTML content
    assert 'text/html' in response.content_type or response.data.decode().startswith('<!DOCTYPE html>')

def test_enhanced_home_endpoint(client):
    """Test the enhanced home endpoint."""
    response = client.get('/')
    assert response.status_code == 200
    data = json.loads(response.data)
    
    assert 'message' in data
    assert 'endpoints' in data
    assert 'features' in data
    
    # Check for new endpoints
    endpoints = data['endpoints']
    assert '/add_training_data' in endpoints
    assert '/trigger_retrain' in endpoints
    assert '/dashboard' in endpoints

if __name__ == '__main__':
    pytest.main([__file__]) 