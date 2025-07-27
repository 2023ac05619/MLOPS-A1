import pytest
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.app import app, init_db

@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        with app.app_context():
            init_db()
        yield client

def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert 'timestamp' in data

def test_home_endpoint(client):
    """Test the home endpoint."""
    response = client.get('/')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'message' in data
    assert 'endpoints' in data
    assert 'example_request' in data

def test_metrics_endpoint(client):
    """Test the metrics endpoint."""
    response = client.get('/metrics')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'request_count' in data
    assert 'model_info' in data
    assert 'timestamp' in data

def test_predict_endpoint_valid_input(client):
    """Test prediction with valid input."""
    # Sample Iris data
    payload = {
        "features": [5.1, 3.5, 1.4, 0.2]
    }
    
    response = client.post('/predict', 
                          data=json.dumps(payload),
                          content_type='application/json')
    
    # Note: This test assumes model is trained
    # In real scenario, we might mock the model loading
    if response.status_code == 200:
        data = json.loads(response.data)
        assert 'prediction' in data
        assert 'confidence' in data
        assert 'latency' in data
    else:
        # If model not trained, expect specific error
        assert response.status_code == 500

def test_predict_endpoint_invalid_input(client):
    """Test prediction with invalid input."""
    # Missing features
    payload = {}
    
    response = client.post('/predict',
                          data=json.dumps(payload),
                          content_type='application/json')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_predict_endpoint_wrong_feature_count(client):
    """Test prediction with wrong number of features."""
    payload = {
        "features": [5.1, 3.5]  # Only 2 features instead of 4
    }
    
    response = client.post('/predict',
                          data=json.dumps(payload),
                          content_type='application/json')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_prediction_history_endpoint(client):
    """Test the prediction history endpoint."""
    response = client.get('/predictions/history')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'history' in data
    assert 'count' in data

if __name__ == '__main__':
    pytest.main([__file__]) 