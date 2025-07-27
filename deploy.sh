#!/bin/bash
echo "Starting Local Deployment Pipeline..."

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Create logs directory if it doesn't exist
mkdir -p logs

# Exit on any error
set -e

log "Starting local deployment process..."

# Step 1: Clean up existing containers
log "Cleaning up existing containers..."
docker stop offline-ml-api 2>/dev/null || true
docker rm offline-ml-api 2>/dev/null || true
log "[SUCCESS] Cleaned up existing container"

# Step 2: Build Docker image
log "Building Docker image..."
if ! docker build -t offline-ml-api .; then
    log "[ERROR] Docker build failed!"
    exit 1
fi
log "[SUCCESS] Docker image built successfully"

# Step 3: Start container
log "Starting container..."
if ! docker run -d \
    --name offline-ml-api \
    -p 5000:5000 \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/models:/app/models \
    offline-ml-api; then
    log "[ERROR] Container startup failed!"
    exit 1
fi
log "[SUCCESS] Container started successfully"

# Step 4: Wait for service to be ready
log "Waiting for service to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:5000/health > /dev/null 2>&1; then
        log "[SUCCESS] Service is healthy and ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        log "[ERROR] Service health check failed after 30 attempts"
        docker logs offline-ml-api
        exit 1
    fi
    sleep 2
done

# Step 5: Test endpoints
log "Testing prediction endpoint..."
if ! curl -s -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [5.1, 3.5, 1.4, 0.2]}' > /dev/null; then
    log "[ERROR] Prediction endpoint test failed!"
    exit 1
fi
log "[SUCCESS] Prediction endpoint test passed"

if ! curl -s http://localhost:5000/metrics > /dev/null; then
    log "[ERROR] Metrics endpoint test failed!"
    exit 1
fi
log "[SUCCESS] Metrics endpoint test passed"

# Step 6: Display service information
log "Deployment completed successfully!"

echo ""
echo "Service Information:"
echo "==================="
echo "Container Name: offline-ml-api"
echo "Host Port: 5000"
echo "Status: Running"
echo "Health Check: http://localhost:5000/health"

echo ""
echo "API Endpoints:"
echo "=============="
echo "- GET  http://localhost:5000/health"
echo "- POST http://localhost:5000/predict"
echo "- GET  http://localhost:5000/metrics"
echo "- GET  http://localhost:5000/predictions/history"

echo ""
echo "Useful Commands:"
echo "================"
echo "- View logs: docker logs offline-ml-api"
echo "- Stop service: docker stop offline-ml-api"
echo "- Remove container: docker rm offline-ml-api"
echo "- Rebuild: ./deploy.sh"

log "Local deployment pipeline completed successfully!" 