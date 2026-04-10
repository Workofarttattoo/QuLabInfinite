#!/bin/bash
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
# QuLab AI Production Deployment Script

set -e

echo "=================================================="
echo "QuLab AI Production Deployment"
echo "=================================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
DOCKER_IMAGE="qulab-ai:latest"
CONTAINER_NAME="qulab-api"
PORT=8000

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi
print_success "Docker is installed"

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_warning "docker-compose is not installed. Will use docker commands instead."
    USE_COMPOSE=false
else
    print_success "docker-compose is installed"
    USE_COMPOSE=true
fi

# Stop existing containers
echo ""
echo "Stopping existing containers..."
if [ "$USE_COMPOSE" = true ]; then
    docker-compose down || true
else
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
fi
print_success "Stopped existing containers"

# Build Docker image
echo ""
echo "Building Docker image..."
docker build -t $DOCKER_IMAGE .
print_success "Docker image built: $DOCKER_IMAGE"

# Create logs directory
mkdir -p logs
print_success "Logs directory created"

# Start containers
echo ""
echo "Starting containers..."
if [ "$USE_COMPOSE" = true ]; then
    docker-compose up -d
else
    docker run -d \
        --name $CONTAINER_NAME \
        -p $PORT:8000 \
        -v $(pwd)/logs:/app/logs \
        -v $(pwd)/data:/app/data:ro \
        --restart unless-stopped \
        $DOCKER_IMAGE
fi
print_success "Containers started"

# Wait for API to be ready
echo ""
echo "Waiting for API to be ready..."
for i in {1..30}; do
    if curl -f http://localhost:$PORT/health > /dev/null 2>&1; then
        print_success "API is ready!"
        break
    fi
    echo "Waiting... ($i/30)"
    sleep 2
done

# Check health
echo ""
echo "Checking API health..."
HEALTH_RESPONSE=$(curl -s http://localhost:$PORT/health)
if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    print_success "Health check passed"
    echo "$HEALTH_RESPONSE" | python -m json.tool
else
    print_error "Health check failed"
    exit 1
fi

# Show running containers
echo ""
echo "Running containers:"
docker ps | grep qulab

# Show logs
echo ""
echo "Recent logs:"
docker logs --tail 20 $CONTAINER_NAME

echo ""
echo "=================================================="
print_success "Deployment complete!"
echo "=================================================="
echo ""
echo "API Documentation: http://localhost:$PORT/api/docs"
echo "Health Check: http://localhost:$PORT/health"
echo "Metrics: http://localhost:$PORT/metrics"
if [ "$USE_COMPOSE" = true ]; then
    echo "Grafana Dashboard: http://localhost:3000 (admin/admin)"
    echo "Prometheus: http://localhost:9090"
fi
echo ""
echo "To view logs: docker logs -f $CONTAINER_NAME"
echo "To stop: docker-compose down (or: docker stop $CONTAINER_NAME)"
echo ""
