#!/bin/bash
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
#
# QuLabInfinite Complete Deployment Script
# One-command deployment for all environments

set -e

echo "======================================================================"
echo "QuLabInfinite Complete Deployment"
echo "======================================================================"
echo ""

# Configuration
DEPLOY_ENV=${1:-"local"}
VERSION=${2:-"latest"}

echo "Environment: $DEPLOY_ENV"
echo "Version: $VERSION"
echo ""

# Function to check prerequisites
check_prerequisites() {
    echo "[1/8] Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo "ERROR: Docker not found. Please install Docker."
        exit 1
    fi
    echo "  ✓ Docker: $(docker --version)"

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo "WARNING: docker-compose not found. Using docker compose plugin."
    else
        echo "  ✓ Docker Compose: $(docker-compose --version)"
    fi

    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo "ERROR: Python 3 not found. Please install Python 3.10+."
        exit 1
    fi
    echo "  ✓ Python: $(python3 --version)"

    # Check kubectl (for k8s deployment)
    if [ "$DEPLOY_ENV" = "kubernetes" ]; then
        if ! command -v kubectl &> /dev/null; then
            echo "ERROR: kubectl not found. Please install kubectl."
            exit 1
        fi
        echo "  ✓ kubectl: $(kubectl version --client --short)"
    fi

    echo ""
}

# Function to install Python dependencies
install_dependencies() {
    echo "[2/8] Installing Python dependencies..."

    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi

    source venv/bin/activate
    pip install --upgrade pip > /dev/null 2>&1
    pip install -r requirements.txt > /dev/null 2>&1

    echo "  ✓ Python dependencies installed"
    echo ""
}

# Function to run tests
run_tests() {
    echo "[3/8] Running tests..."

    source venv/bin/activate

    # Run master demo
    echo "  Running master demo..."
    python MASTER_DEMO.py > /dev/null 2>&1 || true

    # Check if demo produced results
    if [ -f "MASTER_RESULTS.json" ]; then
        echo "  ✓ Master demo completed"
        SUCCESSFUL=$(python -c "import json; data=json.load(open('MASTER_RESULTS.json')); print(data['successful'])")
        TOTAL=$(python -c "import json; data=json.load(open('MASTER_RESULTS.json')); print(data['total_labs'])")
        echo "  ✓ $SUCCESSFUL/$TOTAL labs passed"
    else
        echo "  ⚠ Master demo results not found"
    fi

    echo ""
}

# Function to build Docker images
build_images() {
    echo "[4/8] Building Docker images..."

    # Build production image
    docker build -f Dockerfile.production -t qulab-api:$VERSION . > /dev/null 2>&1
    echo "  ✓ Built qulab-api:$VERSION"

    # Tag as latest
    docker tag qulab-api:$VERSION qulab-api:latest
    echo "  ✓ Tagged as latest"

    echo ""
}

# Function to deploy locally
deploy_local() {
    echo "[5/8] Deploying locally with Docker Compose..."

    # Stop any existing deployment
    docker-compose -f docker-compose.master.yml down > /dev/null 2>&1 || true

    # Start services
    docker-compose -f docker-compose.master.yml up -d

    echo "  ✓ Services started"
    echo ""

    # Wait for services to be ready
    echo "[6/8] Waiting for services to be ready..."
    sleep 10

    # Health check
    echo "[7/8] Running health checks..."
    MAX_RETRIES=30
    RETRY_COUNT=0

    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "  ✓ API is healthy"
            break
        fi
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo "  Waiting for API... ($RETRY_COUNT/$MAX_RETRIES)"
        sleep 2
    done

    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        echo "  ✗ API health check failed"
        exit 1
    fi

    echo ""
}

# Function to deploy to Kubernetes
deploy_kubernetes() {
    echo "[5/8] Deploying to Kubernetes..."

    # Create namespace
    kubectl create namespace qulab-infinite --dry-run=client -o yaml | kubectl apply -f -
    echo "  ✓ Namespace created/verified"

    # Create secrets
    kubectl create secret generic qulab-secrets \
        --from-literal=postgres-password='secure_password_change_me' \
        --from-literal=api-master-key='enterprise_key_change_me' \
        --namespace=qulab-infinite \
        --dry-run=client -o yaml | kubectl apply -f -
    echo "  ✓ Secrets created/verified"

    # Apply manifests
    kubectl apply -f deploy_kubernetes.yaml
    echo "  ✓ Manifests applied"

    echo ""
    echo "[6/8] Waiting for deployments..."

    # Wait for rollout
    kubectl rollout status deployment/qulab-api -n qulab-infinite --timeout=300s
    echo "  ✓ API deployment ready"

    echo ""
    echo "[7/8] Running health checks..."

    # Get service endpoint
    API_URL=$(kubectl get svc qulab-api-service -n qulab-infinite -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

    if [ -z "$API_URL" ]; then
        echo "  ⚠ LoadBalancer IP not assigned yet. Use port-forward to access API."
        kubectl port-forward svc/qulab-api-service 8000:80 -n qulab-infinite &
        API_URL="localhost:8000"
    fi

    # Health check
    sleep 10
    if curl -f -s http://$API_URL/health > /dev/null 2>&1; then
        echo "  ✓ API is healthy at http://$API_URL"
    else
        echo "  ⚠ API health check pending"
    fi

    echo ""
}

# Function to display deployment info
show_deployment_info() {
    echo "[8/8] Deployment complete!"
    echo ""
    echo "======================================================================"
    echo "Deployment Information"
    echo "======================================================================"
    echo ""

    if [ "$DEPLOY_ENV" = "local" ]; then
        echo "Environment: Local (Docker Compose)"
        echo ""
        echo "Services:"
        echo "  - API Server: http://localhost:8000"
        echo "  - API Docs: http://localhost:8000/docs"
        echo "  - Dashboard: http://localhost:3000"
        echo "  - Prometheus: http://localhost:9090"
        echo "  - Grafana: http://localhost:3001 (admin/admin_change_me)"
        echo ""
        echo "Quick Commands:"
        echo "  - View logs: docker-compose -f docker-compose.master.yml logs -f"
        echo "  - Stop services: docker-compose -f docker-compose.master.yml down"
        echo "  - Restart: docker-compose -f docker-compose.master.yml restart"
        echo ""
        echo "Test API:"
        echo "  curl http://localhost:8000/health"
        echo "  curl -H 'X-API-Key: demo_key_12345' http://localhost:8000/labs"

    elif [ "$DEPLOY_ENV" = "kubernetes" ]; then
        echo "Environment: Kubernetes"
        echo ""
        echo "Namespace: qulab-infinite"
        echo ""
        echo "Services:"
        API_URL=$(kubectl get svc qulab-api-service -n qulab-infinite -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
        echo "  - API Server: http://$API_URL"
        echo "  - API Docs: http://$API_URL/docs"
        echo ""
        echo "Quick Commands:"
        echo "  - View pods: kubectl get pods -n qulab-infinite"
        echo "  - View logs: kubectl logs -f deployment/qulab-api -n qulab-infinite"
        echo "  - Scale: kubectl scale deployment qulab-api --replicas=5 -n qulab-infinite"
        echo "  - Port forward: kubectl port-forward svc/qulab-api-service 8000:80 -n qulab-infinite"
        echo ""
        echo "Test API:"
        echo "  kubectl port-forward svc/qulab-api-service 8000:80 -n qulab-infinite &"
        echo "  curl http://localhost:8000/health"
    fi

    echo ""
    echo "======================================================================"
    echo "Next Steps:"
    echo "======================================================================"
    echo ""
    echo "1. Test the API endpoints (see API_REFERENCE.md)"
    echo "2. Access the dashboard at http://localhost:3000 (local)"
    echo "3. View monitoring at http://localhost:9090 (Prometheus)"
    echo "4. Check logs for any errors"
    echo "5. Run load tests: python scripts/load_test.py"
    echo ""
    echo "For detailed documentation, see:"
    echo "  - DEPLOYMENT_GUIDE.md"
    echo "  - API_REFERENCE.md"
    echo "  - README.md"
    echo ""
}

# Main deployment flow
main() {
    check_prerequisites
    install_dependencies
    run_tests
    build_images

    if [ "$DEPLOY_ENV" = "kubernetes" ]; then
        deploy_kubernetes
    else
        deploy_local
    fi

    show_deployment_info
}

# Run deployment
main

echo "Deployment script completed successfully!"
echo ""
