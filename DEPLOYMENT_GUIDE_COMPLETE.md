# QuLabInfinite Complete Deployment Guide

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

## Table of Contents

1. [Quick Start](#quick-start)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Monitoring & Logging](#monitoring--logging)
7. [Security Configuration](#security-configuration)
8. [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

- Python 3.11+
- Docker 20.10+
- Kubernetes 1.24+ (for K8s deployment)
- 8GB RAM minimum, 16GB recommended
- 20GB disk space

### Run Master Demo

```bash
cd /Users/noone/QuLabInfinite
python MASTER_DEMO_COMPLETE.py
```

This will:
- Execute all 20 labs sequentially
- Generate comprehensive results in `MASTER_RESULTS_COMPLETE.json`
- Create summary report in `MASTER_SUMMARY_COMPLETE.txt`
- Validate all lab functionality

Expected output:
```
Total Labs Run: 20
Successful: 20
Failed: 0
Success Rate: 100.0%
```

## Local Development

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/QuLabInfinite.git
cd QuLabInfinite
```

### 2. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run Unified API

```bash
python api/unified_api.py
```

Access at: `http://localhost:9000`

API Documentation: `http://localhost:9000/docs`

### 4. Test Individual Labs

```bash
# Alzheimer's Lab
python alzheimers_early_detection.py

# Quantum Lab
cd quantum_lab && python demo.py

# Materials Lab
cd materials_lab && python demo.py
```

## Docker Deployment

### Single Container

```bash
# Build image
docker build -t qulabinfinite:latest -f Dockerfile.complete .

# Run container
docker run -d \
  -p 9000:9000 \
  --name qulab-api \
  qulabinfinite:latest

# Check logs
docker logs -f qulab-api

# Health check
curl http://localhost:9000/health
```

### Docker Compose (Complete Stack)

```bash
# Start all services
docker-compose -f docker-compose-complete.yml up -d

# Check status
docker-compose -f docker-compose-complete.yml ps

# View logs
docker-compose -f docker-compose-complete.yml logs -f api-gateway

# Stop all services
docker-compose -f docker-compose-complete.yml down
```

Services included:
- **API Gateway** (port 9000): Unified API server
- **Medical Labs** (ports 8001-8010): 10 medical diagnostic labs
- **Scientific Labs** (ports 9001-9010): 10 scientific simulation labs
- **PostgreSQL** (port 5432): Primary database
- **Redis** (port 6379): Caching layer
- **Nginx** (ports 80, 443): Reverse proxy
- **Prometheus** (port 9090): Metrics collection
- **Grafana** (port 3001): Monitoring dashboards

## Kubernetes Deployment

### 1. Create Namespace and Apply Configuration

```bash
# Apply complete Kubernetes configuration
kubectl apply -f kubernetes-complete.yaml

# Check deployment status
kubectl get pods -n qulab
kubectl get services -n qulab
kubectl get ingress -n qulab
```

### 2. Scale Deployments

```bash
# Scale API gateway
kubectl scale deployment api-gateway -n qulab --replicas=5

# Scale medical labs
kubectl scale deployment medical-labs -n qulab --replicas=3

# Scale scientific labs
kubectl scale deployment scientific-labs -n qulab --replicas=3
```

### 3. Update Deployments

```bash
# Update API gateway image
kubectl set image deployment/api-gateway \
  api-gateway=qulabinfinite/api-gateway:v1.1.0 \
  -n qulab

# Check rollout status
kubectl rollout status deployment/api-gateway -n qulab

# Rollback if needed
kubectl rollout undo deployment/api-gateway -n qulab
```

### 4. Access Services

```bash
# Port forward API gateway
kubectl port-forward service/api-gateway-service 9000:9000 -n qulab

# Access at http://localhost:9000

# Get external IP (if using LoadBalancer)
kubectl get service api-gateway-service -n qulab
```

### 5. Horizontal Pod Autoscaling

HPA is pre-configured in `kubernetes-complete.yaml`:
- **Min replicas:** 3
- **Max replicas:** 10
- **CPU target:** 70%
- **Memory target:** 80%

Monitor autoscaling:
```bash
kubectl get hpa -n qulab
kubectl describe hpa api-gateway-hpa -n qulab
```

## Cloud Deployment

### AWS EKS

```bash
# Create EKS cluster
eksctl create cluster \
  --name qulab-production \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type t3.xlarge \
  --nodes 3 \
  --nodes-min 3 \
  --nodes-max 10 \
  --managed

# Configure kubectl
aws eks update-kubeconfig --name qulab-production --region us-west-2

# Deploy QuLabInfinite
kubectl apply -f kubernetes-complete.yaml

# Install ingress controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/aws/deploy.yaml

# Configure DNS
# Point api.qulabinfinite.com to LoadBalancer external IP
```

### Google Cloud GKE

```bash
# Create GKE cluster
gcloud container clusters create qulab-production \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-4 \
  --enable-autoscaling \
  --min-nodes 3 \
  --max-nodes 10

# Get credentials
gcloud container clusters get-credentials qulab-production --zone us-central1-a

# Deploy QuLabInfinite
kubectl apply -f kubernetes-complete.yaml
```

### Azure AKS

```bash
# Create resource group
az group create --name qulab-rg --location eastus

# Create AKS cluster
az aks create \
  --resource-group qulab-rg \
  --name qulab-production \
  --node-count 3 \
  --enable-cluster-autoscaler \
  --min-count 3 \
  --max-count 10 \
  --vm-set-type VirtualMachineScaleSets \
  --node-vm-size Standard_D4s_v3

# Get credentials
az aks get-credentials --resource-group qulab-rg --name qulab-production

# Deploy QuLabInfinite
kubectl apply -f kubernetes-complete.yaml
```

## Monitoring & Logging

### Prometheus Metrics

Access Prometheus at `http://localhost:9090` (via port-forward or ingress)

Key metrics:
- `qulab_requests_total`: Total API requests
- `qulab_request_duration_seconds`: Request latency histogram
- `qulab_active_connections`: Active WebSocket connections
- `qulab_lab_computations_total`: Computations per lab

### Grafana Dashboards

Access Grafana at `http://localhost:3001`

Default credentials: `admin/admin` (change immediately)

Pre-configured dashboards:
- **API Overview**: Request rates, latencies, error rates
- **Lab Performance**: Computation times by lab
- **System Resources**: CPU, memory, disk usage
- **Database Metrics**: PostgreSQL performance
- **Cache Performance**: Redis hit/miss rates

### Application Logs

```bash
# Docker Compose
docker-compose -f docker-compose-complete.yml logs -f api-gateway

# Kubernetes
kubectl logs -f deployment/api-gateway -n qulab

# View logs of specific pod
kubectl logs -f <pod-name> -n qulab

# View logs of previous container (if crashed)
kubectl logs --previous <pod-name> -n qulab
```

### Log Aggregation

For production, integrate with:
- **ELK Stack**: Elasticsearch, Logstash, Kibana
- **Splunk**: Enterprise log management
- **Datadog**: Cloud monitoring platform
- **New Relic**: Application performance monitoring

## Security Configuration

### 1. Update API Keys

Edit `/Users/noone/QuLabInfinite/api/unified_api.py`:

```python
API_KEYS = {
    "your_secure_free_key": {"tier": "free", "rate_limit": 100},
    "your_secure_pro_key": {"tier": "pro", "rate_limit": 1000},
    "your_secure_enterprise_key": {"tier": "enterprise", "rate_limit": 10000}
}
```

Or use environment variables:
```bash
export QULAB_FREE_KEY="your_secure_free_key"
export QULAB_PRO_KEY="your_secure_pro_key"
export QULAB_ENTERPRISE_KEY="your_secure_enterprise_key"
```

### 2. SSL/TLS Configuration

For Kubernetes with Let's Encrypt:

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml

# Create ClusterIssuer
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

Certificate will be automatically provisioned for `api.qulabinfinite.com`.

### 3. Network Security

Enable Kubernetes NetworkPolicies (included in `kubernetes-complete.yaml`):
- API Gateway can only receive traffic from ingress
- API Gateway can only connect to PostgreSQL and Redis
- Labs can only be accessed through API Gateway

### 4. Database Security

```bash
# Update PostgreSQL password
kubectl create secret generic qulab-secrets \
  --from-literal=POSTGRES_PASSWORD='your_secure_password' \
  --dry-run=client -o yaml | kubectl apply -n qulab -f -

# Restart PostgreSQL
kubectl rollout restart deployment/postgres -n qulab
```

### 5. Regular Security Updates

```bash
# Update all images
docker pull python:3.11-slim
docker pull postgres:15-alpine
docker pull redis:7-alpine

# Rebuild QuLabInfinite images
docker build -t qulabinfinite/api-gateway:latest -f Dockerfile.complete .

# Update Kubernetes deployments
kubectl set image deployment/api-gateway \
  api-gateway=qulabinfinite/api-gateway:latest \
  -n qulab
```

## Troubleshooting

### Issue: API Gateway not starting

**Check logs:**
```bash
docker logs qulab-api
# or
kubectl logs deployment/api-gateway -n qulab
```

**Common causes:**
- Database connection failed: Check PostgreSQL is running
- Port already in use: Change port or stop conflicting service
- Missing dependencies: Reinstall requirements

### Issue: High latency

**Check:**
1. Resource usage: `docker stats` or `kubectl top pods -n qulab`
2. Database performance: Check slow query log
3. Network issues: Test connectivity between services

**Solutions:**
- Scale up replicas: `kubectl scale deployment/api-gateway --replicas=5 -n qulab`
- Increase resource limits in Kubernetes manifests
- Add database connection pooling
- Enable Redis caching

### Issue: Rate limit errors

**Check current limits:**
```bash
curl -H "Authorization: Bearer your_key" \
  http://localhost:9000/analytics
```

**Solutions:**
- Upgrade API tier
- Implement request batching
- Use caching for repeated requests

### Issue: Database connection errors

**Check PostgreSQL status:**
```bash
docker-compose -f docker-compose-complete.yml ps postgres
kubectl get pods -l app=postgres -n qulab
```

**Test connection:**
```bash
docker exec -it qulab-postgres psql -U qulab -d qulab
# or
kubectl exec -it deployment/postgres -n qulab -- psql -U qulab -d qulab
```

### Issue: Out of memory

**Check memory usage:**
```bash
docker stats
kubectl top pods -n qulab
```

**Solutions:**
- Increase memory limits in Docker Compose or Kubernetes
- Enable swap (not recommended for production)
- Optimize lab algorithms for lower memory usage
- Scale horizontally instead of vertically

### Support Channels

- **Documentation**: https://docs.qulabinfinite.com
- **GitHub Issues**: https://github.com/yourusername/QuLabInfinite/issues
- **Email Support**: support@qulabinfinite.com
- **Enterprise Support**: enterprise@qulabinfinite.com (24/7)
- **Community Forum**: https://forum.qulabinfinite.com

## Performance Benchmarks

### Expected Performance (single instance)

| Lab Category | Requests/sec | Avg Latency | P99 Latency |
|-------------|--------------|-------------|-------------|
| Medical Labs | 500-1000 | 20-50ms | 100ms |
| Quantum Simulations | 10-50 | 200-1000ms | 2000ms |
| Materials Predictions | 100-500 | 50-200ms | 500ms |
| Chemistry Simulations | 50-200 | 100-500ms | 1000ms |

### Scalability

- **Horizontal scaling**: Up to 50 replicas per deployment
- **Throughput**: 50,000+ requests/sec with 10 API gateway replicas
- **Concurrent users**: 10,000+ with proper load balancing
- **Database**: Up to 1M records with PostgreSQL optimization

---

**Version:** 1.0.0
**Last Updated:** November 3, 2025
**Copyright:** Corporation of Light - Patent Pending
