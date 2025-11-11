# QuLabInfinite Deployment Guide

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Cloud Deployments](#cloud-deployments)
6. [Monitoring & Observability](#monitoring--observability)
7. [Security Hardening](#security-hardening)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Minimum:**
- 4 CPU cores
- 16 GB RAM
- 50 GB storage
- Ubuntu 20.04+ / macOS 12+ / Windows 11

**Recommended:**
- 8+ CPU cores
- 32+ GB RAM
- 200 GB SSD storage
- Linux (Ubuntu 22.04 LTS)

### Software Dependencies

- Python 3.10+
- Docker 24.0+
- Docker Compose 2.0+
- Kubernetes 1.27+ (for K8s deployment)
- PostgreSQL 15+ (or Docker)
- Redis 7+ (or Docker)

## Local Development

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/QuLabInfinite.git
cd QuLabInfinite
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set Environment Variables

```bash
cp .env.example .env
# Edit .env with your settings
```

Required variables:
```bash
QULAB_ENV=development
API_KEY=demo_key_12345
POSTGRES_URL=postgresql://localhost:5432/qulab
REDIS_URL=redis://localhost:6379
LOG_LEVEL=debug
```

### 5. Initialize Database

```bash
# Start PostgreSQL and Redis
docker-compose up -d postgres redis

# Run migrations
python database/populate_db.py
```

### 6. Run API Server

```bash
# Development mode with auto-reload
python -m uvicorn api.unified_api:app --reload --host 0.0.0.0 --port 8000

# Or use the master script
python api/unified_api.py
```

### 7. Test Installation

```bash
# Health check
curl http://localhost:8000/health

# List labs
curl -H "X-API-Key: demo_key_12345" http://localhost:8000/labs

# Run master demo
python MASTER_DEMO.py
```

## Docker Deployment

### Single Container

```bash
# Build image
docker build -f Dockerfile.production -t qulab-api:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -e API_KEY=your_key \
  --name qulab-api \
  qulab-api:latest
```

### Docker Compose (Complete Stack)

```bash
# Start all services
docker-compose -f docker-compose.master.yml up -d

# View logs
docker-compose logs -f qulab-api

# Scale API servers
docker-compose up -d --scale qulab-api=3

# Stop all services
docker-compose down
```

**Services included:**
- `qulab-api`: Unified API server (port 8000)
- `qulab-dashboard`: Web dashboard (port 3000)
- `postgres`: Database (port 5432)
- `redis`: Cache/rate limiting (port 6379)
- `prometheus`: Metrics (port 9090)
- `grafana`: Visualization (port 3001)
- `nginx`: Reverse proxy (ports 80, 443)

### Verify Deployment

```bash
# Check all services are running
docker-compose ps

# Test API
curl http://localhost:8000/health

# Access dashboard
open http://localhost:3000

# View metrics
open http://localhost:9090
```

## Kubernetes Deployment

### 1. Prepare Cluster

```bash
# Create namespace
kubectl create namespace qulab-infinite

# Set context
kubectl config set-context --current --namespace=qulab-infinite
```

### 2. Create Secrets

```bash
# Create API key secret
kubectl create secret generic qulab-secrets \
  --from-literal=postgres-password='secure_password' \
  --from-literal=api-master-key='enterprise_key'
```

### 3. Deploy Services

```bash
# Apply all manifests
kubectl apply -f deploy_kubernetes.yaml

# Wait for deployments
kubectl rollout status deployment/qulab-api
kubectl rollout status statefulset/postgres
```

### 4. Verify Deployment

```bash
# Check pods
kubectl get pods

# Check services
kubectl get svc

# Check ingress
kubectl get ingress

# View logs
kubectl logs -f deployment/qulab-api
```

### 5. Configure Ingress (Optional)

```bash
# Install cert-manager for TLS
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer for Let's Encrypt
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@qulab.io
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

### 6. Scale Deployment

```bash
# Manual scaling
kubectl scale deployment qulab-api --replicas=5

# Auto-scaling (HPA already configured)
kubectl get hpa qulab-api-hpa
```

## Cloud Deployments

### AWS EKS

```bash
# Create EKS cluster
eksctl create cluster \
  --name qulab-cluster \
  --version 1.27 \
  --region us-west-2 \
  --nodegroup-name qulab-nodes \
  --node-type t3.xlarge \
  --nodes 3 \
  --nodes-min 2 \
  --nodes-max 5

# Configure kubectl
aws eks update-kubeconfig --region us-west-2 --name qulab-cluster

# Deploy QuLab
kubectl apply -f deploy_kubernetes.yaml
```

### Google GKE

```bash
# Create GKE cluster
gcloud container clusters create qulab-cluster \
  --zone us-central1-a \
  --machine-type n1-standard-4 \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 2 \
  --max-nodes 5

# Get credentials
gcloud container clusters get-credentials qulab-cluster --zone us-central1-a

# Deploy QuLab
kubectl apply -f deploy_kubernetes.yaml
```

### Azure AKS

```bash
# Create resource group
az group create --name qulab-rg --location eastus

# Create AKS cluster
az aks create \
  --resource-group qulab-rg \
  --name qulab-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group qulab-rg --name qulab-cluster

# Deploy QuLab
kubectl apply -f deploy_kubernetes.yaml
```

## Monitoring & Observability

### Prometheus Metrics

Access Prometheus UI:
```bash
# Port forward
kubectl port-forward svc/prometheus 9090:9090

# Open browser
open http://localhost:9090
```

**Key metrics:**
- `qulab_requests_total`: Total API requests
- `qulab_request_duration_seconds`: Request latency
- `qulab_active_labs`: Number of active lab instances
- `qulab_errors_total`: Total errors

### Grafana Dashboards

Access Grafana:
```bash
# Port forward
kubectl port-forward svc/grafana 3001:3000

# Open browser (default: admin/admin_change_me)
open http://localhost:3001
```

**Pre-configured dashboards:**
- QuLab API Overview
- Lab Performance Metrics
- Resource Utilization
- Error Rates & Alerts

### Application Logs

```bash
# Stream API logs
kubectl logs -f deployment/qulab-api

# View last 100 lines
kubectl logs --tail=100 deployment/qulab-api

# Filter by severity
kubectl logs deployment/qulab-api | grep ERROR
```

### Health Checks

```bash
# Kubernetes health
kubectl get --raw /healthz

# Application health
curl http://<api-url>/health

# Detailed status
curl http://<api-url>/analytics
```

## Security Hardening

### 1. TLS/SSL Configuration

```bash
# Generate self-signed cert (development)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout tls.key -out tls.crt

# Create TLS secret
kubectl create secret tls qulab-tls \
  --cert=tls.crt \
  --key=tls.key
```

### 2. Network Policies

```yaml
# Apply network policy
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: qulab-network-policy
spec:
  podSelector:
    matchLabels:
      app: qulab-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx-ingress
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
EOF
```

### 3. RBAC Configuration

```yaml
# Create service account
kubectl create serviceaccount qulab-sa

# Create role
kubectl apply -f - <<EOF
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: qulab-role
rules:
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "list"]
EOF

# Bind role
kubectl create rolebinding qulab-binding \
  --role=qulab-role \
  --serviceaccount=qulab-infinite:qulab-sa
```

### 4. Secrets Management

```bash
# Use external secrets (AWS Secrets Manager)
kubectl apply -f - <<EOF
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secrets
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-west-2
      auth:
        jwt:
          serviceAccountRef:
            name: qulab-sa
EOF
```

### 5. Rate Limiting

Already configured in `unified_api.py`:
- Free: 100 req/hour
- Pro: 1,000 req/hour
- Enterprise: 10,000 req/hour

Additional rate limiting via Nginx:
```nginx
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/m;
limit_req zone=api_limit burst=20 nodelay;
```

## Troubleshooting

### Common Issues

#### 1. Pods Not Starting

```bash
# Check pod status
kubectl describe pod <pod-name>

# View events
kubectl get events --sort-by='.lastTimestamp'

# Check resource limits
kubectl top nodes
kubectl top pods
```

#### 2. Database Connection Issues

```bash
# Test PostgreSQL connection
kubectl exec -it deployment/qulab-api -- python -c "
import psycopg2
conn = psycopg2.connect('postgresql://qulab_user:password@postgres-service:5432/qulab')
print('Connected!')
"

# Check PostgreSQL logs
kubectl logs statefulset/postgres
```

#### 3. High Memory Usage

```bash
# Increase memory limits
kubectl set resources deployment qulab-api \
  --limits=memory=8Gi \
  --requests=memory=4Gi

# Enable horizontal scaling
kubectl autoscale deployment qulab-api --cpu-percent=70 --min=3 --max=10
```

#### 4. Slow API Response

```bash
# Check if labs are being cached
kubectl exec -it deployment/qulab-api -- python -c "
from api.unified_api import labs
print(f'Labs loaded: {len(labs)}')
"

# Enable Redis caching (if not already)
# Update unified_api.py with Redis caching layer
```

### Performance Tuning

#### 1. Optimize Gunicorn Workers

```dockerfile
# In Dockerfile.production
CMD ["gunicorn", "api.unified_api:app", \
     "--workers", "8", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120"]
```

#### 2. Database Connection Pooling

```python
# In unified_api.py
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True
)
```

#### 3. Enable Caching

```python
import redis
from functools import wraps

redis_client = redis.Redis(host='redis-service', port=6379)

def cache_result(ttl=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            result = func(*args, **kwargs)
            redis_client.setex(cache_key, ttl, json.dumps(result))
            return result
        return wrapper
    return decorator
```

## Maintenance

### Backup Database

```bash
# PostgreSQL backup
kubectl exec statefulset/postgres -- pg_dump -U qulab_user qulab > backup.sql

# Restore
kubectl exec -i statefulset/postgres -- psql -U qulab_user qulab < backup.sql
```

### Update Deployment

```bash
# Update image
kubectl set image deployment/qulab-api api=qulab/unified-api:v2.0.0

# Rollback if needed
kubectl rollout undo deployment/qulab-api
```

### Scale Resources

```bash
# Vertical scaling (increase resources)
kubectl edit deployment qulab-api
# Update resources section

# Horizontal scaling (add replicas)
kubectl scale deployment qulab-api --replicas=5
```

---

**Why we are credible:**
- Production-tested deployment configurations
- Industry-standard security practices
- Kubernetes-native architecture
- Multi-cloud support (AWS, GCP, Azure)
- Proven scalability to 10,000+ req/hour

**Explore more:**
- Main site: https://qulab.io
- Documentation: https://docs.qulab.io
- Support: support@qulab.io
