# QuLab AI Kubernetes Deployment

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

**Phase 3: Production Deployment with Kubernetes**

---

## Overview

This directory contains Kubernetes manifests for deploying QuLab AI in a production environment with:
- High availability (3+ replicas)
- Auto-scaling (HPA)
- Load balancing (Ingress)
- TLS/SSL (cert-manager)
- Resource management
- Health monitoring
- Persistent storage
- Automated backups

---

## Prerequisites

1. **Kubernetes Cluster** (v1.25+)
   - AWS EKS, Google GKE, Azure AKS, or self-hosted
   - Minimum 3 nodes, 4 vCPU, 8GB RAM each

2. **kubectl** (v1.25+)
   ```bash
   kubectl version --client
   ```

3. **Ingress Controller** (NGINX)
   ```bash
   kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml
   ```

4. **cert-manager** (for TLS)
   ```bash
   kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
   ```

5. **Metrics Server** (for HPA)
   ```bash
   kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
   ```

6. **External Secrets Operator** (optional, for AWS/GCP/Azure secrets)
   ```bash
   helm repo add external-secrets https://charts.external-secrets.io
   helm install external-secrets external-secrets/external-secrets -n external-secrets-system --create-namespace
   ```

---

## Quick Start

### 1. Create Namespace and Secrets

```bash
# Create namespace
kubectl apply -f configmap.yaml  # This includes namespace definition

# Generate JWT secret
export JWT_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
export API_SALT=$(python -c "import secrets; print(secrets.token_urlsafe(32))")

# Create secrets
kubectl create secret generic qulab-secrets \
  --namespace=qulab \
  --from-literal=jwt-secret-key="$JWT_SECRET" \
  --from-literal=api-key-salt="$API_SALT"
```

### 2. Build and Push Docker Image

```bash
# Build image
cd /Users/noone/QuLabInfinite
docker build -t qulab-ai:latest .

# Tag for registry (replace with your registry)
docker tag qulab-ai:latest ghcr.io/YOUR_ORG/qulab-ai:latest

# Push to registry
docker push ghcr.io/YOUR_ORG/qulab-ai:latest

# Update deployment.yaml with your image path
```

### 3. Deploy Application

```bash
# Apply manifests in order
kubectl apply -f configmap.yaml
kubectl apply -f storage.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml
kubectl apply -f autoscaling.yaml

# Verify deployment
kubectl get all -n qulab
```

### 4. Configure DNS

```bash
# Get Load Balancer IP/hostname
kubectl get ingress qulab-api-ingress -n qulab

# Add DNS A/CNAME records:
# api.qulab.ai -> <LOAD_BALANCER_IP>
# qulab.ai -> <LOAD_BALANCER_IP>
```

### 5. Verify TLS Certificate

```bash
# Check certificate status
kubectl get certificate -n qulab
kubectl describe certificate qulab-tls-cert -n qulab

# Should show "Ready: True"
```

---

## Deployment Commands

### Check Status

```bash
# All resources
kubectl get all -n qulab

# Pods
kubectl get pods -n qulab -o wide

# Services
kubectl get svc -n qulab

# Ingress
kubectl get ingress -n qulab

# HPA status
kubectl get hpa -n qulab

# PVC status
kubectl get pvc -n qulab
```

### View Logs

```bash
# All pods
kubectl logs -n qulab -l app=qulab-api --tail=100 -f

# Specific pod
kubectl logs -n qulab <POD_NAME> -f

# Previous container (if crashed)
kubectl logs -n qulab <POD_NAME> --previous
```

### Debug Pod

```bash
# Exec into pod
kubectl exec -it -n qulab <POD_NAME> -- /bin/bash

# Port forward (local testing)
kubectl port-forward -n qulab svc/qulab-api-internal 8000:8000

# Test locally
curl http://localhost:8000/health
```

### Scale Manually

```bash
# Scale to 5 replicas
kubectl scale deployment qulab-api -n qulab --replicas=5

# Check scaling
kubectl get deployment qulab-api -n qulab
```

### Update Application

```bash
# Update image
kubectl set image deployment/qulab-api -n qulab qulab-api=ghcr.io/YOUR_ORG/qulab-ai:v2.1.0

# Check rollout status
kubectl rollout status deployment/qulab-api -n qulab

# Rollback if needed
kubectl rollout undo deployment/qulab-api -n qulab
```

---

## Configuration

### Environment Variables

Edit `configmap.yaml` to change:
- Log level
- Rate limits
- CORS settings
- Worker count
- Timeouts

Apply changes:
```bash
kubectl apply -f configmap.yaml
kubectl rollout restart deployment/qulab-api -n qulab
```

### Secrets

Update secrets:
```bash
kubectl create secret generic qulab-secrets \
  --namespace=qulab \
  --from-literal=jwt-secret-key="NEW_SECRET" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl rollout restart deployment/qulab-api -n qulab
```

### Resource Limits

Edit `deployment.yaml` resources section:
```yaml
resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 2000m
    memory: 4Gi
```

### Auto-scaling

Edit `autoscaling.yaml`:
- `minReplicas`: Minimum pod count
- `maxReplicas`: Maximum pod count
- `averageUtilization`: CPU/memory thresholds

---

## Monitoring

### Health Checks

```bash
# Health endpoint
curl https://api.qulab.ai/health

# Metrics endpoint
curl https://api.qulab.ai/metrics
```

### Kubernetes Events

```bash
# Recent events
kubectl get events -n qulab --sort-by='.lastTimestamp'

# Watch events
kubectl get events -n qulab --watch
```

### Resource Usage

```bash
# Top pods (CPU/memory)
kubectl top pods -n qulab

# Top nodes
kubectl top nodes
```

### HPA Metrics

```bash
# HPA status
kubectl get hpa qulab-api-hpa -n qulab

# Detailed metrics
kubectl describe hpa qulab-api-hpa -n qulab
```

---

## Backup and Restore

### Manual Backup

```bash
# Create snapshot
kubectl apply -f - <<EOF
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: qulab-data-manual-backup
  namespace: qulab
spec:
  volumeSnapshotClassName: csi-snapclass
  source:
    persistentVolumeClaimName: qulab-data-pvc
EOF

# Verify snapshot
kubectl get volumesnapshot -n qulab
```

### Restore from Backup

```bash
# Create PVC from snapshot
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: qulab-data-restored
  namespace: qulab
spec:
  storageClassName: fast-ssd
  dataSource:
    name: qulab-data-manual-backup
    kind: VolumeSnapshot
    apiGroup: snapshot.storage.k8s.io
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 50Gi
EOF

# Update deployment to use restored PVC
# Edit deployment.yaml, change claimName to qulab-data-restored
kubectl apply -f deployment.yaml
```

### Automated Backups

Backups run daily at 2 AM via CronJob:
```bash
# Check backup jobs
kubectl get cronjob -n qulab
kubectl get jobs -n qulab

# View backup logs
kubectl logs -n qulab job/qulab-data-backup-<timestamp>
```

---

## Troubleshooting

### Pods Not Starting

```bash
# Check pod status
kubectl describe pod <POD_NAME> -n qulab

# Common issues:
# - Image pull errors: Check registry credentials
# - Resource constraints: Check node resources
# - Config errors: Check configmap/secrets
```

### High Latency

```bash
# Check HPA metrics
kubectl get hpa -n qulab

# Scale up manually if needed
kubectl scale deployment qulab-api -n qulab --replicas=10

# Check resource usage
kubectl top pods -n qulab
```

### Certificate Issues

```bash
# Check cert status
kubectl describe certificate qulab-tls-cert -n qulab

# Check cert-manager logs
kubectl logs -n cert-manager -l app=cert-manager

# Recreate certificate
kubectl delete certificate qulab-tls-cert -n qulab
kubectl apply -f ingress.yaml
```

### Ingress Not Working

```bash
# Check ingress
kubectl describe ingress qulab-api-ingress -n qulab

# Check NGINX controller
kubectl get pods -n ingress-nginx
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx

# Test backend directly
kubectl port-forward -n qulab svc/qulab-api-internal 8000:8000
curl http://localhost:8000/health
```

---

## Performance Tuning

### Horizontal Scaling

Increase replicas for higher throughput:
```bash
# Set min/max replicas
kubectl patch hpa qulab-api-hpa -n qulab -p '{"spec":{"minReplicas":5,"maxReplicas":50}}'
```

### Vertical Scaling

Increase resources per pod:
```bash
# Edit deployment
kubectl edit deployment qulab-api -n qulab

# Update resources section
resources:
  requests:
    cpu: 1000m
    memory: 2Gi
  limits:
    cpu: 4000m
    memory: 8Gi
```

### Load Testing

```bash
# Port forward
kubectl port-forward -n qulab svc/qulab-api-internal 8000:8000

# Run performance tests
cd /Users/noone/QuLabInfinite
python tests/test_performance.py
```

---

## Security

### Network Policies

```bash
# Apply network policies (create separate file)
kubectl apply -f network-policy.yaml
```

### Pod Security Standards

```bash
# Enable pod security
kubectl label namespace qulab pod-security.kubernetes.io/enforce=restricted
```

### RBAC

Service account has minimal permissions (get configmaps/secrets/pods only).
Review `deployment.yaml` RBAC section.

---

## Production Checklist

- [ ] Docker image built and pushed to registry
- [ ] Secrets created with strong random values
- [ ] DNS records configured (api.qulab.ai, qulab.ai)
- [ ] TLS certificate provisioned and verified
- [ ] HPA configured and metrics-server running
- [ ] PVC provisioned with sufficient storage
- [ ] Resource requests/limits tuned for workload
- [ ] Backup CronJob enabled and tested
- [ ] Monitoring/alerting configured (Prometheus/Grafana)
- [ ] Load testing completed
- [ ] Rollback procedure tested
- [ ] On-call team briefed

---

## Support

For issues or questions:
- Email: joshua@corporationoflight.com
- GitHub: https://github.com/YOUR_ORG/QuLabInfinite

---

**Generated:** October 30, 2025
**Version:** 2.0.0
**Status:** Production Ready
