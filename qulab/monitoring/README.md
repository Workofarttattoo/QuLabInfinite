# QuLab AI Monitoring Stack

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

**Phase 4: Production Monitoring with Prometheus & Grafana**

---

## Overview

This directory contains monitoring configuration for QuLab AI production deployment:
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization dashboards
- **Alert Rules**: Proactive incident detection
- **Exporters**: System and application metrics

---

## Quick Start

### 1. Install Prometheus and Grafana

Using Helm (recommended):

```bash
# Add Helm repos
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Create monitoring namespace
kubectl create namespace monitoring

# Install Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
  -f prometheus-values.yaml

# Install Grafana (if not included in kube-prometheus-stack)
helm install grafana grafana/grafana \
  --namespace monitoring \
  --set adminPassword='CHANGE_ME' \
  -f grafana-values.yaml
```

### 2. Apply Custom Configuration

```bash
# Create ConfigMap from prometheus-config.yaml
kubectl create configmap prometheus-config \
  --from-file=prometheus.yml=prometheus-config.yaml \
  --namespace monitoring

# Apply alert rules
kubectl create configmap prometheus-alerts \
  --from-file=alerts/qulab-alerts.yaml \
  --namespace monitoring

# Reload Prometheus configuration
kubectl rollout restart statefulset prometheus-prometheus-kube-prometheus-prometheus -n monitoring
```

### 3. Import Grafana Dashboards

```bash
# Get Grafana admin password
kubectl get secret --namespace monitoring grafana -o jsonpath="{.data.admin-password}" | base64 --decode ; echo

# Port forward to Grafana
kubectl port-forward -n monitoring svc/grafana 3000:80

# Open browser to http://localhost:3000
# Login: admin / <password from above>

# Import dashboard:
# 1. Go to Dashboards → Import
# 2. Upload grafana/qulab-api-dashboard.json
# 3. Select Prometheus datasource
# 4. Click Import
```

---

## Accessing Monitoring Services

### Prometheus

```bash
# Port forward
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090

# Open http://localhost:9090
```

**Key URLs:**
- Graph: http://localhost:9090/graph
- Alerts: http://localhost:9090/alerts
- Targets: http://localhost:9090/targets
- Configuration: http://localhost:9090/config

### Grafana

```bash
# Port forward
kubectl port-forward -n monitoring svc/grafana 3000:80

# Open http://localhost:3000
```

**Pre-configured Dashboards:**
- QuLab API Production Dashboard: Main API metrics
- Kubernetes Cluster Dashboard: Cluster health
- Node Exporter Dashboard: System metrics

### Alertmanager

```bash
# Port forward
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-alertmanager 9093:9093

# Open http://localhost:9093
```

---

## Metrics Available

### Application Metrics (QuLab API)

From `/metrics` endpoint:

```prometheus
# Request metrics
http_requests_total{method, endpoint, status}
http_request_duration_seconds_bucket{method, endpoint, le}
http_request_size_bytes_bucket{method, endpoint, le}
http_response_size_bytes_bucket{method, endpoint, le}

# Error metrics
http_errors_total{error_type}
authentication_failures_total{reason}
rate_limit_exceeded_total

# Resource metrics
process_cpu_seconds_total
process_resident_memory_bytes
process_open_fds

# Custom business metrics
molecule_parses_total
spectrum_encodings_total
api_key_validations_total
```

### Kubernetes Metrics

From kube-state-metrics:

```prometheus
# Pod metrics
kube_pod_status_phase{pod, namespace, phase}
kube_pod_container_status_restarts_total{pod, namespace, container}
kube_pod_container_resource_requests{pod, namespace, resource}
kube_pod_container_resource_limits{pod, namespace, resource}

# Deployment metrics
kube_deployment_status_replicas{deployment, namespace}
kube_deployment_status_replicas_available{deployment, namespace}

# HPA metrics
kube_horizontalpodautoscaler_status_current_replicas{horizontalpodautoscaler}
kube_horizontalpodautoscaler_spec_max_replicas{horizontalpodautoscaler}
```

### Node Metrics

From node-exporter:

```prometheus
# CPU
node_cpu_seconds_total{cpu, mode}

# Memory
node_memory_MemTotal_bytes
node_memory_MemAvailable_bytes
node_memory_Cached_bytes

# Disk
node_filesystem_size_bytes{device, mountpoint}
node_filesystem_avail_bytes{device, mountpoint}

# Network
node_network_receive_bytes_total{device}
node_network_transmit_bytes_total{device}
```

---

## Alert Rules

### Critical Alerts

**Immediate Action Required:**

1. **APIDown** - API is completely unavailable
2. **HighErrorRate** - Error rate > 5% for 5+ minutes
3. **DatabaseConnectionPoolExhausted** - DB connections exhausted

### Warning Alerts

**Investigate Soon:**

1. **HighLatency** - p99 latency > 500ms
2. **HighMemoryUsage** - Memory > 85% for 5+ minutes
3. **HighCPUUsage** - CPU > 85% for 5+ minutes
4. **PodRestartLoop** - Pods restarting frequently
5. **LowDiskSpace** - Disk < 15% remaining

### Info Alerts

**Nice to Know:**

1. **LowThroughput** - Request rate < 10 req/s
2. **FrequentRateLimiting** - Rate limiting triggered often
3. **HPAMaxCapacity** - Autoscaler at maximum replicas

---

## Configuring Alertmanager

### Slack Notifications

Edit Alertmanager configuration:

```yaml
global:
  resolve_timeout: 5m
  slack_api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'slack-notifications'

receivers:
- name: 'slack-notifications'
  slack_configs:
  - channel: '#qulab-alerts'
    title: '{{ .GroupLabels.alertname }}'
    text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
```

Apply configuration:

```bash
kubectl create secret generic alertmanager-prometheus-kube-prometheus-alertmanager \
  --from-file=alertmanager.yaml=alertmanager-config.yaml \
  --namespace monitoring \
  --dry-run=client -o yaml | kubectl apply -f -
```

### PagerDuty Integration

```yaml
receivers:
- name: 'pagerduty'
  pagerduty_configs:
  - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
    description: '{{ .GroupLabels.alertname }}: {{ .CommonAnnotations.summary }}'
```

### Email Notifications

```yaml
receivers:
- name: 'email'
  email_configs:
  - to: 'ops@qulab.ai'
    from: 'alertmanager@qulab.ai'
    smarthost: 'smtp.gmail.com:587'
    auth_username: 'alertmanager@qulab.ai'
    auth_password: 'YOUR_APP_PASSWORD'
```

---

## Custom Dashboards

### Creating Custom Dashboards

1. Open Grafana UI
2. Click "+" → "Dashboard"
3. Add Panel
4. Select Prometheus datasource
5. Enter PromQL query
6. Configure visualization
7. Save dashboard

### Useful PromQL Queries

**Request rate by endpoint:**
```promql
sum(rate(http_requests_total{job="qulab-api"}[5m])) by (endpoint)
```

**Error rate percentage:**
```promql
100 * sum(rate(http_requests_total{job="qulab-api",status=~"5.."}[5m])) / sum(rate(http_requests_total{job="qulab-api"}[5m]))
```

**Top endpoints by latency:**
```promql
topk(10, histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{job="qulab-api"}[5m])) by (endpoint, le)))
```

**Pod restarts in last 24h:**
```promql
changes(kube_pod_container_status_restarts_total{pod=~"qulab-api-.*"}[24h])
```

---

## Monitoring Best Practices

### 1. Alert Fatigue Prevention

- Set appropriate thresholds
- Use `for:` clauses to avoid flapping
- Group related alerts
- Use severity levels correctly

### 2. Dashboard Organization

- One dashboard per service/component
- Use template variables for filtering
- Include SLO/SLI metrics
- Add links between related dashboards

### 3. Retention Policy

Configure Prometheus retention:

```yaml
prometheus:
  prometheusSpec:
    retention: 30d
    retentionSize: "50GB"
```

For long-term storage, use Thanos or Cortex.

### 4. High Availability

Deploy Prometheus in HA mode:

```yaml
prometheus:
  prometheusSpec:
    replicas: 2
```

---

## Troubleshooting

### Prometheus Not Scraping Metrics

Check targets:
```bash
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090
# Visit http://localhost:9090/targets
```

Check pod annotations:
```bash
kubectl get pods -n qulab -o yaml | grep -A5 annotations
```

Should include:
```yaml
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "8000"
  prometheus.io/path: "/metrics"
```

### Grafana Dashboard Shows "No Data"

1. Check datasource configuration
2. Verify PromQL query syntax
3. Check time range
4. Verify metrics exist in Prometheus

Test query in Prometheus first:
```bash
curl -G 'http://localhost:9090/api/v1/query' \
  --data-urlencode 'query=up{job="qulab-api"}'
```

### Alerts Not Firing

Check Prometheus rules:
```bash
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090
# Visit http://localhost:9090/rules
```

Check Alertmanager:
```bash
kubectl logs -n monitoring alertmanager-prometheus-kube-prometheus-alertmanager-0
```

---

## Performance Tuning

### Reduce Cardinality

Avoid high-cardinality labels:
- ❌ Bad: `user_id="12345"` (millions of unique values)
- ✅ Good: `endpoint="/api/v2/parse/molecule"` (tens of unique values)

### Optimize Query Performance

Use recording rules for expensive queries:

```yaml
groups:
- name: qulab_recording_rules
  interval: 30s
  rules:
  - record: job:http_requests_total:rate5m
    expr: sum(rate(http_requests_total{job="qulab-api"}[5m]))
```

---

## Maintenance

### Backup Prometheus Data

```bash
# Stop Prometheus
kubectl scale statefulset prometheus-prometheus-kube-prometheus-prometheus -n monitoring --replicas=0

# Backup data directory
kubectl exec -n monitoring prometheus-prometheus-kube-prometheus-prometheus-0 -- tar czf /tmp/backup.tar.gz /prometheus

# Copy backup
kubectl cp monitoring/prometheus-prometheus-kube-prometheus-prometheus-0:/tmp/backup.tar.gz ./prometheus-backup.tar.gz

# Restart Prometheus
kubectl scale statefulset prometheus-prometheus-kube-prometheus-prometheus -n monitoring --replicas=2
```

### Upgrade Monitoring Stack

```bash
# Update Helm repos
helm repo update

# Upgrade Prometheus
helm upgrade prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --reuse-values

# Upgrade Grafana
helm upgrade grafana grafana/grafana \
  --namespace monitoring \
  --reuse-values
```

---

## Support

For monitoring-related issues:
- Email: joshua@corporationoflight.com
- Slack: #qulab-monitoring
- Grafana Docs: https://grafana.com/docs/
- Prometheus Docs: https://prometheus.io/docs/

---

**Generated:** October 30, 2025
**Version:** 2.0.0
**Status:** Production Ready
