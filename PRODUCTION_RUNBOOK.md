# QuLab AI Production Runbook

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

**Version:** 1.0.0
**Last Updated:** October 30, 2025
**On-Call Contact:** joshua@corporationoflight.com

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Deployment](#deployment)
4. [Monitoring](#monitoring)
5. [Common Issues](#common-issues)
6. [Emergency Procedures](#emergency-procedures)
7. [Maintenance](#maintenance)

---

## Quick Start

### Prerequisites
- Docker 20.10+
- docker-compose 1.29+
- 4GB RAM minimum
- 10GB disk space

### Deploy to Production

```bash
cd /Users/noone/QuLabInfinite
./scripts/deploy_production.sh
```

### Check System Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-30T12:00:00.000Z",
  "version": "1.0.0",
  "system": {
    "cpu_percent": 25.4,
    "memory_percent": 45.2,
    "disk_percent": 60.1
  },
  "dependencies": {
    "pint": true,
    "rdkit": true,
    "logging": true
  }
}
```

---

## Architecture Overview

### Components

```
┌─────────────────────────────────────────────────┐
│              Load Balancer (Optional)           │
└────────────────────┬────────────────────────────┘
                     │
         ┌───────────┴──────────┐
         │                      │
    ┌────▼─────┐          ┌────▼─────┐
    │ QuLab API│          │ QuLab API│
    │ Instance │          │ Instance │
    │   :8000  │          │   :8000  │
    └────┬─────┘          └────┬─────┘
         │                      │
         └───────────┬──────────┘
                     │
         ┌───────────▼──────────┐
         │   Shared Components   │
         ├───────────────────────┤
         │ • Logging (JSON)      │
         │ • Metrics (Prometheus)│
         │ • Health Checks       │
         │ • Error Handling      │
         └───────────────────────┘
```

### Key Files

| File | Purpose |
|------|---------|
| `api/production_api.py` | Main API server |
| `qulab_ai/production/logging_config.py` | Logging configuration |
| `qulab_ai/production/error_handling.py` | Error handling utilities |
| `Dockerfile` | Container image definition |
| `docker-compose.yml` | Multi-container orchestration |
| `scripts/deploy_production.sh` | Deployment automation |

---

## Deployment

### Standard Deployment

```bash
# 1. Pull latest code
git pull origin main

# 2. Run deployment script
./scripts/deploy_production.sh

# 3. Verify deployment
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

### Rolling Update (Zero Downtime)

```bash
# 1. Build new image
docker build -t qulab-ai:v1.1.0 .

# 2. Start new container on different port
docker run -d --name qulab-api-new -p 8001:8000 qulab-ai:v1.1.0

# 3. Verify new container is healthy
curl http://localhost:8001/health

# 4. Update load balancer to point to :8001

# 5. Wait for connections to drain from old container

# 6. Stop old container
docker stop qulab-api

# 7. Remove old container
docker rm qulab-api

# 8. Rename new container
docker rename qulab-api-new qulab-api
```

### Rollback Procedure

```bash
# 1. Stop current version
docker-compose down

# 2. Checkout previous version
git checkout HEAD~1

# 3. Redeploy
./scripts/deploy_production.sh

# 4. Verify
curl http://localhost:8000/health
```

---

## Monitoring

### Health Endpoints

**Primary Health Check:**
```bash
curl http://localhost:8000/health
```

**Detailed Metrics:**
```bash
curl http://localhost:8000/metrics
```

### Key Metrics

| Metric | Normal Range | Warning | Critical |
|--------|-------------|---------|----------|
| CPU Usage | < 70% | 70-85% | > 85% |
| Memory Usage | < 75% | 75-90% | > 90% |
| Disk Usage | < 80% | 80-90% | > 90% |
| Response Time (p50) | < 100ms | 100-500ms | > 500ms |
| Response Time (p99) | < 500ms | 500ms-2s | > 2s |
| Error Rate | < 0.1% | 0.1-1% | > 1% |

### Viewing Logs

**Real-time logs:**
```bash
docker logs -f qulab-api
```

**Structured JSON logs:**
```bash
tail -f /Users/noone/QuLabInfinite/logs/qulab_ai.log | python -m json.tool
```

**Filter for errors:**
```bash
cat logs/qulab_ai.log | jq 'select(.level=="ERROR")'
```

### Grafana Dashboards

Access: http://localhost:3000 (admin/admin)

**Key Dashboards:**
1. **API Overview** - Request rate, error rate, latency
2. **System Resources** - CPU, memory, disk usage
3. **Application Metrics** - Parser success rates, processing times

---

## Common Issues

### Issue: API Not Responding

**Symptoms:**
- Health check fails
- Timeouts on API requests

**Diagnosis:**
```bash
# Check if container is running
docker ps | grep qulab-api

# Check container logs
docker logs --tail 100 qulab-api

# Check system resources
docker stats qulab-api
```

**Resolution:**
```bash
# Restart container
docker restart qulab-api

# If that fails, redeploy
docker-compose down
./scripts/deploy_production.sh
```

### Issue: High Memory Usage

**Symptoms:**
- Memory usage > 90%
- OOM kills in logs

**Diagnosis:**
```bash
# Check memory usage
docker stats qulab-api

# Check for memory leaks in logs
cat logs/qulab_ai.log | jq 'select(.message | contains("memory"))'
```

**Resolution:**
```bash
# Increase container memory limit in docker-compose.yml
# Change: memory: 4G -> memory: 8G

# Redeploy
docker-compose down
docker-compose up -d
```

### Issue: Slow Response Times

**Symptoms:**
- Response time > 500ms
- High p99 latency

**Diagnosis:**
```bash
# Check metrics
curl http://localhost:8000/metrics

# Check slow operations in logs
cat logs/qulab_ai.log | jq 'select(.duration_ms > 100)'
```

**Resolution:**
1. Check if circuit breakers are open
2. Verify external dependencies are healthy
3. Consider adding caching
4. Scale horizontally (add more instances)

### Issue: Parser Failures

**Symptoms:**
- `PARSER_ERROR` in logs
- Failed molecule/structure/spectrum parsing

**Diagnosis:**
```bash
# Check error details
cat logs/qulab_ai.log | jq 'select(.error_code=="PARSER_ERROR")'
```

**Resolution:**
1. Verify input format is correct
2. Check if dependency (RDKit/PyMatGen) is available
3. Review parser error details in logs
4. Apply fallback parser if available

---

## Emergency Procedures

### System Down - All Services Unresponsive

**Priority:** P0 - Critical
**Response Time:** Immediate

**Steps:**
1. Check if Docker daemon is running: `systemctl status docker`
2. Check system resources: `top`, `df -h`
3. Restart Docker: `systemctl restart docker`
4. Redeploy: `./scripts/deploy_production.sh`
5. Notify stakeholders if downtime > 5 minutes

### Data Corruption

**Priority:** P1 - High
**Response Time:** < 15 minutes

**Steps:**
1. Stop all write operations
2. Restore from last known good backup
3. Verify data integrity
4. Resume operations
5. Post-mortem analysis

### Security Breach

**Priority:** P0 - Critical
**Response Time:** Immediate

**Steps:**
1. Isolate affected systems: `docker network disconnect`
2. Capture forensic evidence: `docker logs > incident.log`
3. Rotate all credentials
4. Patch vulnerability
5. Conduct security audit
6. Document incident

---

## Maintenance

### Daily Checks

```bash
# Health check
curl http://localhost:8000/health

# Check disk space
df -h /Users/noone/QuLabInfinite

# Review error logs
cat logs/qulab_ai.log | jq 'select(.level=="ERROR")' | tail -20
```

### Weekly Tasks

1. **Log Rotation:** Verify logs are rotating properly
   ```bash
   ls -lh logs/
   ```

2. **Backup Verification:** Test backup restoration
   ```bash
   # Restore test backup
   docker-compose down
   # ... restore data ...
   ./scripts/deploy_production.sh
   ```

3. **Dependency Updates:** Check for security patches
   ```bash
   pip list --outdated
   ```

### Monthly Tasks

1. **Performance Review:** Analyze metrics trends
2. **Capacity Planning:** Forecast resource needs
3. **Security Audit:** Review access logs
4. **Documentation Update:** Keep runbook current

### Quarterly Tasks

1. **Disaster Recovery Drill:** Test full system recovery
2. **Security Penetration Test:** External security audit
3. **Architecture Review:** Evaluate scalability needs
4. **Training:** Update team on new features/procedures

---

## Contact Information

### On-Call Rotation

| Role | Contact | Hours |
|------|---------|-------|
| Primary On-Call | joshua@corporationoflight.com | 24/7 |
| Backup On-Call | TBD | 24/7 |
| Engineering Lead | TBD | Business hours |

### Escalation Path

1. **Level 1:** On-call engineer (immediate)
2. **Level 2:** Engineering lead (< 1 hour)
3. **Level 3:** CTO (< 4 hours)
4. **Level 4:** CEO (critical incidents only)

---

## Appendix

### Useful Commands

```bash
# View all containers
docker ps -a

# View container resource usage
docker stats

# Execute command in container
docker exec -it qulab-api bash

# View container environment variables
docker exec qulab-api env

# Backup logs
tar -czf logs-backup-$(date +%Y%m%d).tar.gz logs/

# Test API endpoint
curl -X POST http://localhost:8000/api/v1/parse/molecule \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CCO"}'
```

### Configuration Files

**Prometheus Config:** `monitoring/prometheus.yml`
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'qulab-api'
    static_configs:
      - targets: ['qulab-api:8000']
```

**Grafana Datasource:** Add Prometheus at http://prometheus:9090

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-10-30 | Joshua Hendricks Cole | Initial production runbook |

---

**End of Runbook**

*For questions or updates, contact: joshua@corporationoflight.com*
