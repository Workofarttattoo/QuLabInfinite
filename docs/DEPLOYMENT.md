# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

# QuLab Master API - Deployment Instructions

## Quick Start (5 Commands)

```bash
# 1. Build and start all services
docker-compose up -d --build

# 2. Verify health
curl http://localhost:8000/health

# 3. List all labs
curl -H "Authorization: Bearer qulab_demo_key" http://localhost:8000/labs

# 4. Access API docs
open http://localhost:8000/docs

# 5. Monitor with Grafana
open http://localhost:3000  # admin/qulab2025
```

## Architecture

- **Master API** (Port 8000): Unified gateway aggregating 20+ quantum labs
- **Redis** (Port 6379): Rate limiting and caching
- **Prometheus** (Port 9090): Metrics collection
- **Grafana** (Port 3000): Visualization dashboards
- **Nginx** (Port 80/443): Reverse proxy with rate limiting

## API Keys

- **Demo**: `qulab_demo_key` (60 req/min)
- **Enterprise**: `qulab_master_key_2025` (1000 req/min)

## Available Labs (20+)

1. quantum - Quantum computing/simulation
2. materials - Materials science discovery
3. chemistry - Chemical synthesis
4. frequency - EM frequency analysis
5. oncology - Cancer treatment optimization
6. protein_folding - Protein structure prediction
7. cardiovascular - Cardiovascular disease modeling
8. tumor_evolution - Tumor dynamics simulation
9. genetic_variants - Genetic variant analysis
10. cancer_metabolic - Cancer metabolic pathways
11. drug_interaction - Drug-drug interactions
12. immune_response - Immune system simulation
13. neurotransmitter - Neurotransmitter optimization
14. microbiome - Gut microbiome optimization
15. metabolic_syndrome - Metabolic syndrome reversal
16. stem_cell - Stem cell differentiation
17. medical_safety - Toxicity assessment

## Usage Examples

### List Labs
```bash
curl -H "Authorization: Bearer qulab_demo_key" \
  http://localhost:8000/labs
```

### Optimize with Quantum Lab
```bash
curl -X POST \
  -H "Authorization: Bearer qulab_demo_key" \
  -H "Content-Type: application/json" \
  -d '{
    "lab_name": "quantum",
    "parameters": {
      "algorithm": "vqe",
      "qubits": 8
    }
  }' \
  http://localhost:8000/labs/quantum/optimize
```

### Check Metrics
```bash
curl -H "Authorization: Bearer qulab_demo_key" \
  http://localhost:8000/metrics
```

## Production Deployment

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- 8GB+ RAM
- 4+ CPU cores

### Environment Variables
```bash
export LOG_LEVEL=info
export WORKERS=4
```

### SSL Configuration
1. Place SSL certificates in `nginx/ssl/`
2. Uncomment HTTPS block in `nginx/nginx.conf`
3. Update domain name
4. Restart nginx: `docker-compose restart nginx`

## Monitoring

- **Health**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs

## Scaling

### Horizontal Scaling
```bash
docker-compose up -d --scale qulab-master-api=3
```

### Resource Limits
Edit `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      cpus: '8.0'
      memory: 16G
```

## Troubleshooting

### Check logs
```bash
docker-compose logs -f qulab-master-api
```

### Restart services
```bash
docker-compose restart
```

### Reset everything
```bash
docker-compose down -v
docker-compose up -d --build
```

## Security

- API keys validated on every request
- Rate limiting per tier (demo: 60/min, enterprise: 1000/min)
- CORS enabled with configurable origins
- Security headers via Nginx
- Health endpoint excluded from rate limits

## License
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
