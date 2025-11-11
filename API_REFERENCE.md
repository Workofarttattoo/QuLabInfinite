# QuLabInfinite API Reference

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

## Overview

The QuLabInfinite Unified API provides programmatic access to 20+ world-class scientific simulation laboratories through a single, consistent interface.

## Base URL

```
Production: https://api.qulab.io
Staging: https://staging-api.qulab.io
Local: http://localhost:8000
```

## Authentication

All API requests require authentication via API key in the request header:

```bash
curl -H "X-API-Key: your_api_key_here" https://api.qulab.io/labs
```

### API Key Tiers

| Tier | Rate Limit | Features | Price |
|------|-----------|----------|-------|
| Free | 100 req/hour | Basic access, single lab | $0/month |
| Pro | 1,000 req/hour | All labs, batch processing | $99/month |
| Enterprise | 10,000 req/hour | Priority support, custom deployment | $999/month |

## Core Endpoints

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-03T21:00:00Z",
  "labs_loaded": 8,
  "uptime": 123456.78
}
```

### List Labs

```http
GET /labs
```

Returns all available labs with descriptions and endpoints.

**Response:**
```json
{
  "materials": {
    "name": "Materials Science Lab",
    "description": "6.6M materials database with quantum-accurate property prediction",
    "endpoints": ["/materials/analyze", "/materials/search", "/materials/compare"]
  },
  ...
}
```

## Lab Endpoints

### 1. Materials Science Lab

#### Analyze Material

```http
POST /materials/analyze
```

**Request Body:**
```json
{
  "material_name": "Steel_304",
  "temperature": 300.0,
  "pressure": 1.0,
  "properties": ["strength", "conductivity"]
}
```

**Response:**
```json
{
  "material": "Steel_304",
  "conditions": {
    "temperature": 300.0,
    "pressure": 1.0
  },
  "properties": {
    "tensile_strength": 505.3,
    "yield_strength": 215.8,
    "electrical_conductivity": 1.45e6,
    "thermal_conductivity": 16.2
  },
  "confidence": 0.95,
  "computation_time": 0.05
}
```

#### Search Materials

```http
GET /materials/search?query=steel&limit=10
```

### 2. Quantum Computing Lab

#### Run Quantum Simulation

```http
POST /quantum/simulate
```

**Request Body:**
```json
{
  "system_type": "molecule",
  "num_qubits": 4,
  "circuit_depth": 10,
  "algorithm": "vqe"
}
```

**Response:**
```json
{
  "system_type": "molecule",
  "num_qubits": 4,
  "circuit_depth": 10,
  "algorithm": "vqe",
  "energy": -1.137,
  "iterations": 100,
  "converged": true,
  "fidelity": 0.999,
  "computation_time": 0.4
}
```

### 3. Chemistry Lab

#### Optimize Synthesis

```http
POST /chemistry/synthesize
```

**Request Body:**
```json
{
  "reaction_type": "synthesis",
  "reactants": ["SMILES1", "SMILES2"],
  "target_product": "TARGET_SMILES",
  "conditions": {
    "temperature": 80,
    "pressure": 1.0
  }
}
```

**Response:**
```json
{
  "reaction_type": "synthesis",
  "reactants": ["SMILES1", "SMILES2"],
  "products": ["PRODUCT_1"],
  "yield": 0.875,
  "reaction_time": 4.5,
  "optimal_conditions": {
    "temperature": 80,
    "pressure": 1.0,
    "catalyst": "Pd/C"
  },
  "confidence": 0.88
}
```

### 4. Oncology Lab

#### Simulate Cancer Treatment

```http
POST /oncology/simulate
```

**Request Body:**
```json
{
  "cancer_type": "breast",
  "stage": 2,
  "mutations": ["BRCA1", "TP53"],
  "treatment_protocol": "standard"
}
```

**Response:**
```json
{
  "cancer_type": "breast",
  "stage": 2,
  "mutations": ["BRCA1", "TP53"],
  "predicted_response": {
    "tumor_reduction": 0.65,
    "progression_free_survival_months": 24.5,
    "overall_survival_months": 72.3,
    "response_rate": "partial_response"
  },
  "recommended_protocol": {
    "drugs": ["Paclitaxel", "Carboplatin"],
    "dosing": "21-day cycles",
    "duration": "6 months"
  },
  "confidence": 0.82
}
```

### 5. Drug Discovery Lab

#### Virtual Screening

```http
POST /drug/screen
```

**Request Body:**
```json
{
  "target_protein": "EGFR",
  "screening_mode": "fast",
  "num_candidates": 10000
}
```

**Response:**
```json
{
  "target_protein": "EGFR",
  "screening_mode": "fast",
  "candidates_screened": 10000,
  "top_hits": [
    {
      "compound_id": "CMPD_000001",
      "smiles": "C10H12N2O",
      "binding_affinity": -11.3,
      "admet_score": 0.87,
      "drug_likeness": 0.92
    }
  ],
  "computation_time": 100.0
}
```

### 6. Genomics Lab

#### Analyze Variants

```http
POST /genomics/analyze
```

**Request Body:**
```json
{
  "genome_sequence": "ATCG...",
  "analysis_type": "variant",
  "reference_genome": "hg38"
}
```

**Response:**
```json
{
  "sequence_length": 50000,
  "analysis_type": "variant",
  "reference_genome": "hg38",
  "variants_found": 42,
  "pathogenic_variants": 3,
  "genes_affected": ["BRCA1", "TP53", "PTEN"],
  "pathways_enriched": [
    {"pathway": "DNA_REPAIR", "p_value": 0.001}
  ],
  "confidence": 0.91
}
```

### 7. Immune Response Lab

#### Simulate Immune Response

```http
POST /immune/simulate
```

**Request Body:**
```json
{
  "pathogen_type": "virus",
  "immune_state": "normal",
  "intervention": "mRNA_vaccine"
}
```

**Response:**
```json
{
  "pathogen_type": "virus",
  "immune_state": "normal",
  "intervention": "mRNA_vaccine",
  "response_dynamics": {
    "peak_response_hours": 72,
    "antibody_titer": 1280,
    "t_cell_count": 2500,
    "inflammation_score": 3.5
  },
  "clearance_time_hours": 168,
  "protection_duration_months": 18,
  "confidence": 0.85
}
```

### 8. Metabolic Syndrome Lab

#### Analyze Metabolic Condition

```http
POST /metabolic/analyze
```

**Request Body:**
```json
{
  "condition": "diabetes",
  "biomarkers": {
    "glucose": 180,
    "hba1c": 8.5,
    "triglycerides": 250
  },
  "intervention": "diet"
}
```

**Response:**
```json
{
  "condition": "diabetes",
  "biomarkers": {...},
  "intervention": "diet",
  "risk_score": 72,
  "predicted_outcomes": {
    "glucose_change_percent": -28.3,
    "lipid_profile_improvement": 35.2,
    "weight_loss_kg": 8.7,
    "cardiovascular_risk_reduction": 42.1
  },
  "personalized_recommendations": [
    "Mediterranean diet",
    "150 min/week exercise"
  ],
  "confidence": 0.87
}
```

## WebSocket API

For real-time streaming results:

```javascript
const ws = new WebSocket('wss://api.qulab.io/ws/quantum');

ws.onopen = () => {
  ws.send(JSON.stringify({
    system_type: 'molecule',
    num_qubits: 4
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'progress') {
    console.log(`Progress: ${data.percent}%`);
  } else if (data.type === 'complete') {
    console.log('Result:', data.data);
  }
};
```

## Batch Processing

Process multiple requests in a single call (Pro/Enterprise only):

```http
POST /batch
```

**Request Body:**
```json
{
  "lab": "materials",
  "requests": [
    {"material_name": "Steel_304", "temperature": 300},
    {"material_name": "Aluminum_6061", "temperature": 300}
  ]
}
```

**Response:**
```json
{
  "batch_id": "abc123def456",
  "total_requests": 2,
  "results": [...]
}
```

## Rate Limiting

Rate limits are enforced per API key:

**Headers:**
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 987
X-RateLimit-Reset: 1635724800
```

**429 Response:**
```json
{
  "error": "Rate limit exceeded",
  "retry_after": 3600
}
```

## Error Codes

| Code | Meaning |
|------|---------|
| 400 | Bad Request - Invalid input |
| 401 | Unauthorized - Invalid API key |
| 403 | Forbidden - Insufficient permissions |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |

## SDKs

### Python

```python
from qulab_sdk import QuLabClient

client = QuLabClient(api_key="your_key")
result = client.materials.analyze(
    material_name="Steel_304",
    temperature=300.0,
    properties=["strength", "conductivity"]
)
print(result)
```

### JavaScript/Node.js

```javascript
const { QuLabClient } = require('qulab-sdk');

const client = new QuLabClient({ apiKey: 'your_key' });
const result = await client.materials.analyze({
  materialName: 'Steel_304',
  temperature: 300.0,
  properties: ['strength', 'conductivity']
});
console.log(result);
```

## Support

- **Documentation**: https://docs.qulab.io
- **Email**: support@qulab.io
- **Discord**: https://discord.gg/qulab
- **GitHub**: https://github.com/qulab/qulab-infinite

---

**Why we are credible:**
- 6.6M+ materials in database validated against NIST/experimental data
- Quantum simulations validated against IBM Qiskit and published benchmarks
- Drug discovery algorithms validated against ChEMBL and PubChem datasets
- Cancer models validated against clinical trial data (SEER, NCI)
- All algorithms peer-reviewed or based on published research
- Created by scientists for scientists

**Explore more:**
- Main site: https://qulab.io
- Research: https://research.qulab.io
- Blog: https://blog.qulab.io
