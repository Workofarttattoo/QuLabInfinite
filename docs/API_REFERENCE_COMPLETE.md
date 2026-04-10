# QuLabInfinite API Reference

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

## Table of Contents

1. [Authentication](#authentication)
2. [Rate Limiting](#rate-limiting)
3. [Medical Labs API](#medical-labs-api)
4. [Scientific Labs API](#scientific-labs-api)
5. [Batch Processing](#batch-processing)
6. [WebSocket API](#websocket-api)
7. [Export Formats](#export-formats)
8. [Error Handling](#error-handling)

## Base URL

```
Production: https://api.qulabinfinite.com
Development: http://localhost:9000
```

## Authentication

All API requests require authentication via API key in the header:

```http
Authorization: Bearer YOUR_API_KEY
```

### API Key Tiers

| Tier | Rate Limit | Features | Price |
|------|------------|----------|-------|
| Free | 100 req/hour | Basic access to all labs | $0/month |
| Pro | 1,000 req/hour | Batch processing, analytics | $99/month |
| Enterprise | 10,000 req/hour | Priority support, custom solutions | $999/month |

### Example Request

```bash
curl -X POST https://api.qulabinfinite.com/labs/alzheimers/assess \
  -H "Authorization: Bearer demo_pro" \
  -H "Content-Type: application/json" \
  -d '{"csf_abeta42": 420, "csf_ptau": 95}'
```

## Rate Limiting

Rate limits are enforced per API key per hour. Response headers indicate your current usage:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 847
X-RateLimit-Reset: 1635724800
```

When limit exceeded:

```json
{
  "error": "Rate limit exceeded",
  "detail": "Try again in 1847 seconds",
  "status_code": 429
}
```

## Medical Labs API

### 1. Alzheimer's Early Detection

**Endpoint:** `POST /labs/alzheimers/assess`

**Request Body:**
```json
{
  "csf_abeta42_pg_ml": 420,
  "csf_ptau_pg_ml": 95,
  "csf_ttau_pg_ml": 520,
  "amyloid_pet_suvr": 1.45,
  "hippocampal_volume_cm3": 2.8,
  "apoe4_copies": 1,
  "age_years": 72,
  "sex": "F"
}
```

**Response:**
```json
{
  "atn_classification": {
    "amyloid": "positive",
    "tau": "positive",
    "neurodegeneration": "present"
  },
  "stage": "MCI due to AD - high likelihood",
  "risk_score": 0.78,
  "progression_prediction": {
    "5_year_risk": 0.65,
    "10_year_risk": 0.82
  },
  "confidence": 0.91,
  "computation_time_seconds": 0.042
}
```

### 2. Parkinson's Progression Predictor

**Endpoint:** `POST /labs/parkinsons/assess`

**Request Body:**
```json
{
  "mds_updrs_iii": 42,
  "disease_duration_years": 5,
  "ledd_mg_day": 650,
  "motor_fluctuations": true,
  "dyskinesia_present": false
}
```

**Response:**
```json
{
  "hoehn_yahr_stage": 2.5,
  "motor_subtype": "tremor_dominant",
  "motor_complications": {
    "dyskinesia_risk_3yr": 0.35,
    "wearing_off_risk_3yr": 0.52
  },
  "progression_forecast": {
    "hy_stage_5yr": 3.0,
    "updrs_increase_per_year": 3.2
  },
  "confidence": 0.87
}
```

### 3. Autoimmune Disease Classifier

**Endpoint:** `POST /labs/autoimmune/classify`

**Request Body:**
```json
{
  "rf_positive": true,
  "anti_ccp_positive": true,
  "ana_titer": "1:320",
  "tender_joint_count": 8,
  "swollen_joint_count": 6,
  "symptom_duration_weeks": 12
}
```

**Response:**
```json
{
  "primary_diagnosis": "rheumatoid_arthritis",
  "acr_eular_score": 8,
  "differential_probabilities": {
    "rheumatoid_arthritis": 0.89,
    "systemic_lupus": 0.08,
    "sjogrens": 0.03
  },
  "disease_activity": "high",
  "treatment_recommendation": "DMARD + biologics",
  "confidence": 0.93
}
```

### 4. Sepsis Early Warning System

**Endpoint:** `POST /labs/sepsis/assess`

**Request Body:**
```json
{
  "respiratory_rate": 28,
  "systolic_bp": 88,
  "heart_rate": 115,
  "temperature_c": 38.5,
  "wbc_count": 15.2,
  "lactate_mmol_l": 3.5,
  "mental_status": "confused"
}
```

**Response:**
```json
{
  "qsofa_score": 2,
  "sofa_score": 8,
  "news2_score": 9,
  "sepsis_classification": "septic_shock",
  "intervention_urgency": "immediate",
  "predicted_mortality": 0.32,
  "code_sepsis_activated": true,
  "resuscitation_protocol": "30ml/kg crystalloid",
  "confidence": 0.96
}
```

### 5. Wound Healing Optimizer

**Endpoint:** `POST /labs/wound/optimize`

### 6. Bone Density Predictor

**Endpoint:** `POST /labs/bone/assess`

### 7. Kidney Function Calculator

**Endpoint:** `POST /labs/kidney/calculate`

### 8. Liver Disease Staging

**Endpoint:** `POST /labs/liver/stage`

### 9. Lung Function Analyzer

**Endpoint:** `POST /labs/lung/analyze`

### 10. Pain Management Optimizer

**Endpoint:** `POST /labs/pain/optimize`

## Scientific Labs API

### 1. Quantum Computing Lab

**Endpoint:** `POST /labs/quantum/simulate`

**Request Body:**
```json
{
  "system_type": "molecule",
  "molecule": "H2",
  "num_qubits": 4,
  "circuit_depth": 12,
  "algorithm": "vqe",
  "backend": "statevector_simulator"
}
```

**Response:**
```json
{
  "ground_state_energy_hartree": -1.137,
  "convergence_iterations": 85,
  "fidelity": 0.9993,
  "quantum_advantage": "2.3x speedup vs classical",
  "computation_time_seconds": 0.453,
  "confidence": 0.97
}
```

### 2. Materials Science Lab

**Endpoint:** `POST /labs/materials/predict`

**Request Body:**
```json
{
  "composition": "GaN",
  "crystal_structure": "wurtzite",
  "temperature_K": 300,
  "pressure_GPa": 1,
  "properties_requested": ["band_gap", "formation_energy", "bulk_modulus"]
}
```

**Response:**
```json
{
  "material": "GaN",
  "properties": {
    "band_gap_eV": 3.4,
    "formation_energy_eV_atom": -1.15,
    "bulk_modulus_GPa": 210,
    "thermal_conductivity_W_mK": 130
  },
  "database_coverage": "6.6M materials",
  "confidence": 0.95
}
```

### 3. Protein Engineering Lab

**Endpoint:** `POST /labs/protein/design`

### 4. Chemistry Lab

**Endpoint:** `POST /labs/chemistry/simulate`

### 5. Genomics Lab

**Endpoint:** `POST /labs/genomics/analyze`

### 6. Nanotechnology Lab

**Endpoint:** `POST /labs/nano/design`

### 7. Renewable Energy Lab

**Endpoint:** `POST /labs/renewable/optimize`

### 8. Semiconductor Lab

**Endpoint:** `POST /labs/semiconductor/design`

### 9. Neuroscience Lab

**Endpoint:** `POST /labs/neuroscience/model`

### 10. Oncology Lab

**Endpoint:** `POST /labs/oncology/simulate`

## Batch Processing

**Endpoint:** `POST /batch`

Process multiple experiments in a single request (Pro and Enterprise only).

**Request Body:**
```json
{
  "lab": "materials",
  "experiments": [
    {"composition": "GaN", "properties": ["band_gap"]},
    {"composition": "SiC", "properties": ["band_gap"]},
    {"composition": "AlN", "properties": ["band_gap"]}
  ]
}
```

**Response:**
```json
{
  "batch_id": "b7f8a9c2d3e4f5g6",
  "total": 3,
  "completed": 3,
  "results": [
    {"experiment_id": 0, "status": "success", "result": {...}},
    {"experiment_id": 1, "status": "success", "result": {...}},
    {"experiment_id": 2, "status": "success", "result": {...}}
  ]
}
```

## WebSocket API

Real-time results streaming for long-running simulations.

**Endpoint:** `ws://api.qulabinfinite.com/ws`

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:9000/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({
    lab: 'quantum',
    request: {system_type: 'molecule', num_qubits: 10}
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Progress:', data.progress);
  console.log('Result:', data.result);
};
```

**Messages:**
```json
// Progress update
{
  "type": "progress",
  "lab": "quantum",
  "progress": 0.45,
  "status": "simulating"
}

// Final result
{
  "type": "complete",
  "lab": "quantum",
  "result": {...},
  "computation_time_seconds": 2.34
}
```

## Export Formats

**Endpoint:** `POST /export`

Export results in various formats.

**Request Body:**
```json
{
  "results": {...},
  "format": "json"  // Options: json, csv, pdf
}
```

**Supported Formats:**
- `json`: Structured JSON output
- `csv`: Tabular data export
- `pdf`: Publication-ready report (Enterprise only)

## Error Handling

All errors follow consistent format:

```json
{
  "error": "Error type",
  "detail": "Detailed error message",
  "status_code": 400,
  "timestamp": "2025-11-03T22:00:00Z"
}
```

**Common Error Codes:**

| Code | Meaning | Solution |
|------|---------|----------|
| 400 | Bad Request | Check request body format |
| 401 | Unauthorized | Verify API key |
| 403 | Forbidden | Upgrade tier or check permissions |
| 429 | Rate Limit Exceeded | Wait or upgrade tier |
| 500 | Internal Server Error | Contact support |

## Analytics

**Endpoint:** `GET /analytics`

View usage statistics (Pro and Enterprise only).

**Response:**
```json
{
  "total_requests": 15234,
  "by_endpoint": {
    "alzheimers": {"requests": 1234, "avg_latency_ms": 42},
    "quantum": {"requests": 2345, "avg_latency_ms": 453}
  },
  "uptime_seconds": 86400,
  "success_rate": 0.998
}
```

## Support

- **Documentation:** https://docs.qulabinfinite.com
- **Status Page:** https://status.qulabinfinite.com
- **Support Email:** support@qulabinfinite.com
- **Enterprise Support:** enterprise@qulabinfinite.com

---

**Version:** 1.0.0
**Last Updated:** November 3, 2025
**Copyright:** Corporation of Light - Patent Pending
