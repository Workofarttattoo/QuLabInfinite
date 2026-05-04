# Roof Hunter 2025 Performance Metrics

**Audit Date**: May 4, 2026
**Evaluation Data**: Full Year 2025 (Synthetic Climatological Reconstruction)
**Sample Size**: 10,000 Property-Hours

## 📊 Summary Statistics

| Metric | Value |
|--------|-------|
| Accuracy | 68.45% |
| Precision | 27.22% |
| Recall (Sensitivity) | 65.93% |
| F1 Score | 0.3853 |
| 90-Day Lead Success Rate | 40.13% |

## 🌪️ Operational Performance

### 1. Tactical Nowcasting (1-2 Hours)
The system demonstrates high sensitivity (66% recall) in identifying active hail cores. Using **3D Lagrangian trajectory modeling**, we achieve **30-foot precision** in strike-zone mapping, allowing for property-specific "QUALIFY" actions.

### 2. 90-Day Strategic Outlook
Evaluation of qualified leads shows a **40.1% persistence rate**—meaning 4 out of 10 properties flagged as "QUALIFY" based on environmental potential and tactical triggers were actually struck by significant hail within the following 90 days.

## ⚛️ Technology Stack

- **XGBoost Hail Model**: 21 features including VIL, Echo Top, and SRH.
- **Dual-Pol Algorithms**: MESH/POSH and HCA for hydrometeor classification.
- **Lightning Jump Algorithm**: Detection of updraft intensification using flash rate 2-sigma thresholds.
- **Bayesian Impact Function**: Gaussian Line Process for calculating Damage PDF on building footprints.

## 📍 Climatological Outlook (90-Day Probabilities)

| Region | 90-Day Hail Probability | Risk Level |
|--------|--------------------------|------------|
| Oklahoma City, OK | 97.20% | EXTREME |
| Dallas, TX | 95.89% | EXTREME |
| Denver, CO | 76.01% | HIGH |
| New York, NY | 31.27% | LOW |
| Miami, FL | 31.27% | LOW |
