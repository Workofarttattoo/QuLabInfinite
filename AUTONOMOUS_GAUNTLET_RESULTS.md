# Autonomous Lab Gauntlet — Evidence Sheet
**Command to reproduce:**
```bash
python autonomous_lab_gauntlet.py
```
Full machine-readable results are written to `reports/autonomous_gauntlet_results.json`.

---
## SNR Definition
```
SNR = (max(signal) − min(signal)) / (2 × std(baseline_region))
```
Status rubric:
- **INSUFFICIENT_SNR**: SNR < 5
- **HIGH_UNCERTAINTY**: 5 ≤ SNR < 15
- **RELIABLE**: SNR ≥ 15

---
## Run Summary
| Run | Initial SNR | Final SNR | Gain | Iterations | Final status |
|---:|---:|---:|---:|---:|:---|
| 1 | 2.79 | 58.90 | 21.11× | 1 | RELIABLE |
| 2 | 12.72 | 37.48 | 2.95× | 1 | RELIABLE |
| 3 | 8.98 | 46.35 | 5.16× | 1 | RELIABLE |
| 4 | 3.10 | 21.82 | 7.04× | 2 | RELIABLE |

---
## Hard-Mode Run 4 Contract
- First redesign SNR: **3.86** (must remain < 5)
- Final SNR: **21.82** (must reach ≥ 15)
- Iterations: **2** (must be 2)
