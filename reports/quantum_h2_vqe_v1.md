# quantum_h2_vqe_v1 – VQE Benchmark Check

**Date:** 2025-10-30  
**Dataset:** `data/raw/quantum/h2_sto3g_vqe.json`  
**Canonical summary:** `data/canonical/quantum/h2_vqe_summary.json`

## Inputs
- H₂ molecule at 0.74 Å, STO-3G basis.
- VQE results for:
  - Deterministic statevector backend (shots = 0).
  - Mock noisy simulator (shots = 8192).
- Reference exact energy: −1.13727 Hartree.

## Results

| Metric | Target | Observed | Status |
| --- | --- | --- | --- |
| Mean absolute error (mHa) | ≤ 1.0 | 2.245 | ❌ |
| Coverage @ 95% (CI contains reference) | ≥ 0.9 | 0.00 | ❌ |
| Backends evaluated | – | 2 | – |

## Observations
- The noisy simulator deviates by ~4.1 mHa, pulling the average MAE above the 1 mHa gate.
- Neither backend reports confidence intervals that include the exact reference energy, yielding 0% coverage.
- The canonical calibration metadata (mae=0.4 mHa, ci coverage=0.8 mHa) is inconsistent with the stored raw data.

## Recommendations
1. **Separate benchmarks.** Track noiseless and noisy backends independently with appropriate acceptance thresholds.
2. **Update confidence intervals.** Recompute statistical bounds so the reference energy sits within the claimed coverage band.
3. **Revise canonical summary.** Align the stored calibration evidence with the observed metrics or regenerate the raw dataset.
