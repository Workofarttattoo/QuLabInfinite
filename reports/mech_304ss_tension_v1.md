# mech_304ss_tension_v1 – Johnson-Cook Calibration Check

**Date:** 2025-10-30  
**Dataset:** `QuLabInfinite/data/raw/mechanics/304ss_tension_{298K,673K}.json`  
**Canonical summary:** `data/canonical/mechanics/304ss_tension_summary.json`

## Inputs
- Quasi-static tensile curves for annealed AISI 304 stainless steel at 298 K and 673 K
- Reference Johnson-Cook parameters from the canonical summary (`A=275 MPa`, `B=525 MPa`, `n=0.45`, `C=0.015`, `m=0.9`)

## Results

| Metric | Target | Canonical Params | Refitted (random search) | Status |
| --- | --- | --- | --- | --- |
| Mean absolute error (MPa) | ≤ 15 | 53.1 | 37.4 | ❌ |
| RMSE (MPa) | (diagnostic) | 87.6 | 60.6 | – |
| Coverage @ 90% (fraction within ±1.645σ) | ≥ 0.88 | 0.25 | 0.25 | ❌ |
| Stress points evaluated | – | 16 | 16 | – |

## Observations
- The canonical coefficients substantially over-predict the 298 K curve and under-predict high strain response at 673 K, yielding MAE ≈ 53 MPa.
- A bounded random search across {A, B, n, m} cannot push MAE below ~37 MPa given the current data and model form. Coverage never exceeds 0.44, far below the 0.88 gate.
- Measurement uncertainties in the raw files (σ ≈ 6–8 MPa) make the current acceptance thresholds unattainable for the supplied data.

## Recommendations
1. **Revisit target gates.** Either relax the MAE / coverage requirements or supply higher fidelity curves that justify the original <15 MPa target.
2. **Augment the model.** Consider adding strain hardening saturation or temperature-dependent strength terms beyond the single exponent `m`.
3. **Document limitations.** Until recalibrated, communicate that Johnson-Cook predictions carry ~40 MPa typical error for the provided dataset.
4. **Store fitted parameters.** If the refitted coefficients are adopted, update the canonical summary and provenance metadata accordingly.
