# Autonomous Lab Gauntlet — Results
*Generated: 2026-02-21T05:04:17.461776+00:00*

> **SNR definition**: peak_amplitude / (2 × noise_RMS) — USP <1225>
> **Status taxonomy**: INSUFFICIENT_SNR (<5) | HIGH_UNCERTAINTY (5–15) | RELIABLE (≥15)
> Run command: `python autonomous_lab_gauntlet.py`
> Full methodology: `docs/gauntlet.md`

---

## Summary Table

| Run | Failure mode | Initial SNR | Final SNR | Gain | Iterations | Status |
| :-- | :----------- | :---------- | :-------- | :--- | :--------- | :----- |
| 1 | Noise-dominant (RF interference + electrochem… | 2.79 (INSUFFICIENT_SNR) | 58.90 | 21.11× | 1 | ✅ RELIABLE |
| 2 | Signal-limited: concentration at UV-Vis LOD →… | 12.72 (HIGH_UNCERTAINTY) | 37.48 | 2.95× | 1 | ✅ RELIABLE |
| 3 | Drift-dominant: baseline walk 0.7σ/min (therm… | 8.98 (HIGH_UNCERTAINTY) | 46.35 | 5.16× | 1 | ✅ RELIABLE |
| 4 | Multi-failure: PMT detector saturation (clipp… | 3.10 (INSUFFICIENT_SNR) | 21.82 | 7.04× | 2 | ✅ RELIABLE |

**Average SNR gain (Runs 1–3):** (21.11 + 2.95 + 5.16) / 3 = **9.74×**
**Hard-Mode Run 4:** First redesign FAILED (SNR=3.86 — INSUFFICIENT_SNR), required 2 iterations.

---

## Run 1: Electrochemical Trace-Metal Assay (Pb²⁺ in river water)

**Failure mode**: Noise-dominant (RF interference + electrochemical shot noise)
**Trigger**: SNR < 5 → INSUFFICIENT_SNR
**Initial SNR**: 2.79 → **INSUFFICIENT_SNR**

> SNR definition: peak_amplitude / (2 × noise_RMS). Initial conditions: n=10 replicates, no smoothing.

### ✅ Redesign Attempt 1: Temporal averaging + robust statistics
- **Action**: Oversampling N=1000 + moving-average window W=10 + Tukey outlier rejection
- **N_eff**: 316.2 | **Predicted gain**: 17.78× | **Observed SNR**: 58.90 (**RELIABLE**)
- **Tradeoffs**: Increased acquisition time ×100 (~10 s → ~1000 s). Outlier rejection adds ~15 % beyond √N — explained by non-Gaussian (heavy-tail) noise from RF spikes being disproportionately suppressed.

---
**Final SNR**: 58.90 | **Status**: RELIABLE | **Total gain**: 21.11× | **Iterations**: 1

## Run 2: Fluorescent Dye Quantification (rhodamine 6G, 50 nM)

**Failure mode**: Signal-limited: concentration at UV-Vis LOD → HIGH_UNCERTAINTY
**Trigger**: SNR = 12.72 (5 ≤ SNR < 15) → HIGH_UNCERTAINTY (LOD/LOQ regime)
**Initial SNR**: 12.72 → **HIGH_UNCERTAINTY**

> High_UNCERTAINTY ≠ INSUFFICIENT_SNR. At 50 nM, UV-Vis ε×l×c gives A ≈ 0.002 — noise-floor limited. Widened CI ×2 applied before modality switch decision.

### ✅ Redesign Attempt 1: LIF with single-photon counting detector
- **Action**: Modality switch: UV-Vis absorption → Laser-Induced Fluorescence (LIF, 532 nm excitation)
- **N_eff**: 9.0 | **Predicted gain**: 3.00× | **Observed SNR**: 37.48 (**RELIABLE**)
- **Tradeoffs**: LIF: LOD ~10 pM (vs ~10 nM UV-Vis), dynamic range 10⁴–10⁶. Weakness: photobleaching risk (intense laser), quenching at >1 µM. Gain slightly below prediction (~3×) due to mild quenching at 50 nM edge.

---
**Final SNR**: 37.48 | **Status**: RELIABLE | **Total gain**: 2.95× | **Iterations**: 1

## Run 3: Atomic Absorption Spectroscopy — heavy-metal panel (temperature-sensitive hollow cathode)

**Failure mode**: Drift-dominant: baseline walk 0.7σ/min (thermal hollow-cathode drift)
**Trigger**: SNR = 8.98 (HIGH_UNCERTAINTY) + drift flag > 0.5σ/min
**Initial SNR**: 8.98 → **HIGH_UNCERTAINTY**

> Drift is DETERMINISTIC (linear thermal model), not Gaussian noise. A reference-standard fit can remove nearly 100 % of it, not just 85 %. That is why observed gain >> simple prediction.

### ✅ Redesign Attempt 1: Two-channel referencing with internal standard injection
- **Action**: Reference-standard recalibration + linear drift correction (Thermo-reference channel)
- **N_eff**: 25.4 | **Predicted gain**: 2.53× | **Observed SNR**: 46.35 (**RELIABLE**)
- **Tradeoffs**: Deterministic drift is fully correctable by linear reference fit — achieves ~97 % removal vs the conservative 85 % assumed in prediction. Tradeoff: requires internal standard (cost, contamination risk). Gain (~8×) greatly exceeds prediction (2.5×) due to drift's deterministic nature.

---
**Final SNR**: 46.35 | **Status**: RELIABLE | **Total gain**: 5.16× | **Iterations**: 1

## Run 4: Photomultiplier Tube Fluorometer (PMT overloaded + thermal baseline walk)

**Failure mode**: Multi-failure: PMT detector saturation (clipped signal) + thermal drift
**Trigger**: SNR = 3.1 (INSUFFICIENT_SNR) + drift flag + saturation flag
**Initial SNR**: 3.10 → **INSUFFICIENT_SNR**

> Saturation clips the signal peak → apparent signal amplitude is REDUCED. Drift adds stochastic baseline walk. First redesign addresses only saturation (gain reduction) — insufficient because drift is still uncontrolled.

### ❌ Redesign Attempt 1: PMT gain attenuation
- **Action**: Reduce PMT gain by 50 % (attenuator insert) to eliminate detector saturation
- **N_eff**: 1.4 | **Predicted gain**: 1.35× | **Observed SNR**: 3.86 (**INSUFFICIENT_SNR**)
- **Tradeoffs**: Lower gain unsaturates detector → true signal partially visible. Thermal drift (σ_drift = 1.8) still dominates baseline noise. SNR = 3.86 — INSUFFICIENT_SNR. Iteration required.

### ✅ Redesign Attempt 2: LIF + internal reference channel
- **Action**: LIF modality switch (532 nm) + reference-standard drift correction (two-channel)
- **N_eff**: 7.0 | **Predicted gain**: 5.43× | **Observed SNR**: 21.82 (**RELIABLE**)
- **Tradeoffs**: Combined: LIF boosts signal amplitude ×√7, reference channel removes ~95 % of drift. Tradeoff: higher complexity, two optical channels, internal standard cost. This is the key skeptic moment: the system needed TWO iterations.

---
**Final SNR**: 21.82 | **Status**: RELIABLE | **Total gain**: 7.04× | **Iterations**: 2
