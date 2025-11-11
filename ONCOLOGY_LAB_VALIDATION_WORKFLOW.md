# Oncology Lab Validation Workflow

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

## Quick Reference: Parameter Changes & Testing

### When You Change Parameters

Whenever you modify **intervention deltas** or **growth multipliers**, follow this workflow to ensure end-to-end consistency.

---

## Key Parameter Locations

### 1. Intervention Deltas (`oncology_lab/ten_field_controller.py`)

**Lines 311-383** - Field intervention effects (per hour):

```python
# Example intervention deltas:
ketogenic_diet = FieldIntervention(
    effects={
        'glucose_mm': -0.5,      # ← Reduce glucose by 0.5 mM/hour
        'glutamine_mm': -0.04,   # ← Reduce glutamine
        'atp_adp_ratio': +0.02,  # ← Increase ATP/ADP
    },
)

hbot = FieldIntervention(
    effects={'oxygen_percent': +0.5},  # ← Increase O₂ by 0.5%/hour
)

hyperthermia = FieldIntervention(
    effects={'temperature_c': +0.1},  # ← Increase temp by 0.1°C/hour
)
```

**Current deltas:**
- Glucose: -0.5 mM/hour (ketogenic diet)
- Glutamine: -0.04 mM/hour
- Oxygen: +0.5%/hour (HBOT)
- Temperature: +0.1°C/hour (hyperthermia)
- ROS: +0.2 µM/hour (oxidative therapy)
- pH: +0.03/hour (normalization)
- Lactate: -0.4 mM/hour

---

### 2. Growth Multipliers (`oncology_lab/oncology_lab.py`)

**Lines 333-352** - Stage-dependent growth/sensitivity scaling:

```python
stage_growth = {
    CancerStage.STAGE_I: 0.75,   # ← 75% of base growth rate
    CancerStage.STAGE_II: 1.0,   # ← 100% (baseline)
    CancerStage.STAGE_III: 1.25, # ← 125% faster
    CancerStage.STAGE_IV: 1.45,  # ← 145% faster (metastatic)
}

stage_capacity = {
    CancerStage.STAGE_I: 0.6,    # ← 60% of base capacity
    CancerStage.STAGE_II: 1.0,
    CancerStage.STAGE_III: 1.2,
    CancerStage.STAGE_IV: 1.35,
}

stage_sensitivity = {
    CancerStage.STAGE_I: 1.15,   # ← More sensitive to drugs
    CancerStage.STAGE_II: 1.0,
    CancerStage.STAGE_III: 0.8,
    CancerStage.STAGE_IV: 0.65,  # ← Less sensitive (resistant)
}
```

---

### 3. Drug Response Curves (`oncology_lab/drug_response.py`)

**Lines 175-391** - Drug PK/PD parameters:

```python
"cisplatin": Drug(
    ic50=1.5,  # µM - concentration for 50% inhibition
    ec50=2.0,  # µM - concentration for 50% effect
    emax=0.95, # Maximum effect (95% kill)
    hill_coefficient=2.0,  # Steepness of dose-response
    pk_model=PharmacokineticModel(
        half_life=0.8,  # hours
        clearance=15.0,  # L/h
        tissue_penetration=0.3,  # 30% reaches tumor
    ),
)
```

---

## Validation Workflow

### Step 1: Run the Master Validation Script

```bash
python validate_oncology_consistency.py
```

This runs **7 comprehensive tests**:
1. ✅ Basic smoke test
2. ✅ Validation helpers
3. ✅ Import consistency
4. ✅ Parameter sanity (all 32 tumor/stage combinations)
5. ✅ Field intervention delta ranges
6. ✅ Drug database integrity
7. ✅ End-to-end simulation

**Expected output:**
```
================================================================================
  ✓ ALL TESTS PASSED - ONCOLOGY LAB IS CONSISTENT
================================================================================
```

---

### Step 2: Run Individual Test Components (Optional)

If the master script fails, drill down:

#### a) Basic Smoke Test
```bash
python test_oncology_lab.py
```
- Tests: Lab creation, untreated growth, drug admin, protocol application
- Duration: ~10 seconds

#### b) Validation Helpers
```bash
python oncology_lab/validation.py
```
- Shows: Clinical benchmarks, parameter sources
- Duration: ~5 seconds

#### c) Full Demo (Comprehensive)
```bash
python oncology_lab_demo.py
```
- Runs: Control, chemo, ECH0 protocol comparison
- Generates: `oncology_lab_comparison.png` plot
- Duration: ~60 seconds

---

### Step 3: Review Parameter Impact

After validation passes, check that your changes have the intended effect:

```python
from oncology_lab import OncologyLaboratory, OncologyLabConfig, TumorType, CancerStage
from oncology_lab.ten_field_controller import create_ech0_three_stage_protocol

# Create lab
config = OncologyLabConfig(
    tumor_type=TumorType.BREAST_CANCER,
    stage=CancerStage.STAGE_III,  # Test your stage
    initial_tumor_cells=100,
)
lab = OncologyLaboratory(config)

# Apply protocol with your modified deltas
protocol = create_ech0_three_stage_protocol()
lab.apply_intervention_protocol(protocol)

# Run and check field evolution
lab.run_experiment(duration_days=7, report_interval_hours=24)
results = lab.get_results()

# Verify intervention deltas are working
print(f"Glucose: {results['field_data']['glucose_mm'][0]:.2f} → {results['field_data']['glucose_mm'][-1]:.2f} mM")
print(f"Oxygen: {results['field_data']['oxygen_percent'][0]:.2f} → {results['field_data']['oxygen_percent'][-1]:.2f}%")
```

---

## Common Issues & Fixes

### Issue 1: Tests fail after changing intervention deltas

**Symptom:** Field values hit clipping limits (e.g., glucose = 0.0)

**Fix:** Reduce delta magnitudes:
```python
# Before (too aggressive)
'glucose_mm': -2.0,  # Hits 0 in ~3 hours

# After (gradual)
'glucose_mm': -0.5,  # Takes ~20 hours to deplete
```

---

### Issue 2: Growth multipliers cause negative values

**Symptom:** `AssertionError: Negative growth rate`

**Fix:** Ensure all multipliers are positive:
```python
stage_growth = {
    CancerStage.STAGE_I: 0.75,  # ✓ Positive
    CancerStage.STAGE_II: 1.0,
    # NOT: CancerStage.STAGE_III: -0.5,  # ✗ Would cause error
}
```

---

### Issue 3: Drug sensitivity out of range

**Symptom:** `AssertionError: Sensitivity out of range`

**Fix:** Keep sensitivity in [0.4, 1.4] (enforced at line 392):
```python
# Automatic clipping applied:
profile['drug_sensitivity'] = np.clip(profile['drug_sensitivity'], 0.4, 1.4)
```

---

## Parameter Change Checklist

When modifying parameters:

- [ ] Update the parameter in source file
- [ ] Run `python validate_oncology_consistency.py`
- [ ] Verify all 7 tests pass
- [ ] Run spot-check simulation to verify behavior
- [ ] Document the change rationale (why this value?)
- [ ] If changing based on new literature, update citations

---

## Validation Test Coverage

| Test | What It Checks | Files Validated |
|------|----------------|----------------|
| Basic Smoke | Lab creation, drug admin, protocols | All core modules |
| Validation Helpers | Clinical benchmarks loaded | `validation.py` |
| Import Consistency | No circular dependencies | All `__init__.py` |
| Parameter Sanity | 32 tumor/stage combos valid | `oncology_lab.py:333-392` |
| Field Deltas | Intervention effects reasonable | `ten_field_controller.py:311-383` |
| Drug Database | PK/PD params consistent | `drug_response.py:175-391` |
| End-to-End | Full simulation completes | Integration test |

**Total coverage:** 7 tests, ~100 assertions, 32 parameter combinations

---

## Files Modified by This System

```
QuLabInfinite/
├── oncology_lab/
│   ├── oncology_lab.py              ← Growth multipliers (lines 333-392)
│   ├── ten_field_controller.py      ← Intervention deltas (lines 311-383)
│   ├── drug_response.py             ← Drug PK/PD parameters
│   ├── tumor_simulator.py           ← Cell-level dynamics
│   └── validation.py                ← Clinical benchmarks
│
├── test_oncology_lab.py             ← Basic smoke test
├── oncology_lab_demo.py             ← Comprehensive demo
└── validate_oncology_consistency.py ← Master validation (THIS SCRIPT)
```

---

## Quick Command Reference

```bash
# Run everything (recommended after parameter changes)
python validate_oncology_consistency.py

# Individual test components
python test_oncology_lab.py                # Basic smoke test
python oncology_lab/validation.py          # Clinical benchmarks
python oncology_lab_demo.py                # Full comparison demo

# Check specific parameter
python -c "from oncology_lab.oncology_lab import OncologyLaboratory, OncologyLabConfig; \
           lab = OncologyLaboratory(OncologyLabConfig()); \
           print(lab._derive_tumor_profile(TumorType.BREAST_CANCER, CancerStage.STAGE_III))"
```

---

## Next Steps

After validation passes:

1. **Iterate on parameters** - Tune deltas/multipliers based on experiments
2. **Compare to clinical data** - Use `validation.py` benchmarks
3. **Document changes** - Update this file with your tuning rationale
4. **Version control** - Commit validated parameter sets

---

## Contact

For questions about validation workflow:
- **Author:** Joshua Hendricks Cole
- **System:** QuLabInfinite Oncology Lab
- **Validation Script:** `validate_oncology_consistency.py`

---

**Last Updated:** November 2025
**Validation Status:** ✅ ALL TESTS PASSING (7/7)
