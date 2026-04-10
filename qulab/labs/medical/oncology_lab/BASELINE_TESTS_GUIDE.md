# Oncology Lab Baseline Accuracy Tests

## Purpose

Quick sanity checks to verify core simulation mechanics before running full clinical trial validation.

## Test Suite (8 Tests)

### Test 1: Tumor Growth Without Treatment
- **What**: Verifies untreated tumor grows exponentially then saturates
- **Expected**: 5-15x growth in 30 days
- **Why**: Validates Gompertzian growth model

### Test 2: Drug Pharmacokinetics
- **What**: Verifies drug concentration decays with correct half-life
- **Expected**: After 2 half-lives, ~25% of peak concentration remains
- **Why**: Validates PK model (critical for dose-response)

### Test 3: Chemotherapy Efficacy
- **What**: Standard AC regimen (doxorubicin + cyclophosphamide)
- **Expected**: 50-90% cell kill in 21 days
- **Why**: Validates chemotherapy cell death mechanics

### Test 4: Targeted Therapy Specificity
- **What**: Trastuzumab + paclitaxel in HER2+ breast cancer
- **Expected**: 60-95% reduction
- **Why**: Validates targeted therapy is more effective than chemo

### Test 5: Immunotherapy Response
- **What**: Pembrolizumab in PD-L1+ lung cancer
- **Expected**: 30-70% reduction
- **Why**: Validates immunotherapy has lower response rate but still effective

### Test 6: Stage-Dependent Response
- **What**: Cisplatin in Stage II vs Stage IV ovarian cancer
- **Expected**: Stage II responds 1.5-3x better
- **Why**: Validates stage affects treatment response

### Test 7: Combination Synergy
- **What**: FOLFOX vs 5-FU alone in colorectal cancer
- **Expected**: Combination 1.2-2x more effective
- **Why**: Validates drug combinations work synergistically

### Test 8: Tumor Type Specificity
- **What**: Paclitaxel in ovarian cancer vs melanoma
- **Expected**: Ovarian responds 1.5-3x better
- **Why**: Validates different tumor types respond differently

## Running the Tests

```bash
# From QuLabInfinite directory
cd /Users/noone/QuLabInfinite
python3 -m oncology_lab.baseline_accuracy_tests
```

## Expected Runtime

- **Per test**: 2-4 minutes
- **Total suite**: ~20-30 minutes
- Much faster than full clinical trial validation (hours)

## Passing Criteria

- **Target**: 100% pass rate (8/8 tests)
- **Acceptable**: ≥75% pass rate (6/8 tests)
- **Needs tuning**: <75% pass rate

## Interpreting Results

### All Tests Pass (8/8)
✅ **System ready for clinical trial validation**
- Core mechanics validated
- Drug parameters accurate
- Tumor models realistic

### Most Tests Pass (6-7/8)
⚠️ **System mostly accurate, minor tuning needed**
- Check which test failed
- Adjust specific parameter (growth rate, drug sensitivity, etc.)
- Re-run failed tests

### Many Tests Fail (<6/8)
❌ **System needs calibration**
- Review tumor growth parameters
- Check drug IC50/EC50 values
- Verify PK parameters (half-lives, clearance)
- Re-validate against known outcomes

## Common Issues

### Issue: Test 1 fails (Tumor growth too fast/slow)
**Fix**: Adjust `intrinsic_growth_rate` and `gompertz_retardation` in tumor profiles

### Issue: Test 2 fails (Drug decay incorrect)
**Fix**: Verify `half_life` and `elimination_rate` in drug PK models

### Issue: Test 3 fails (Chemo not killing enough cells)
**Fix**: Increase `emax` or decrease `ec50` for chemotherapy drugs

### Issue: Test 5 fails (Immunotherapy too effective)
**Fix**: Reduce `emax` for PD-1/PD-L1 inhibitors (should be 0.4-0.6)

### Issue: Test 6 fails (No stage difference)
**Fix**: Ensure `_derive_tumor_profile()` adjusts `drug_sensitivity` by stage

### Issue: Test 7 fails (No combination synergy)
**Fix**: May need to implement explicit synergy multipliers for known combinations

## Next Steps After Baseline Tests

1. **If all pass**: Run full 100-trial validation
2. **If some fail**: Tune parameters based on failed tests
3. **If many fail**: Review fundamental model assumptions

## Technical Details

### Tolerance Levels
- Most tests: ±15-20% tolerance
- Immunotherapy: ±25% (higher variance expected)
- Tumor type specificity: ±30% (more variability)

### Simulation Parameters
- Time step: 1.0 hour (balance of accuracy vs speed)
- Simulation duration: 14-30 days per test
- Cell counts: 100,000-800,000 initial cells

### Why These 8 Tests?

Chosen to cover:
- ✅ Core mechanics (growth, PK)
- ✅ All major drug classes (chemo, targeted, immuno)
- ✅ Key clinical phenomena (synergy, resistance, specificity)
- ✅ Quick to run (20-30 min vs hours for full validation)

## Validation Hierarchy

```
Baseline Tests (8 tests, 20-30 min)
    ↓ Pass
Clinical Trial Subset (10 trials, 1-2 hours)
    ↓ Pass
Full Validation (100 trials, 6-10 hours)
    ↓ Pass
Production Ready
```

## Troubleshooting

### Tests timeout or hang
- Reduce simulation steps in tests
- Use larger time step (dt=2.0 instead of 1.0)
- Check for infinite loops in tumor growth

### Results highly variable between runs
- Add random seed for reproducibility
- Check if stochastic cell death is too random
- Increase cell count for more stable statistics

### All tests fail catastrophically
- Verify drug database loaded correctly
- Check tumor initialization
- Ensure PK model calculates concentrations properly
