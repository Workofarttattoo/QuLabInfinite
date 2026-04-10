#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Quick Sanity Check - Fast verification that oncology lab is working
"""

import sys
from pathlib import Path

print("\n" + "="*80)
print("  QUICK SANITY CHECK - Oncology Lab")
print("="*80)

# Test 1: Import modules
print("\n[1/5] Testing imports...", end=" ")
try:
    from .oncology_lab import OncologyLaboratory, OncologyLabConfig, TumorType, CancerStage
    from .drug_response import get_drug_from_database, DRUG_DATABASE
    print("✅ PASS")
except Exception as e:
    print(f"❌ FAIL: {e}")
    sys.exit(1)

# Test 2: Drug database loaded
print("[2/5] Checking drug database...", end=" ")
try:
    num_drugs = len(DRUG_DATABASE)
    if num_drugs >= 68:
        print(f"✅ PASS ({num_drugs} drugs)")
    else:
        print(f"❌ FAIL (only {num_drugs} drugs, expected 68+)")
        sys.exit(1)
except Exception as e:
    print(f"❌ FAIL: {e}")
    sys.exit(1)

# Test 3: Can retrieve specific drugs
print("[3/5] Testing drug retrieval...", end=" ")
try:
    test_drugs = ["doxorubicin", "cisplatin", "pembrolizumab", "olaparib", "trametinib"]
    for drug_name in test_drugs:
        drug = get_drug_from_database(drug_name)
        if not drug:
            print(f"❌ FAIL (missing {drug_name})")
            sys.exit(1)
    print(f"✅ PASS (all {len(test_drugs)} test drugs found)")
except Exception as e:
    print(f"❌ FAIL: {e}")
    sys.exit(1)

# Test 4: Can initialize lab
print("[4/5] Testing lab initialization...", end=" ")
try:
    config = OncologyLabConfig(
        tumor_type=TumorType.BREAST_CANCER,
        stage=CancerStage.STAGE_II,
        initial_tumor_cells=10000
    )
    lab = OncologyLaboratory(config)
    initial_cells = lab.tumor.get_statistics()['alive_cells']
    if initial_cells == 10000:
        print("✅ PASS")
    else:
        print(f"❌ FAIL (expected 10000 cells, got {initial_cells})")
        sys.exit(1)
except Exception as e:
    print(f"❌ FAIL: {e}")
    sys.exit(1)

# Test 5: Can administer drug and simulate
print("[5/5] Testing drug administration and simulation...", end=" ")
try:
    # Compare treated vs untreated
    # Untreated tumor
    config_control = OncologyLabConfig(
        tumor_type=TumorType.BREAST_CANCER,
        stage=CancerStage.STAGE_II,
        initial_tumor_cells=10000
    )
    lab_control = OncologyLaboratory(config_control)

    # Treated tumor
    config_treated = OncologyLabConfig(
        tumor_type=TumorType.BREAST_CANCER,
        stage=CancerStage.STAGE_II,
        initial_tumor_cells=10000
    )
    lab_treated = OncologyLaboratory(config_treated)

    dox = get_drug_from_database("doxorubicin")
    cyclo = get_drug_from_database("cyclophosphamide")
    if dox is None or cyclo is None:
        print("❌ FAIL (required chemotherapy drugs missing from database)")
        sys.exit(1)

    lab_treated.administer_drug("doxorubicin", dose_mg=dox.standard_dose_mg)
    lab_treated.administer_drug("cyclophosphamide", dose_mg=cyclo.standard_dose_mg)

    # Simulate 7 days
    for day in range(7):
        for hour in range(24):
            lab_control.step(dt=1.0)
            lab_treated.step(dt=1.0)

    control_cells = lab_control.tumor.get_statistics()['alive_cells']
    treated_cells = lab_treated.tumor.get_statistics()['alive_cells']

    # Treated should have fewer cells than untreated
    if treated_cells < control_cells:
        ratio = control_cells / treated_cells if treated_cells > 0 else float('inf')
        print(f"✅ PASS (treated has {ratio:.2f}x fewer cells)")
    else:
        print(f"❌ FAIL (treated={treated_cells}, control={control_cells})")
        sys.exit(1)
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("  ✅ ALL QUICK CHECKS PASSED - System is operational")
print("="*80)
print("\nNext steps:")
print("  • Run baseline_accuracy_tests.py for comprehensive validation")
print("  • Run batch_trial_validator.py for clinical trial comparison")
print()

sys.exit(0)
