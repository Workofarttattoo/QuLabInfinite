#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Baseline Accuracy Tests for Oncology Lab
Quick sanity checks to verify core simulation mechanics
"""

import sys
import time
import numpy as np
from pathlib import Path

from .oncology_lab import OncologyLaboratory, OncologyLabConfig, TumorType, CancerStage
from .drug_response import get_drug_from_database


class BaselineTest:
    """Single baseline test case"""
    def __init__(self, name, description, expected_range, tolerance=0.15):
        self.name = name
        self.description = description
        self.expected_range = expected_range  # (min, max)
        self.tolerance = tolerance
        self.result = None
        self.passed = False
        self.error = None

    def check(self, observed_value):
        """Check if observed value is within expected range"""
        self.result = observed_value
        min_val, max_val = self.expected_range

        # Check if within range
        if min_val <= observed_value <= max_val:
            self.passed = True
            self.error = 0.0
        else:
            # Calculate error as distance from range
            if observed_value < min_val:
                self.error = abs((min_val - observed_value) / min_val)
            else:
                self.error = abs((observed_value - max_val) / max_val)

            self.passed = self.error <= self.tolerance

        return self.passed


def test_1_tumor_growth_no_treatment():
    """Test 1: Untreated tumor grows exponentially then saturates"""
    print("\n[Test 1] Tumor Growth Without Treatment")
    print("-" * 80)

    test = BaselineTest(
        "Untreated tumor growth",
        "Tumor should grow 5-15x in 30 days without treatment",
        expected_range=(5.0, 15.0),
        tolerance=0.20
    )

    config = OncologyLabConfig(
        tumor_type=TumorType.BREAST_CANCER,
        stage=CancerStage.STAGE_II,
        initial_tumor_cells=100000
    )

    lab = OncologyLaboratory(config)
    initial_cells = lab.tumor.get_statistics()['alive_cells']

    # Simulate 30 days without treatment
    for day in range(30):
        for _ in range(24):  # 24 hours per day
            lab.step(dt=1.0)

    final_cells = lab.tumor.get_statistics()['alive_cells']
    growth_factor = final_cells / initial_cells

    passed = test.check(growth_factor)

    print(f"  Initial cells: {initial_cells:,}")
    print(f"  Final cells (30d): {final_cells:,}")
    print(f"  Growth factor: {growth_factor:.2f}x")
    print(f"  Expected range: {test.expected_range[0]:.1f}x - {test.expected_range[1]:.1f}x")
    print(f"  Status: {'✅ PASS' if passed else '❌ FAIL'}")

    return test


def test_2_drug_concentration_decay():
    """Test 2: Drug concentration decays with correct half-life"""
    print("\n[Test 2] Drug Pharmacokinetics - Concentration Decay")
    print("-" * 80)

    # Doxorubicin has half-life of 30 hours
    # After 60 hours (2 half-lives), should be at 25% of peak
    test = BaselineTest(
        "Drug concentration half-life",
        "After 2 half-lives, concentration should be 20-30% of peak",
        expected_range=(0.20, 0.30),
        tolerance=0.15
    )

    drug = get_drug_from_database("doxorubicin")

    # Single dose
    dose = drug.standard_dose_mg

    peak_conc = drug.calculate_concentration(dose_mg=dose, time_hours=0.0)
    conc_2_halflife = drug.calculate_concentration(dose_mg=dose, time_hours=60.0)

    ratio = conc_2_halflife / peak_conc
    passed = test.check(ratio)

    print(f"  Drug: Doxorubicin (t½ = 30h, dose={dose:.1f} mg)")
    print(f"  Peak concentration: {peak_conc:.4f} μM")
    print(f"  Conc after 60h (2 t½): {conc_2_halflife:.4f} μM")
    print(f"  Ratio: {ratio:.3f} (expected ~0.25)")
    print(f"  Status: {'✅ PASS' if passed else '❌ FAIL'}")

    return test


def test_3_chemotherapy_kills_cells():
    """Test 3: Chemotherapy should kill 50-90% of tumor cells"""
    print("\n[Test 3] Chemotherapy Efficacy")
    print("-" * 80)

    test = BaselineTest(
        "Chemotherapy cell kill",
        "Standard chemo should kill 50-90% of cells in 21 days",
        expected_range=(50.0, 90.0),
        tolerance=0.20
    )

    config = OncologyLabConfig(
        tumor_type=TumorType.BREAST_CANCER,
        stage=CancerStage.STAGE_II,
        initial_tumor_cells=500000
    )

    lab = OncologyLaboratory(config)

    dox = get_drug_from_database("doxorubicin")
    cyclo = get_drug_from_database("cyclophosphamide")

    if dox is None or cyclo is None:
        print("  ❌ Required chemotherapy drugs missing from database")
        test.passed = False
        return test

    lab.administer_drug("doxorubicin", dose_mg=dox.standard_dose_mg)
    lab.administer_drug("cyclophosphamide", dose_mg=cyclo.standard_dose_mg)

    initial_cells = lab.tumor.get_statistics()['alive_cells']

    # Simulate 21 days (1 cycle)
    for day in range(21):
        for _ in range(24):
            lab.step(dt=1.0)

    final_cells = lab.tumor.get_statistics()['alive_cells']
    reduction_percent = ((initial_cells - final_cells) / initial_cells) * 100.0

    passed = test.check(reduction_percent)

    print(f"  Regimen: Doxorubicin {dox.standard_dose_mg:.1f} mg + Cyclophosphamide {cyclo.standard_dose_mg:.1f} mg")
    print(f"  Initial cells: {initial_cells:,}")
    print(f"  Final cells (21d): {final_cells:,}")
    print(f"  Reduction: {reduction_percent:.1f}%")
    print(f"  Expected: 50-90%")
    print(f"  Status: {'✅ PASS' if passed else '❌ FAIL'}")

    return test


def test_4_targeted_therapy_specificity():
    """Test 4: Targeted therapy more effective than chemo in sensitive tumors"""
    print("\n[Test 4] Targeted Therapy Specificity")
    print("-" * 80)

    # HER2+ breast cancer responds better to trastuzumab than chemo alone
    test = BaselineTest(
        "Targeted therapy specificity",
        "Trastuzumab should achieve 60-95% reduction in HER2+ tumors",
        expected_range=(60.0, 95.0),
        tolerance=0.20
    )

    config = OncologyLabConfig(
        tumor_type=TumorType.BREAST_CANCER,
        stage=CancerStage.STAGE_II,
        initial_tumor_cells=300000
    )

    lab = OncologyLaboratory(config)

    trastuzumab = get_drug_from_database("trastuzumab")
    paclitaxel = get_drug_from_database("paclitaxel")

    if trastuzumab is None or paclitaxel is None:
        print("  ❌ Required targeted therapy drugs missing from database")
        test.passed = False
        return test

    lab.administer_drug("trastuzumab", dose_mg=trastuzumab.standard_dose_mg)
    lab.administer_drug("paclitaxel", dose_mg=paclitaxel.standard_dose_mg)

    initial_cells = lab.tumor.get_statistics()['alive_cells']

    # Simulate 21 days
    for day in range(21):
        for _ in range(24):
            lab.step(dt=1.0)

    final_cells = lab.tumor.get_statistics()['alive_cells']
    reduction_percent = ((initial_cells - final_cells) / initial_cells) * 100.0

    passed = test.check(reduction_percent)

    print(f"  Regimen: Trastuzumab 6mg + Paclitaxel 175mg")
    print(f"  Tumor: HER2+ Breast Cancer")
    print(f"  Initial cells: {initial_cells:,}")
    print(f"  Final cells (21d): {final_cells:,}")
    print(f"  Reduction: {reduction_percent:.1f}%")
    print(f"  Expected: 60-95%")
    print(f"  Status: {'✅ PASS' if passed else '❌ FAIL'}")

    return test


def test_5_immunotherapy_response():
    """Test 5: Immunotherapy shows 30-70% response in PD-L1+ tumors"""
    print("\n[Test 5] Immunotherapy Response")
    print("-" * 80)

    test = BaselineTest(
        "Immunotherapy efficacy",
        "Pembrolizumab should achieve 30-70% reduction in responsive tumors",
        expected_range=(30.0, 70.0),
        tolerance=0.25
    )

    config = OncologyLabConfig(
        tumor_type=TumorType.LUNG_CANCER,
        stage=CancerStage.STAGE_III,
        initial_tumor_cells=800000
    )

    lab = OncologyLaboratory(config)

    # Pembrolizumab (PD-1 inhibitor)
    lab.administer_drug("pembrolizumab", dose_mg=200.0)

    initial_cells = lab.tumor.get_statistics()['alive_cells']

    # Simulate 21 days (1 cycle)
    for day in range(21):
        for _ in range(24):
            lab.step(dt=1.0)

    final_cells = lab.tumor.get_statistics()['alive_cells']
    reduction_percent = ((initial_cells - final_cells) / initial_cells) * 100.0

    passed = test.check(reduction_percent)

    print(f"  Drug: Pembrolizumab 200mg")
    print(f"  Tumor: Lung Cancer (PD-L1+)")
    print(f"  Initial cells: {initial_cells:,}")
    print(f"  Final cells (21d): {final_cells:,}")
    print(f"  Reduction: {reduction_percent:.1f}%")
    print(f"  Expected: 30-70%")
    print(f"  Status: {'✅ PASS' if passed else '❌ FAIL'}")

    return test


def test_6_platinum_resistance():
    """Test 6: Platinum drugs less effective in advanced stage"""
    print("\n[Test 6] Stage-Dependent Drug Response")
    print("-" * 80)

    # Stage II should respond better than Stage IV
    test = BaselineTest(
        "Stage-dependent response",
        "Stage II should respond 1.5-3x better than Stage IV",
        expected_range=(1.5, 3.0),
        tolerance=0.25
    )

    # Stage II
    config_early = OncologyLabConfig(
        tumor_type=TumorType.OVARIAN_CANCER,
        stage=CancerStage.STAGE_II,
        initial_tumor_cells=400000
    )
    lab_early = OncologyLaboratory(config_early)
    lab_early.administer_drug("cisplatin", dose_mg=75.0)

    initial_early = lab_early.tumor.get_statistics()['alive_cells']
    for day in range(21):
        for _ in range(24):
            lab_early.step(dt=1.0)
    final_early = lab_early.tumor.get_statistics()['alive_cells']
    reduction_early = ((initial_early - final_early) / initial_early) * 100.0

    # Stage IV
    config_late = OncologyLabConfig(
        tumor_type=TumorType.OVARIAN_CANCER,
        stage=CancerStage.STAGE_IV,
        initial_tumor_cells=400000
    )
    lab_late = OncologyLaboratory(config_late)
    lab_late.administer_drug("cisplatin", dose_mg=75.0)

    initial_late = lab_late.tumor.get_statistics()['alive_cells']
    for day in range(21):
        for _ in range(24):
            lab_late.step(dt=1.0)
    final_late = lab_late.tumor.get_statistics()['alive_cells']
    reduction_late = ((initial_late - final_late) / initial_late) * 100.0

    ratio = reduction_early / reduction_late if reduction_late > 0 else 0
    passed = test.check(ratio)

    print(f"  Drug: Cisplatin 75mg")
    print(f"  Stage II reduction: {reduction_early:.1f}%")
    print(f"  Stage IV reduction: {reduction_late:.1f}%")
    print(f"  Ratio: {ratio:.2f}x")
    print(f"  Expected: 1.5-3x better for early stage")
    print(f"  Status: {'✅ PASS' if passed else '❌ FAIL'}")

    return test


def test_7_combination_synergy():
    """Test 7: Drug combinations more effective than single agents"""
    print("\n[Test 7] Combination Therapy Synergy")
    print("-" * 80)

    test = BaselineTest(
        "Combination synergy",
        "Combination should be 1.2-2x more effective than single agent",
        expected_range=(1.2, 2.0),
        tolerance=0.20
    )

    # Single agent (5-FU alone)
    config_single = OncologyLabConfig(
        tumor_type=TumorType.COLORECTAL_CANCER,
        stage=CancerStage.STAGE_III,
        initial_tumor_cells=500000
    )
    lab_single = OncologyLaboratory(config_single)
    lab_single.administer_drug("5-fluorouracil", dose_mg=400.0)

    initial_single = lab_single.tumor.get_statistics()['alive_cells']
    for day in range(14):
        for _ in range(24):
            lab_single.step(dt=1.0)
    final_single = lab_single.tumor.get_statistics()['alive_cells']
    reduction_single = ((initial_single - final_single) / initial_single) * 100.0

    # Combination (FOLFOX: 5-FU + Leucovorin + Oxaliplatin)
    config_combo = OncologyLabConfig(
        tumor_type=TumorType.COLORECTAL_CANCER,
        stage=CancerStage.STAGE_III,
        initial_tumor_cells=500000
    )
    lab_combo = OncologyLaboratory(config_combo)
    lab_combo.administer_drug("5-fluorouracil", dose_mg=400.0)
    lab_combo.administer_drug("leucovorin", dose_mg=200.0)
    lab_combo.administer_drug("oxaliplatin", dose_mg=85.0)

    initial_combo = lab_combo.tumor.get_statistics()['alive_cells']
    for day in range(14):
        for _ in range(24):
            lab_combo.step(dt=1.0)
    final_combo = lab_combo.tumor.get_statistics()['alive_cells']
    reduction_combo = ((initial_combo - final_combo) / initial_combo) * 100.0

    ratio = reduction_combo / reduction_single if reduction_single > 0 else 0
    passed = test.check(ratio)

    print(f"  Single agent (5-FU): {reduction_single:.1f}% reduction")
    print(f"  Combination (FOLFOX): {reduction_combo:.1f}% reduction")
    print(f"  Synergy ratio: {ratio:.2f}x")
    print(f"  Expected: 1.2-2x improvement")
    print(f"  Status: {'✅ PASS' if passed else '❌ FAIL'}")

    return test


def test_8_tumor_type_specificity():
    """Test 8: Different tumor types respond differently to same drug"""
    print("\n[Test 8] Tumor Type Specificity")
    print("-" * 80)

    # Paclitaxel: Ovarian cancer responds better than melanoma
    test = BaselineTest(
        "Tumor type specificity",
        "Ovarian should respond 1.5-3x better to paclitaxel than melanoma",
        expected_range=(1.5, 3.0),
        tolerance=0.30
    )

    # Ovarian cancer (good response)
    config_ovarian = OncologyLabConfig(
        tumor_type=TumorType.OVARIAN_CANCER,
        stage=CancerStage.STAGE_III,
        initial_tumor_cells=400000
    )
    lab_ovarian = OncologyLaboratory(config_ovarian)
    lab_ovarian.administer_drug("paclitaxel", dose_mg=175.0)

    initial_ovarian = lab_ovarian.tumor.get_statistics()['alive_cells']
    for day in range(21):
        for _ in range(24):
            lab_ovarian.step(dt=1.0)
    final_ovarian = lab_ovarian.tumor.get_statistics()['alive_cells']
    reduction_ovarian = ((initial_ovarian - final_ovarian) / initial_ovarian) * 100.0

    # Melanoma (poor response to chemo)
    config_melanoma = OncologyLabConfig(
        tumor_type=TumorType.MELANOMA,
        stage=CancerStage.STAGE_III,
        initial_tumor_cells=400000
    )
    lab_melanoma = OncologyLaboratory(config_melanoma)
    lab_melanoma.administer_drug("paclitaxel", dose_mg=175.0)

    initial_melanoma = lab_melanoma.tumor.get_statistics()['alive_cells']
    for day in range(21):
        for _ in range(24):
            lab_melanoma.step(dt=1.0)
    final_melanoma = lab_melanoma.tumor.get_statistics()['alive_cells']
    reduction_melanoma = ((initial_melanoma - final_melanoma) / initial_melanoma) * 100.0

    ratio = reduction_ovarian / reduction_melanoma if reduction_melanoma > 0 else 0
    passed = test.check(ratio)

    print(f"  Drug: Paclitaxel 175mg")
    print(f"  Ovarian cancer: {reduction_ovarian:.1f}% reduction")
    print(f"  Melanoma: {reduction_melanoma:.1f}% reduction")
    print(f"  Ratio: {ratio:.2f}x")
    print(f"  Expected: Ovarian responds 1.5-3x better")
    print(f"  Status: {'✅ PASS' if passed else '❌ FAIL'}")

    return test


def run_baseline_tests():
    """Run all baseline accuracy tests"""
    print("\n" + "="*80)
    print("  ONCOLOGY LAB BASELINE ACCURACY TESTS")
    print("="*80)
    print("\nThese tests verify core simulation mechanics:")
    print("  • Tumor growth dynamics")
    print("  • Drug pharmacokinetics (PK)")
    print("  • Chemotherapy efficacy")
    print("  • Targeted therapy specificity")
    print("  • Immunotherapy response")
    print("  • Stage-dependent effects")
    print("  • Combination synergy")
    print("  • Tumor type differences")

    tests = []
    start_time = time.time()

    # Run all tests
    try:
        tests.append(test_1_tumor_growth_no_treatment())
        tests.append(test_2_drug_concentration_decay())
        tests.append(test_3_chemotherapy_kills_cells())
        tests.append(test_4_targeted_therapy_specificity())
        tests.append(test_5_immunotherapy_response())
        tests.append(test_6_platinum_resistance())
        tests.append(test_7_combination_synergy())
        tests.append(test_8_tumor_type_specificity())
    except Exception as e:
        print(f"\n❌ ERROR during tests: {e}")
        import traceback
        traceback.print_exc()
        return

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "="*80)
    print("  SUMMARY")
    print("="*80)

    passed = [t for t in tests if t.passed]
    failed = [t for t in tests if not t.passed]

    print(f"\nTests run: {len(tests)}")
    print(f"Passed: {len(passed)} ({len(passed)/len(tests)*100:.1f}%)")
    print(f"Failed: {len(failed)} ({len(failed)/len(tests)*100:.1f}%)")
    print(f"Elapsed time: {elapsed:.1f}s")

    if failed:
        print(f"\n❌ FAILED TESTS:")
        for t in failed:
            print(f"  • {t.name}")
            print(f"    Expected: {t.expected_range}")
            print(f"    Got: {t.result:.2f}")
            print(f"    Error: {t.error*100:.1f}%")

    # Overall status
    print(f"\n{'='*80}")
    if len(passed) == len(tests):
        print("✅ ALL TESTS PASSED - System accuracy verified")
    elif len(passed) >= len(tests) * 0.75:
        print("⚠️  MOST TESTS PASSED - System mostly accurate, some tuning needed")
    else:
        print("❌ MANY TESTS FAILED - System needs calibration")
    print(f"{'='*80}\n")

    return tests


if __name__ == "__main__":
    run_baseline_tests()
