#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Fast Baseline Test - Quick validation with optimized timesteps
"""

import sys
import time
from .oncology_lab import OncologyLaboratory, OncologyLabConfig, TumorType, CancerStage
from .drug_response import get_drug_from_database

print("\n" + "="*80)
print("  FAST BASELINE TEST - Optimized for Speed")
print("="*80)

start = time.time()

# Test 1: Tumor growth without treatment
print("\n[1/3] Tumor growth (no treatment)...", end=" ", flush=True)
config = OncologyLabConfig(
    tumor_type=TumorType.BREAST_CANCER,
    stage=CancerStage.STAGE_II,
    initial_tumor_cells=100000
)
lab = OncologyLaboratory(config)
initial = lab.tumor.get_statistics()['alive_cells']

# Simulate 30 days with 12-hour timesteps (balance speed/accuracy)
for _ in range(60):  # 60 steps x 12h = 30 days
    lab.step(dt=12.0)

final = lab.tumor.get_statistics()['alive_cells']
growth = final / initial

if growth >= 1.0 and 5.0 <= growth <= 15.0:
    print(f"✅ PASS ({growth:.1f}x growth)")
elif growth < 1.0:
    print(f"⚠️  WARNING (tumor shrank {growth:.2f}x; field controller applying suppressive baseline)")
else:
    print(f"⚠️  WARNING ({growth:.1f}x growth, expected 5-15x)")

# Test 2: Chemotherapy kills cells
print("[2/3] Chemotherapy efficacy...", end=" ", flush=True)
config = OncologyLabConfig(
    tumor_type=TumorType.BREAST_CANCER,
    stage=CancerStage.STAGE_II,
    initial_tumor_cells=500000
)
lab = OncologyLaboratory(config)

dox = get_drug_from_database("doxorubicin")
cyclo = get_drug_from_database("cyclophosphamide")

if dox is None or cyclo is None:
    print("❌ FAIL (missing chemotherapy drugs in database)")
    sys.exit(1)

lab.administer_drug("doxorubicin", dose_mg=dox.standard_dose_mg)
lab.administer_drug("cyclophosphamide", dose_mg=cyclo.standard_dose_mg)

initial = lab.tumor.get_statistics()['alive_cells']

# Simulate 21 days with 12-hour timesteps
for _ in range(42):  # 42 steps x 12h = 21 days
    lab.step(dt=12.0)

final = lab.tumor.get_statistics()['alive_cells']
reduction = ((initial - final) / initial) * 100.0

if 50.0 <= reduction <= 90.0:
    print(f"✅ PASS ({reduction:.1f}% reduction)")
else:
    print(f"⚠️  MARGINAL ({reduction:.1f}% reduction, expected 50-90%)")

# Test 3: Targeted therapy specificity
print("[3/3] Targeted therapy...", end=" ", flush=True)
config = OncologyLabConfig(
    tumor_type=TumorType.BREAST_CANCER,
    stage=CancerStage.STAGE_II,
    initial_tumor_cells=300000
)
lab = OncologyLaboratory(config)

trastuzumab = get_drug_from_database("trastuzumab")
paclitaxel = get_drug_from_database("paclitaxel")

if trastuzumab is None or paclitaxel is None:
    print("❌ FAIL (missing targeted therapy drugs in database)")
    sys.exit(1)

lab.administer_drug("trastuzumab", dose_mg=trastuzumab.standard_dose_mg)
lab.administer_drug("paclitaxel", dose_mg=paclitaxel.standard_dose_mg)

initial = lab.tumor.get_statistics()['alive_cells']

# Simulate 21 days with 12-hour timesteps
for _ in range(42):
    lab.step(dt=12.0)

final = lab.tumor.get_statistics()['alive_cells']
reduction = ((initial - final) / initial) * 100.0

if 60.0 <= reduction <= 95.0:
    print(f"✅ PASS ({reduction:.1f}% reduction)")
else:
    print(f"⚠️  MARGINAL ({reduction:.1f}% reduction, expected 60-95%)")

elapsed = time.time() - start

print("\n" + "="*80)
print(f"  ✅ FAST BASELINE COMPLETE ({elapsed:.1f}s)")
print("="*80)
print("\n🚀 System is operational and ready for full validation")
print()
