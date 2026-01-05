#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

test_ode_debug - Part of Oncology Lab
"""

"""Debug ODE model to see what's happening"""

from oncology_lab.fast_ode_validator import TumorODEModel
from oncology_lab.drug_response import get_drug_from_database

# Test case: Breast cancer with doxorubicin + cyclophosphamide
# Expected: ~65% reduction (from clinical trial BREAST_001)

print("="*80)
print("  ODE MODEL DEBUG TEST")
print("="*80)

model = TumorODEModel(
    initial_cells=2500000.0,  # BREAST_001
    tumor_type='breast_cancer',
    stage=2
)

print(f"\nInitial cells: {model.cell_count:,.0f}")
print(f"Growth rate: {model.growth_rate}")
print(f"Carrying capacity: {model.carrying_capacity:,.0f}")

# Administer drugs
dox = get_drug_from_database("doxorubicin")
cyclo = get_drug_from_database("cyclophosphamide")

print(f"\nDoxorubicin:")
print(f"  Standard dose: {dox.standard_dose_mg} mg")
print(f"  Bioavailability: {dox.pk_model.bioavailability}")
print(f"  Vd: {dox.pk_model.volume_of_distribution} L")
print(f"  Half-life: {dox.pk_model.half_life} h")
print(f"  EC50: {dox.ec50} μM")
print(f"  Emax: {dox.emax}")

model.administer_drug("doxorubicin", dose_mg=dox.standard_dose_mg, time=0.0)
model.administer_drug("cyclophosphamide", dose_mg=cyclo.standard_dose_mg, time=0.0)

print(f"\nDrug info after administration:")
for i, drug_info in enumerate(model.drugs):
    print(f"  Drug {i}: {drug_info['drug'].name}")
    print(f"    C0: {drug_info['c0']:.4f} mg/L")
    print(f"    k: {drug_info['elimination_rate']:.6f} /h")

# Simulate with detailed output
print(f"\nSimulating 21 days (504 hours)...")
print(f"{'Time (h)':>10} {'Cells':>15} {'Growth Rate':>15} {'Kill Rate':>15}")
print(f"{'-'*10} {'-'*15} {'-'*15} {'-'*15}")

initial = model.cell_count
dt = 24.0
current_time = 0.0
target_time = 21 * 24  # 21 days

step_num = 0
while current_time < target_time:
    growth_rate = model._calculate_growth_rate(current_time)
    kill_rate = model._calculate_kill_rate(current_time)

    if step_num % 3 == 0:  # Print every 3 days
        print(f"{current_time:10.0f} {model.cell_count:15,.0f} {growth_rate:15,.0f} {kill_rate:15,.0f}")

    model.step(dt, current_time)
    current_time += dt
    step_num += 1

final = model.cell_count

print(f"\nFinal cells: {final:,.0f}")
reduction = ((initial - final) / initial) * 100.0
print(f"Reduction: {reduction:.1f}%")
print(f"Expected: ~65%")
print(f"Error: {abs(reduction - 65.0):.1f}%")

# Check drug concentration decay
print(f"\nDrug concentration at key timepoints:")
for drug_info in model.drugs[:2]:
    drug_name = drug_info['drug'].name
    print(f"\n{drug_name}:")
    for t in [0, 24, 72, 168, 504]:
        conc = model._get_drug_concentration(drug_info, float(t))
        conc_uM = (conc * 1000.0) / drug_info['drug'].molecular_weight
        print(f"  t={t:4.0f}h: {conc:.6f} mg/L = {conc_uM:.6f} μM")
