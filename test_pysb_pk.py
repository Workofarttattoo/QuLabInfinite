"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Test PySB-PKPD library for cisplatin modeling
Compare against our current simple implementation
"""
import os
import pytest

if os.environ.get("QULAB_RUN_HEAVY_TESTS") != "1":
    pytest.skip("Set QULAB_RUN_HEAVY_TESTS=1 to run pharmacokinetic tests", allow_module_level=True)

import numpy as np
from pysb import Model, Parameter, Observable
from pysb.simulator import ScipyOdeSimulator
from pysb.pkpd import drug_monomer, one_compartment, dose_bolus, clearance, emax

print("=" * 80)
print("Testing Professional PK/PD Library (PySB) with Cisplatin")
print("=" * 80)

# Create new model
Model()

# Define cisplatin as the drug
drug_monomer('Cisplatin')

# Create 1-compartment model for cisplatin
# Central compartment with volume of distribution
one_compartment('Cisplatin', volume=20.0)  # V = 20 L (from FDA)

# Add clearance (elimination)
clearance('Cisplatin', kel=0.75)  # k_el = CL/V = 15/20 = 0.75 per hour

# Add bolus dose at t=0
dose_bolus('Cisplatin', amount=135.0, time=0.0)  # 135 mg dose

# Add observable for plasma concentration
Observable('Cisplatin_plasma', Cisplatin(loc='C'))

# Get the model
model = Model.model

print("\n✓ PySB Model Created")
print(f"  Compartments: {len(model.compartments)}")
print(f"  Parameters: {len(model.parameters)}")
print(f"  Rules: {len(model.rules)}")
print(f"  Observables: {len(model.observables)}")

# Simulate over 24 hours
print("\nSimulating 24 hours of pharmacokinetics...")

tspan = np.linspace(0, 24, 1000)
sim = ScipyOdeSimulator(model, tspan=tspan, integrator='lsoda')
result = sim.run()

# Get concentrations in molecules
cisplatin_molecules = result.observables['Cisplatin_plasma']

# Convert to mg (assuming drug amount is in molecules, need to convert)
# Actually, pysb.pkpd uses drug amounts in mass units, so molecules = mg
cisplatin_mg = cisplatin_molecules  # Already in mg

# Convert mg to μM: (mg / V_liters) / MW_g_per_mol * 1000
# V = 20 L, MW = 300.1 g/mol
V_L = 20.0
MW_g_per_mol = 300.1
cisplatin_mg_per_L = cisplatin_mg / V_L
cisplatin_uM = (cisplatin_mg_per_L / MW_g_per_mol) * 1000.0

print("✓ Simulation Complete")

# Display results at key timepoints
print(f"\nPlasma Concentrations:")
print(f"{'Time (h)':>10} {'Conc (μM)':>15} {'Molecules (mg)':>20}")
print("-" * 46)

timepoints = [0, 0.8, 2.0, 6.0, 12.0, 24.0]
for t in timepoints:
    idx = np.argmin(np.abs(tspan - t))
    conc_uM = cisplatin_uM[idx]
    molecules = cisplatin_mg[idx]
    print(f"{tspan[idx]:>10.2f} {conc_uM:>15.2f} {molecules:>20.2f}")

# Compare with standard 1-compartment equation
# C(t) = (Dose/V) * exp(-k*t)
print("\n" + "=" * 80)
print("Validation Against Standard PK Equation")
print("=" * 80)

dose_mg = 135.0
k_el = 0.75  # per hour
V = 20.0  # L

print(f"\n{'Time (h)':>10} {'PySB (μM)':>15} {'Standard (μM)':>15} {'Match':>10}")
print("-" * 52)

all_match = True
for t in timepoints:
    idx = np.argmin(np.abs(tspan - t))
    pysb_conc = cisplatin_uM[idx]

    # Standard equation
    c_mg_per_L = (dose_mg / V) * np.exp(-k_el * t)
    standard_conc = (c_mg_per_L / MW_g_per_mol) * 1000.0

    error_pct = abs(pysb_conc - standard_conc) / standard_conc * 100 if standard_conc > 0.01 else 0
    match = "✓" if error_pct < 1.0 else "✗"
    if error_pct >= 1.0:
        all_match = False

    print(f"{t:>10.1f} {pysb_conc:>15.2f} {standard_conc:>15.2f} {match:>10}")

# Assessment
print("\n" + "=" * 80)
print("PROFESSIONAL LIBRARY ASSESSMENT")
print("=" * 80)

if all_match:
    print("\n✓ PySB-PKPD matches standard equations within 1% error")
else:
    print("\n✗ Some deviations from standard equations detected")

print("\nAdvantages of PySB-PKPD:")
print("  ✓ Professional-grade ODE solver (LSODA)")
print("  ✓ Validated macros for PK/PD modeling")
print("  ✓ Can extend to multi-compartment models easily")
print("  ✓ Can add PD effects (Emax, sigmoidal, etc.)")
print("  ✓ Parameter estimation capabilities")
print("  ✓ Published and peer-reviewed")

print("\nWhat PySB-PKPD DOESN'T solve for cancer:")
print("  ✗ Still need tumor biology (cell cycle, death)")
print("  ✗ Still need spatial heterogeneity")
print("  ✗ Still need resistance mechanisms")
print("  ✗ Still need validation against clinical trials")

print("\nRecommendation:")
print("  • Use PySB-PKPD for ACCURATE drug pharmacokinetics")
print("  • Keep our tumor biology simulation")
print("  • Integrate the two for best of both worlds")

print("\n" + "=" * 80)
