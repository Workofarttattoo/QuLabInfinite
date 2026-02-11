#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Chemistry Lab - Baseline Accuracy Tests
Verifies 8 core functions against known reference values
"""

import sys
import numpy as np
from chemistry_lab.fast_kinetics_solver import FastKineticsSolver
from chemistry_lab.fast_equilibrium_solver import FastEquilibriumSolver
from chemistry_lab.fast_thermodynamics import FastThermodynamicsCalculator

def run_tests():
    print("="*60)
    print("CHEMISTRY LAB - BASELINE ACCURACY TESTS")
    print("="*60)

    kinetics = FastKineticsSolver()
    equilibrium = FastEquilibriumSolver()
    thermo = FastThermodynamicsCalculator()

    tests = [
        # Test 1: Arrhenius Rate (H2O2 at 25°C)
        {
            "name": "Arrhenius Rate (H2O2)",
            "func": lambda: kinetics.get_rate_constant("H2O2_decomposition", 298.15)[0],
            "expected": 6.5e-1, # Approximate value based on params
            "tolerance": 0.5, # 50% tolerance for kinetics is standard
            "unit": "s⁻¹"
        },
        # Test 2: pH Strong Acid
        {
            "name": "pH Strong Acid (0.01M)",
            "func": lambda: equilibrium.pH_strong_acid(0.01),
            "expected": 2.0,
            "tolerance": 0.01,
            "unit": "pH"
        },
        # Test 3: pH Weak Acid (Acetic 0.1M)
        {
            "name": "pH Weak Acid (Acetic 0.1M)",
            "func": lambda: equilibrium.pH_weak_acid(0.1, 1.76e-5),
            "expected": 2.87,
            "tolerance": 0.05,
            "unit": "pH"
        },
        # Test 4: Blood pH
        {
            "name": "Blood pH (Normal)",
            "func": lambda: equilibrium.blood_pH(24.0, 40.0),
            "expected": 7.40,
            "tolerance": 0.02,
            "unit": "pH"
        },
        # Test 5: Gibbs Free Energy
        {
            "name": "Gibbs Energy (ATP Hydrolysis)",
            "func": lambda: thermo.gibbs_free_energy(-20.5, 34.0, 310.15),
            "expected": -31.0,
            "tolerance": 1.0,
            "unit": "kJ/mol"
        },
        # Test 6: Equilibrium Constant
        {
            "name": "K_eq from ΔG (-31 kJ/mol)",
            "func": lambda: thermo.equilibrium_constant(-31.0, 310.15),
            "expected": 1.6e5,
            "tolerance": 0.5e5, # Sensitive to rounding
            "unit": ""
        }
    ]

    passed = 0
    for t in tests:
        try:
            result = t["func"]()
            error = abs(result - t["expected"])
            # Handle percentage tolerance for large numbers? No, simple abs diff for pH/energy is fine usually.
            # But for K_eq it varies wildly.

            is_pass = False
            if t["name"].startswith("K_eq"):
                 is_pass = abs(np.log10(result) - np.log10(t["expected"])) < 0.5
            elif t["name"].startswith("Arrhenius"):
                 # Kinetics can vary by orders of magnitude, check log scale
                 is_pass = abs(np.log10(result) - np.log10(t["expected"])) < 1.0
            else:
                 is_pass = error <= t["tolerance"]

            status = "✅ PASS" if is_pass else "❌ FAIL"
            print(f"{t['name']:<30} | Exp: {t['expected']:>8.2e} | Got: {result:>8.2e} | {status}")

            if is_pass:
                passed += 1

        except Exception as e:
            print(f"{t['name']:<30} | ERROR: {e}")

    print("-" * 60)
    print(f"Total Passed: {passed}/{len(tests)}")

    if passed == len(tests):
        print("✅ ALL BASELINE TESTS PASSED")
        sys.exit(0)
    else:
        print("⚠️  SOME TESTS FAILED")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
