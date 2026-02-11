#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Chemistry Lab - Quick Sanity Check
Runs fast checks on all core components (<10s)
"""

import sys
import time
import importlib.util

def check_module(module_name):
    print(f"Checking {module_name}...", end=" ", flush=True)
    try:
        if importlib.util.find_spec(module_name) is None:
            print("❌ NOT FOUND")
            return False
        importlib.import_module(module_name)
        print("✅ OK")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_fast_kinetics():
    print("\nTesting Fast Kinetics...", end=" ", flush=True)
    try:
        from chemistry_lab.fast_kinetics_solver import FastKineticsSolver
        solver = FastKineticsSolver()
        k, _ = solver.get_rate_constant("H2O2_decomposition", 298.15)
        assert k > 0
        print(f"✅ OK (k={k:.2e})")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_fast_equilibrium():
    print("Testing Fast Equilibrium...", end=" ", flush=True)
    try:
        from chemistry_lab.fast_equilibrium_solver import FastEquilibriumSolver
        solver = FastEquilibriumSolver()
        pH = solver.pH_weak_acid(0.1, 1.8e-5) # Acetic acid
        assert 2.0 < pH < 4.0
        print(f"✅ OK (pH={pH:.2f})")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_fast_thermodynamics():
    print("Testing Fast Thermodynamics...", end=" ", flush=True)
    try:
        from chemistry_lab.fast_thermodynamics import FastThermodynamicsCalculator
        calc = FastThermodynamicsCalculator()
        dG = calc.gibbs_free_energy(-45.0, -85.0, 310.15)
        assert dG < 0 # Spontaneous
        print(f"✅ OK (dG={dG:.2f} kJ/mol)")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def main():
    print("="*60)
    print("CHEMISTRY LAB - QUICK SANITY CHECK")
    print("="*60)

    modules = [
        "chemistry_lab.fast_kinetics_solver",
        "chemistry_lab.fast_equilibrium_solver",
        "chemistry_lab.fast_thermodynamics"
    ]

    all_modules_ok = all(check_module(m) for m in modules)

    if all_modules_ok:
        k_ok = test_fast_kinetics()
        e_ok = test_fast_equilibrium()
        t_ok = test_fast_thermodynamics()

        if k_ok and e_ok and t_ok:
            print("\n" + "="*60)
            print("✅ ALL SYSTEMS OPERATIONAL")
            print("="*60)
            sys.exit(0)

    print("\n" + "="*60)
    print("❌ SYSTEM FAILURE")
    print("="*60)
    sys.exit(1)

if __name__ == "__main__":
    main()
