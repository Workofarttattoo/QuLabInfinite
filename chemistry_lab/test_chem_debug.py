#!/usr/bin/env python3
"""
Chemistry Lab Debug Tool
"""
import sys
import code
from chemistry_lab.fast_kinetics_solver import FastKineticsSolver
from chemistry_lab.fast_equilibrium_solver import FastEquilibriumSolver
from chemistry_lab.fast_thermodynamics import FastThermodynamicsCalculator

def debug_shell():
    print("Starting Chemistry Lab Debug Shell...")
    kinetics = FastKineticsSolver()
    equilibrium = FastEquilibriumSolver()
    thermo = FastThermodynamicsCalculator()

    print("Objects available: kinetics, equilibrium, thermo")

    # Start interactive shell
    try:
        import IPython
        IPython.embed(header="Chemistry Lab Debug Shell")
    except ImportError:
        code.interact(local=locals())

if __name__ == "__main__":
    debug_shell()
