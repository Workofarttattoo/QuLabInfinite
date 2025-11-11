"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

ELECTROCHEMISTRY LAB
Free gift to the scientific community from QuLabInfinite.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.constants import k, Avogadro, g, c, h, e, pi, physical_constants
from typing import List
import scipy.constants

# Constants and configuration
R = physical_constants['molar gas constant'][0]  # Molar gas constant

@dataclass
class ElectrochemicalSolution:
    concentration: np.ndarray  # Concentration of species in mol/L
    temperature: float  # Temperature in Kelvin
    potential: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))  # Potential in volts

@dataclass
class Electrode:
    surface_area: float  # Surface area in m^2
    active_material_concentration: np.ndarray  # Active material concentration in mol/m^3
    
@dataclass
class System:
    solution: ElectrochemicalSolution
    electrode: Electrode

@dataclass
class NernstEquation:
    system: System
    n: int  # Number of electrons transferred in the redox reaction
    stoichiometry: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))  # Stoichiometric coefficients for species

@dataclass
class TafelEquation:
    system: System
    beta_anode: float  # Tafel slope at anode in V/decade
    beta_cathode: float  # Tafel slope at cathode in V/decade

def calculate_nernst(equation: NernstEquation) -> np.ndarray:
    return equation.system.solution.potential - (R * equation.system.solution.temperature / (n * e)) * np.log(np.prod((equation.stoichiometry ** equation.system.solution.concentration)))

def calculate_tafel(equation: TafelEquation, overpotential: float) -> np.ndarray:
    i_o = 1e-6  # Exchange current density in A/m^2
    return i_o * (np.exp((overpotential / beta_anode)) - np.exp((-overpotential / beta_cathode)))

def run_demo():
    c_solution = np.array([0.1, 0.2], dtype=np.float64)
    t_solution = 298.15
    p_solution = np.array([-1.0], dtype=np.float64)

    surface_area_electrode = 0.01
    c_active_material_electrode = np.array([0.5, 0.3], dtype=np.float64)
    
    e_solution = ElectrochemicalSolution(c_solution, t_solution, p_solution)
    e_electrode = Electrode(surface_area_electrode, c_active_material_electrode)

    system = System(e_solution, e_electrode)
    
    nernst_equation = NernstEquation(system=system, n=2, stoichiometry=np.array([1.0, 1.0], dtype=np.float64))
    tafel_equation = TafelEquation(system=system, beta_anode=120e-3, beta_cathode=-120e-3)
    
    nernst_potential = calculate_nernst(nernst_equation)
    tafel_current = calculate_tafel(tafel_equation, 0.5)

    print(f"Nernst Potential: {nernst_potential}")
    print(f"Tafel Current Density: {tafel_current}")

if __name__ == '__main__':
    run_demo()