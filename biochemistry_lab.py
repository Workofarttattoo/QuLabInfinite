"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

BIOCHEMISTRY LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List
from scipy.constants import pi, Avogadro

@dataclass
class BiochemistryLab:
    temperature: float = 298.15   # K
    pressure: float = 101325     # Pa (STP)
    
    def calculate_concentration(self, moles: float, volume: float) -> np.float64:
        """Calculate the concentration of a substance given its number of moles and volume."""
        return np.float64(moles / volume)

    def buffer_solution_ph(self, weak_acid_conc: float, conjugate_base_conc: float, pka: float) -> np.float64:
        """Calculate pH of a buffer solution using the Henderson-Hasselbalch equation."""
        numerator = self.calculate_concentration(conjugate_base_conc, 1)
        denominator = self.calculate_concentration(weak_acid_conc, 1)
        ph = pka + np.log10(numerator / denominator)
        return np.float64(ph)

    def standard_molar_enthalpy(self, substance: str) -> Tuple[np.float64, np.float64]:
        """Calculate the standard molar enthalpies of formation for a given chemical substance."""
        # Placeholder function - actual values would come from NIST or other reliable sources
        if substance == "H2O":
            delta_hf_0 = np.float64(-285.83)  # kJ/mol, standard molar enthalpy of formation at 298K
        elif substance == "CO2":
            delta_hf_0 = np.float64(-393.51)  # kJ/mol
        else:
            raise ValueError(f"Standard molar enthalpies for {substance} are not defined.")
        
        return (delta_hf_0, self.temperature)
    
    def rate_constant(self, activation_energy: float, pre_exponential_factor: np.float64, temperature: np.float64) -> np.float64:
        """Calculate the rate constant using Arrhenius equation."""
        r = 8.3145 * 1e-3   # J / K mol
        k = pre_exponential_factor * np.exp(-activation_energy / (r * temperature))
        return np.float64(k)

    def enzyme_kinetics(self, substrate_conc: float, enzyme_conc: float, k_m: float) -> np.float64:
        """Calculate the reaction rate of an enzymatic reaction using Michaelis-Menten equation."""
        v_max = 10.0 * enzyme_conc   # Arbitrary value
        velocity = (v_max * substrate_conc) / (k_m + substrate_conc)
        return np.float64(velocity)

    def solubility_product(self, substance: str) -> Tuple[np.float64, List[str]]:
        """Calculate the solubility product of a given substance."""
        if substance == "AgCl":
            k_sp = 1.77 * 1e-10   # Solubility product constant
            ions = ["Ag+", "Cl-"]
        else:
            raise ValueError(f"Solubility product for {substance} is not defined.")
        
        return (k_sp, ions)
    
def run_demo():
    lab = BiochemistryLab()
    print(lab.calculate_concentration(0.5, 2))
    print(lab.buffer_solution_ph(0.1, 0.09, 4.76))
    standard_hf_0, t = lab.standard_molar_enthalpy("H2O")
    print(f"Standard molar enthalpy of formation for H2O at {t}K: {standard_hf_0:.2f} kJ/mol")
    print(lab.rate_constant(75000, np.float64(1e13), 300))
    print(lab.enzyme_kinetics(0.1, 1e-6, 1e-3))
    solubility_product, ions = lab.solubility_product("AgCl")
    print(f"Solubility product for AgCl: {solubility_product}")
    print(f"Ions involved: {', '.join(ions)}")

if __name__ == '__main__':
    run_demo()