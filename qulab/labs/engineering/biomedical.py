from dataclasses import dataclass, field
import numpy as np
from scipy import constants

@dataclass
class BiomedicalEngineeringLab:
    """
    A class for biomedical engineering calculations and simulations using NumPy and SciPy.

    Attributes:
        T (float): Temperature in Kelvin.
        conc (float): Concentration of a substance in mol/L.
        charge (int): Charge number of an ion or molecule.
        mass (float): Mass in grams.
        volume (float): Volume in liters.
        length (float): Length in meters.

    Methods:
        calculate_viscosity: Calculate the viscosity of blood using given constants and parameters.
        calculate_diffusion_coefficient: Calculate the diffusion coefficient of a substance in blood plasma.
        calculate_thermodynamic_potential: Calculate the thermodynamic potential for a chemical reaction.
    """

    T: float = field(default=310.0, metadata={"help": "Temperature in Kelvin."})
    conc: float = field(default=1.0e-6, metadata={"help": "Concentration in mol/L."})
    charge: int = field(default=1, metadata={"help": "Charge number of an ion or molecule."})
    mass: float = field(default=180.12, metadata={"help": "Mass in grams."})
    volume: float = field(default=1.0, metadata={"help": "Volume in liters."})
    length: float = field(default=0.1, metadata={"help": "Length in meters."})

    def calculate_viscosity(self) -> np.ndarray:
        """
        Calculate the viscosity of blood using constants and parameters.

        Returns:
            numpy.ndarray: Viscosity values.
        """
        return np.array([constants.k * self.T / (self.volume * 1e-3)])

    def calculate_diffusion_coefficient(self) -> np.ndarray:
        """
        Calculate the diffusion coefficient of a substance in blood plasma using constants and parameters.

        Returns:
            numpy.ndarray: Diffusion coefficient values.
        """
        return np.array([constants.k * self.T / (6.0 * np.pi * constants.g * 1e-3)])

    def calculate_thermodynamic_potential(self) -> np.ndarray:
        """
        Calculate the thermodynamic potential for a chemical reaction using constants and parameters.

        Returns:
            numpy.ndarray: Thermodynamic potential values.
        """
        return np.array([constants.k * self.T * np.log(constants.Avogadro * self.conc)])

def run_demo():
    lab = BiomedicalEngineeringLab()
    print(f"Viscosity of blood: {lab.calculate_viscosity()} Pa*s")
    print(f"Diffusion coefficient in plasma: {lab.calculate_diffusion_coefficient()} m^2/s")
    print(f"Thermodynamic potential for a reaction: {lab.calculate_thermodynamic_potential()} J/mol")

if __name__ == '__main__':
    run_demo()