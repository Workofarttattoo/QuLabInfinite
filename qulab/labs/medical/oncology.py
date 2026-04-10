"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

ONCOLOGY LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import TypeVar, Generic
from scipy import constants

T = TypeVar('T')

@dataclass
class OncologyData:
    """Represent patient data for oncology analysis."""
    patient_id: int
    age: int
    gender: str
    cancer_type: str
    tumor_size: float  # in cm
    lymph_nodes_involved: bool
    metastasis: bool

@dataclass
class CancerParameters:
    """Base class to represent parameters related to specific cancers."""
    base_growth_rate: np.ndarray = field(default_factory=lambda: np.array([0.5], dtype=np.float64))
    mutation_rate: np.ndarray = field(default_factory=lambda: np.array([0.2], dtype=np.float64))
    cell_death_rate: np.ndarray = field(default_factory=lambda: np.array([0.1], dtype=np.float64))

@dataclass
class LungCancerParams(CancerParameters):
    """Specific parameters for lung cancer."""
    base_growth_rate: np.ndarray = field(default_factory=lambda: np.array([0.75], dtype=np.float64))
    mutation_rate: np.ndarray = field(default_factory=lambda: np.array([0.35], dtype=np.float64))

@dataclass
class BreastCancerParams(CancerParameters):
    """Specific parameters for breast cancer."""
    base_growth_rate: np.ndarray = field(default_factory=lambda: np.array([0.65], dtype=np.float64))
    mutation_rate: np.ndarray = field(default_factory=lambda: np.array([0.25], dtype=np.float64))

@dataclass
class OncologyLab:
    """Oncology lab to run simulations on cancer growth and drug efficacy."""
    
    def __init__(self, patient_data: OncologyData):
        self.patient_data = patient_data
        self.parameters = None
    
    def set_cancer_type(self, cancer_type: str) -> 'OncologyLab':
        """Set the specific type of cancer for this lab instance."""
        if cancer_type == 'Lung Cancer':
            self.parameters = LungCancerParams()
        elif cancer_type == 'Breast Cancer':
            self.parameters = BreastCancerParams()
        else:
            raise ValueError("Unsupported cancer type.")
        return self
    
    def simulate_growth(self, time_days: int) -> np.ndarray:
        """Simulate tumor growth over a given period."""
        if not self.parameters:
            raise ValueError("Set cancer type before simulating growth.")
        
        growth_rate = self.parameters.base_growth_rate[0]
        # Simplistic exponential growth model
        return np.expm1(growth_rate * time_days) + 1
    
    def simulate_drug_effect(self, drug_efficacy: float, days_treatment: int) -> tuple:
        """Simulate the effect of a given drug on tumor size."""
        if not self.parameters:
            raise ValueError("Set cancer type before simulating growth.")
        
        # Drug efficacy reduces growth rate
        adjusted_growth_rate = np.maximum(self.parameters.base_growth_rate[0] * (1 - drug_efficacy), 0)
        return self.simulate_growth(days_treatment), adjusted_growth_rate
    
    def calculate_probability_survival(self, mutation_rate: float) -> float:
        """Calculate probability of survival based on mutation rate."""
        if not self.parameters:
            raise ValueError("Set cancer type before calculating mutation effects.")
        
        # Simple model where survival decreases with increasing mutation rate
        return np.exp(-self.patient_data.age * self.parameters.mutation_rate[0] / (constants.c + constants.h))

def run_demo():
    """Run a demo of the oncology lab."""
    patient = OncologyData(1, 45, 'F', "Lung Cancer", 3.5, True, False)
    
    lab = OncologyLab(patient).set_cancer_type("Lung Cancer")
    
    growth_over_time = [lab.simulate_growth(t) for t in range(0, 365, 7)]
    print(r"Simulated tumor size over time (cm):", growth_over_time[:10])
    
    drug_effect = lab.simulate_drug_effect(drug_efficacy=0.8, days_treatment=21)
    print(r"Tumor size after treatment (days), adjusted growth rate:", drug_effect)
    
    survival_prob = lab.calculate_probability_survival(mutation_rate=lab.parameters.mutation_rate[0])
    print(r"Calculated probability of survival:", survival_prob)

if __name__ == '__main__':
    run_demo()