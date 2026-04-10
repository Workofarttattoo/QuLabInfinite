"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

CARDIAC FIBROSIS PREDICTOR
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.constants import pi

@dataclass
class CardiacFibrosisParameter:
    heart_rate: float = 60.0
    age: int = 45
    blood_pressure_systolic: int = 120
    blood_pressure_diastolic: int = 80
    cholesterol_total: float = 200
    hdl_cholesterol: float = 35
    triglycerides: float = 150
    diabetes_status: bool = False
    smoking_history_years: int = 0
    family_cardiovascular_disease: bool = False

@dataclass
class CardiacFibrosisModel:
    parameters: CardiacFibrosisParameter = field(default_factory=CardiacFibrosisParameter)

    def calculate_risk_score(self) -> float:
        risk_factors = {
            'heart_rate': 0.1,
            'age': 0.25,
            'blood_pressure_systolic': 0.15,
            'blood_pressure_diastolic': 0.15,
            'cholesterol_total': 0.2,
            'hdl_cholesterol': -0.2,
            'triglycerides': 0.15,
        }
        
        risk_score = 0
        
        for factor, weight in risk_factors.items():
            value = getattr(self.parameters, factor)
            risk_score += weight * (value / self.normal_range(factor))
            
        if self.parameters.diabetes_status:
            risk_score *= 1.3
        if self.parameters.smoking_history_years > 5:
            risk_score *= 1.2
        if self.parameters.family_cardiovascular_disease:
            risk_score *= 1.1
        
        return risk_score

    def normal_range(self, parameter: str) -> float:
        ranges = {
            'heart_rate': 60,
            'age': 45,
            'blood_pressure_systolic': 120,
            'blood_pressure_diastolic': 80,
            'cholesterol_total': 200,
            'hdl_cholesterol': 50,
            'triglycerides': 150,
        }
        return ranges.get(parameter, 1.0)

def run_demo():
    model = CardiacFibrosisModel()
    risk_score = model.calculate_risk_score()
    print(f"Cardiac Fibrosis Risk Score: {risk_score:.2f}")

if __name__ == '__main__':
    run_demo()