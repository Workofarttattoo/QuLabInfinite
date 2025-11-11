"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

DRUG INTERACTION SIMULATOR
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.constants import pi, Avogadro

@dataclass
class Drug:
    name: str
    molecular_weight: float  # in g/mol
    concentration: float     # in mol/L
    volume: float            # in L
    interactions: dict       # dict of interaction coefficients with other drugs


@dataclass
class Environment:
    temperature: float         # in Kelvin
    ph_value: float            # pH value of the environment
    solvent_volume: float      # total volume of solvent in liters

@dataclass
class SimulationParameters:
    time_step: float           # time step for simulation (in seconds)
    total_time: float          # total simulation time (in seconds)

def calculate_drug_interaction(drug1, drug2):
    interaction_coefficient = np.random.uniform(0.1, 1.5)  # Simulate a range of interaction strengths
    return interaction_coefficient

@dataclass
class DrugInteractionSimulator:
    environment: Environment
    drugs: list[Drug]
    parameters: SimulationParameters
    
    def __post_init__(self):
        self.interaction_matrix = np.zeros((len(self.drugs), len(self.drugs)))
        
        # Initialize the interaction matrix with real-world interaction coefficients
        for i in range(len(self.drugs)):
            for j in range(i, len(self.drugs)):
                if i == j:
                    continue  # No self-interaction
                interaction_coeff = calculate_drug_interaction(self.drugs[i], self.drugs[j])
                self.interaction_matrix[i][j] = interaction_coeff
                self.interaction_matrix[j][i] = interaction_coeff

    def simulate(self):
        time_steps = int(self.parameters.total_time / self.parameters.time_step)
        drug_concentrations_over_time = np.zeros((time_steps, len(self.drugs)))
        
        for t in range(time_steps):
            current_drug_concentrations = [drug.concentration for drug in self.drugs]
            drug_concentrations_over_time[t] = current_drug_concentrations
            
            # Update concentrations based on interactions
            new_concentrations = np.zeros(len(self.drugs))
            for i, drug_i in enumerate(self.drugs):
                new_concentrations[i] += self.environment.solvent_volume * (drug_i.concentration - 
                    np.dot(np.multiply(current_drug_concentrations, self.interaction_matrix[:, i]), current_drug_concentrations) / Avogadro)
            
            # Update the drug concentrations for next iteration
            for i in range(len(self.drugs)):
                self.drugs[i].concentration = new_concentrations[i]
        
        return drug_concentrations_over_time

def run_demo():
    environment = Environment(
        temperature=298.15,  # Room temperature (K)
        ph_value=7.4,        # Typical physiological pH
        solvent_volume=10     # Volume of solvent in liters
    )

    drugs = [
        Drug(name="Drug A", molecular_weight=356.4, concentration=1e-2, volume=environment.solvent_volume),
        Drug(name="Drug B", molecular_weight=478.6, concentration=1e-3, volume=environment.solvent_volume)
    ]
    
    parameters = SimulationParameters(
        time_step=0.05,  # Time step for simulation (s)
        total_time=60     # Total simulation time (s)
    )

    simulator = DrugInteractionSimulator(environment, drugs, parameters)
    concentrations_over_time = simulator.simulate()
    
    print("Concentrations over time:")
    print(concentrations_over_time)

if __name__ == '__main__':
    run_demo()