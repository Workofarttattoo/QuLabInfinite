"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

CARDIOVASCULAR PLAQUE FORMATION SIMULATOR
Free gift to the scientific community from QuLabInfinite.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.constants import pi, m_p, e, epsilon_0

@dataclass
class PlaqueFormationParams:
    blood_flow_rate: float = 0.05 # L/min
    vessel_radius: float = 4e-3   # m (average human coronary artery)
    vessel_length: float = 1e-2   # m
    cholesterol_concentration: float = 5e-6  # mol/L
    plasma_protein_binding: float = 0.8     # Fraction of cholesterol bound to proteins
    cell_cholesterol_export_rate: float = 1e-9 # mol/(cell*s)
    cell_density: float = 2e7             # Cells/m^3 (average human coronary artery wall density)
    plaque_growth_time_step: float = 60   # s
    simulation_duration: int = 8760 * 4   # Days (simulating a year)

@dataclass
class SimulationState:
    vessel_radius: float
    cholesterol_concentration_profile: np.ndarray
    blood_flow_rate: float
    plaque_growth: float

def create_cholesterol_concentration_profile(params: PlaqueFormationParams) -> np.ndarray:
    """Create an initial concentration profile of cholesterol in the artery wall."""
    num_points = int(params.vessel_length / (10 * params.vessel_radius))
    distance_from_wall = np.linspace(0, params.vessel_length, num_points)
    concentration_profile = np.zeros(num_points)
    
    # Assuming cholesterol is highest near the endothelium
    for i in range(num_points):
        concentration_profile[i] += params.cholesterol_concentration * (1 - params.plasma_protein_binding)

    return concentration_profile

def update_cholesterol_concentration(state: SimulationState, params: PlaqueFormationParams) -> np.ndarray:
    """Update the cholesterol concentration profile based on cell density and flow rate."""
    # Calculate flow velocity at each point
    wall_shear_stress = 4 * state.blood_flow_rate / (pi * state.vessel_radius ** 2)
    flow_velocity_profile = state.blood_flow_rate / (params.cell_density * pi * state.vessel_radius ** 3)

    # Update cholesterol concentration based on endothelial transport, assuming linear decay with distance
    updated_concentration_profile = np.zeros_like(state.cholesterol_concentration_profile)
    
    for i in range(len(updated_concentration_profile)):
        if i > 0:
            updated_concentration_profile[i] -= params.cell_cholesterol_export_rate * state.plaque_growth / flow_velocity_profile[i]
        
        # Bound cholesterol concentration to non-negative values
        updated_concentration_profile = np.maximum(0, updated_concentration_profile)
    
    return updated_concentration_profile

def simulate_plaque_growth(params: PlaqueFormationParams) -> SimulationState:
    """Simulate the growth of plaques over a given period."""
    state = SimulationState(
        vessel_radius=params.vessel_radius,
        cholesterol_concentration_profile=create_cholesterol_concentration_profile(params),
        blood_flow_rate=params.blood_flow_rate,
        plaque_growth=0.0
    )

    # Main loop for simulation
    t_steps = int(params.simulation_duration * 24 * 3600 / params.plaque_growth_time_step)
    
    for _ in range(t_steps):
        state.cholesterol_concentration_profile = update_EEK(state, params) # Placeholder for EEK calculation
        
        # Update plaque growth based on cholesterol concentration
        if np.any(state.cholesterol_concentration_profile > 0):
            state.plaque_growth += 1e-6 * params.cell_density * params.plaque_growth_time_step / (state.vessel_radius ** 2)
    
    return state

def run_demo():
    params = PlaqueFormationParams()
    final_state = simulate_plaque_growth(params)

    print(f"Final vessel radius after {params.simulation_duration} days: {final_state.vessel_radius}")
    print(f"Plaque growth: {final_state.plaque_growth * 1e3:.2f} mm")

if __name__ == '__main__':
    run_demo()