"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
Astrobiology Lab - Search for extraterrestrial life and habitability analysis
"""

import numpy as np
from typing import Dict, List

class AstrobiologyLab:
    """Analyze conditions for life beyond Earth"""
    
    def __init__(self):
        self.drake_factors = {
            'R_star': 1.0,  # star formation rate
            'f_p': 1.0,      # fraction with planets
            'n_e': 0.4,      # habitable planets per system
            'f_l': 1.0,      # fraction developing life
            'f_i': 0.01,     # fraction intelligent
            'f_c': 0.1,      # fraction communicating
            'L': 10000       # civilization lifetime
        }
        
    def calculate_habitability_index(self, planet_params: Dict) -> float:
        """Calculate planetary habitability index"""
        # Earth similarity index components
        radius_sim = np.exp(-np.abs(planet_params.get('radius', 1) - 1) / 0.5)
        density_sim = np.exp(-np.abs(planet_params.get('density', 5.5) - 5.5) / 2)
        temp_sim = np.exp(-np.abs(planet_params.get('temperature', 288) - 288) / 50)
        escape_vel_sim = np.exp(-np.abs(planet_params.get('escape_velocity', 11.2) - 11.2) / 5)
        
        # Weighted geometric mean
        esi = (radius_sim**0.57 * density_sim**1.07 * escape_vel_sim**0.70 * temp_sim**5.58)**(1/7.92)
        
        return esi
        
    def analyze_biosignatures(self, spectrum: np.ndarray, wavelengths: np.ndarray) -> Dict:
        """Detect potential biosignatures in atmospheric spectrum"""
        biosignatures = {
            'oxygen': 760,      # O2 A-band
            'methane': 3300,    # CH4
            'water': 940,       # H2O
            'ozone': 9600,      # O3
            'phosphine': 2960   # PH3
        }
        
        detected = {}
        for molecule, wavelength in biosignatures.items():
            idx = np.argmin(np.abs(wavelengths - wavelength))
            if idx < len(spectrum):
                absorption = 1 - spectrum[idx]
                if absorption > 0.1:
                    detected[molecule] = absorption
                    
        return detected
        
    def estimate_drake_equation(self, custom_params: Dict = None) -> float:
        """Calculate Drake equation for number of communicating civilizations"""
        params = self.drake_factors.copy()
        if custom_params:
            params.update(custom_params)
            
        N = 1
        for factor in params.values():
            N *= factor
            
        return N