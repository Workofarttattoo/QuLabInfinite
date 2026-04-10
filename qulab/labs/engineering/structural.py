"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

STRUCTURAL ENGINEERING LAB
Free gift to the scientific community from QuLabInfinite.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.constants import k, Avogadro, g, c, h, e, pi
from typing import List

@dataclass
class StructuralElement:
    length: float = 0.0
    area: float = 0.0
    youngs_modulus: float = 210e9
    weight_density: float = 7850
    temperature: float = 20.0

@dataclass
class StructuralLoad:
    load_type: str = "uniform"
    magnitude: float = 0.0
    start: float = 0.0
    end: float = 1.0
    distribution: np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=np.float64))

@dataclass
class StructuralBeam:
    elements: List[StructuralElement] = field(default_factory=list)
    loads: List[StructuralLoad] = field(default_factory=list)

    def add_element(self, element):
        self.elements.append(element)

    def add_load(self, load):
        self.loads.append(load)

    def calculate_bending_moment_and_shear_force(self):
        total_length = sum([e.length for e in self.elements])
        distributed_loads = np.zeros(total_length + 1, dtype=np.float64)
        
        for load in self.loads:
            if load.load_type == "uniform":
                start_index = int(load.start / total_length * len(distributed_loads))
                end_index = min(start_index + int((load.end - load.start) / total_length * len(distributed_loads)) + 1, len(distributed_loads))
                distributed_loads[start_index:end_index] += load.magnitude

        shear_force = np.zeros(total_length + 1, dtype=np.float64)
        bending_moment = np.zeros(total_length + 1, dtype=np.float64)

        for i in range(1, len(shear_force)):
            shear_force[i] = shear_force[i-1] - distributed_loads[i]
        
        for i in range(len(bending_moment) - 2, -1, -1):
            bending_moment[i] = bending_moment[i+1] + (shear_force[i+1] + shear_force[i]) / 2 * (i - (i - 1))
        
        return shear_force, bending_moment

    def calculate_stress_and_deflection(self):
        total_length = sum([e.length for e in self.elements])
        distributed_loads = np.zeros(total_length + 1, dtype=np.float64)
        stresses = np.zeros(total_length + 1, dtype=np.float64)
        deflections = np.zeros(total_length + 1, dtype=np.float64)

        for load in self.loads:
            if load.load_type == "uniform":
                start_index = int(load.start / total_length * len(distributed_loads))
                end_index = min(start_index + int((load.end - load.start) / total_length * len(distributed_loads)) + 1, len(distributed_loads))
                distributed_loads[start_index:end_index] += load.magnitude

        for element in self.elements:
            length_factor = np.float64(element.length)
            
            shear_force, bending_moment = self.calculate_bending_moment_and_shear_force()
            stress_at_end_points = (bending_moment[-1] * element.length) / (element.youngs_modulus * element.area)

            deflection = 0
            for i in range(len(bending_moment)):
                deflection += bending_moment[i]**2 / element.youngs_modulus**2 / element.area**3 * length_factor**4
            
            stresses[-1] += stress_at_end_points
            deflections[-1] += deflection

        return stresses, deflections

def run_demo():
    beam = StructuralBeam()
    
    beam.add_element(StructuralElement(length=5.0, area=0.1))
    beam.add_load(StructuralLoad(load_type="uniform", magnitude=20.0, start=0.0, end=4.0))

    shear_force, bending_moment = beam.calculate_bending_moment_and_shear_force()
    
    stresses, deflections = beam.calculate_stress_and_deflection()

    print("Shear Force at each point (kN):")
    for sf in shear_force:
        print(f"{sf:.3f}", end=" ")
        
    print("\nBending Moment at each point (Nm):")
    for bm in bending_moment:
        print(f"{bm:.2f}", end=" ")

    print("\nStress Distribution along the beam (Pa):")
    for stress in stresses:
        print(f"{stress:.3e}", end=" ")

    print("\nDeflection at each point (m):")
    for defl in deflections:
        print(f"{defl:.6f}", end=" ")

if __name__ == '__main__':
    run_demo()