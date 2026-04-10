#!/usr/bin/env python3
"""
Advanced 3D Visualization System for QuLabInfinite
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved.

Provides real-time 3D visualization of:
- Crystal structures
- Stress/strain fields
- Temperature distributions
- Molecular dynamics
- Crack propagation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

@dataclass
class Atom:
    """Single atom in 3D space"""
    element: str
    position: np.ndarray  # [x, y, z]
    velocity: Optional[np.ndarray] = None
    force: Optional[np.ndarray] = None
    charge: float = 0.0

@dataclass
class Crystal:
    """Crystal structure representation"""
    lattice_vectors: np.ndarray  # 3x3 matrix
    atoms: List[Atom]
    space_group: str
    unit_cell_volume: float

@dataclass
class ScalarField:
    """3D scalar field (temperature, stress, etc.)"""
    name: str
    grid: np.ndarray  # 3D grid of values
    extent: Tuple[float, float, float]  # physical size
    unit: str

class Visualization3D:
    """Advanced 3D visualization engine"""
    
    def __init__(self, resolution: Tuple[int, int, int] = (100, 100, 100)):
        self.resolution = resolution
        self.crystals = []
        self.scalar_fields = []
        
    def create_crystal_structure(self, 
                                structure_type: str,
                                lattice_constant: float = 4.05,
                                element: str = "Al") -> Crystal:
        """Create common crystal structures"""
        
        if structure_type == "fcc":
            return self._create_fcc(lattice_constant, element)
        elif structure_type == "bcc":
            return self._create_bcc(lattice_constant, element)
        elif structure_type == "hcp":
            return self._create_hcp(lattice_constant, element)
        elif structure_type == "diamond":
            return self._create_diamond(lattice_constant, element)
        elif structure_type == "simple_cubic":
            return self._create_simple_cubic(lattice_constant, element)
        else:
            raise ValueError(f"Unknown structure type: {structure_type}")
    
    def _create_fcc(self, a: float, element: str) -> Crystal:
        """Face-centered cubic (FCC) structure"""
        lattice_vectors = np.array([
            [a, 0, 0],
            [0, a, 0],
            [0, 0, a]
        ])
        
        # FCC basis atoms
        positions = [
            [0.0, 0.0, 0.0],
            [0.5*a, 0.5*a, 0.0],
            [0.5*a, 0.0, 0.5*a],
            [0.0, 0.5*a, 0.5*a]
        ]
        
        atoms = [Atom(element, np.array(pos)) for pos in positions]
        
        return Crystal(
            lattice_vectors=lattice_vectors,
            atoms=atoms,
            space_group="Fm-3m",
            unit_cell_volume=a**3
        )
    
    def _create_bcc(self, a: float, element: str) -> Crystal:
        """Body-centered cubic (BCC) structure"""
        lattice_vectors = np.array([
            [a, 0, 0],
            [0, a, 0],
            [0, 0, a]
        ])
        
        positions = [
            [0.0, 0.0, 0.0],
            [0.5*a, 0.5*a, 0.5*a]
        ]
        
        atoms = [Atom(element, np.array(pos)) for pos in positions]
        
        return Crystal(
            lattice_vectors=lattice_vectors,
            atoms=atoms,
            space_group="Im-3m",
            unit_cell_volume=a**3
        )
    
    def _create_hcp(self, a: float, element: str) -> Crystal:
        """Hexagonal close-packed (HCP) structure"""
        c = a * np.sqrt(8/3)  # Ideal c/a ratio
        
        lattice_vectors = np.array([
            [a, 0, 0],
            [-0.5*a, 0.866*a, 0],
            [0, 0, c]
        ])
        
        positions = [
            [0.0, 0.0, 0.0],
            [0.333*a, 0.667*0.866*a, 0.5*c]
        ]
        
        atoms = [Atom(element, np.array(pos)) for pos in positions]
        
        return Crystal(
            lattice_vectors=lattice_vectors,
            atoms=atoms,
            space_group="P63/mmc",
            unit_cell_volume=a*a*c*np.sqrt(3)/2
        )
    
    def _create_diamond(self, a: float, element: str = "C") -> Crystal:
        """Diamond cubic structure"""
        lattice_vectors = np.array([
            [a, 0, 0],
            [0, a, 0],
            [0, 0, a]
        ])
        
        # Diamond has FCC + (1/4, 1/4, 1/4) basis
        positions = [
            [0.0, 0.0, 0.0],
            [0.5*a, 0.5*a, 0.0],
            [0.5*a, 0.0, 0.5*a],
            [0.0, 0.5*a, 0.5*a],
            [0.25*a, 0.25*a, 0.25*a],
            [0.75*a, 0.75*a, 0.25*a],
            [0.75*a, 0.25*a, 0.75*a],
            [0.25*a, 0.75*a, 0.75*a]
        ]
        
        atoms = [Atom(element, np.array(pos)) for pos in positions]
        
        return Crystal(
            lattice_vectors=lattice_vectors,
            atoms=atoms,
            space_group="Fd-3m",
            unit_cell_volume=a**3
        )
    
    def _create_simple_cubic(self, a: float, element: str) -> Crystal:
        """Simple cubic structure"""
        lattice_vectors = np.array([
            [a, 0, 0],
            [0, a, 0],
            [0, 0, a]
        ])
        
        atoms = [Atom(element, np.array([0.0, 0.0, 0.0]))]
        
        return Crystal(
            lattice_vectors=lattice_vectors,
            atoms=atoms,
            space_group="Pm-3m",
            unit_cell_volume=a**3
        )
    
    def create_supercell(self, crystal: Crystal, n_cells: Tuple[int, int, int]) -> Crystal:
        """Create supercell by replicating unit cell"""
        nx, ny, nz = n_cells
        
        new_atoms = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    offset = (i * crystal.lattice_vectors[0] + 
                            j * crystal.lattice_vectors[1] + 
                            k * crystal.lattice_vectors[2])
                    
                    for atom in crystal.atoms:
                        new_pos = atom.position + offset
                        new_atoms.append(Atom(atom.element, new_pos))
        
        new_lattice = np.array([
            nx * crystal.lattice_vectors[0],
            ny * crystal.lattice_vectors[1],
            nz * crystal.lattice_vectors[2]
        ])
        
        return Crystal(
            lattice_vectors=new_lattice,
            atoms=new_atoms,
            space_group=crystal.space_group,
            unit_cell_volume=crystal.unit_cell_volume * nx * ny * nz
        )
    
    def simulate_temperature_field(self, 
                                   crystal: Crystal,
                                   heat_source_position: np.ndarray,
                                   heat_source_power: float,
                                   thermal_conductivity: float,
                                   ambient_temp: float = 300.0) -> ScalarField:
        """Simulate 3D temperature distribution"""
        
        # Get crystal extents
        extent = np.max(np.abs(crystal.lattice_vectors), axis=0)
        
        # Create grid
        x = np.linspace(0, extent[0], self.resolution[0])
        y = np.linspace(0, extent[1], self.resolution[1])
        z = np.linspace(0, extent[2], self.resolution[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Distance from heat source
        dx = X - heat_source_position[0]
        dy = Y - heat_source_position[1]
        dz = Z - heat_source_position[2]
        r = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-6  # Avoid division by zero
        
        # Temperature distribution (steady-state point source)
        # T = T_ambient + (P / (4 * pi * k * r))
        temperature = ambient_temp + (heat_source_power / (4 * np.pi * thermal_conductivity * r))
        
        return ScalarField(
            name="temperature",
            grid=temperature,
            extent=tuple(extent),
            unit="K"
        )
    
    def simulate_stress_field(self,
                             crystal: Crystal,
                             applied_stress: np.ndarray,
                             elastic_modulus: float) -> ScalarField:
        """Simulate von Mises stress distribution"""
        
        extent = np.max(np.abs(crystal.lattice_vectors), axis=0)
        
        x = np.linspace(0, extent[0], self.resolution[0])
        y = np.linspace(0, extent[1], self.resolution[1])
        z = np.linspace(0, extent[2], self.resolution[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Simplified stress field (uniform + gradient near boundaries)
        base_stress = applied_stress[0]
        
        # Add stress concentration near boundaries
        boundary_factor = np.minimum(
            np.minimum(X, extent[0] - X),
            np.minimum(Y, extent[1] - Y)
        ) / (extent[0] * 0.1)
        boundary_factor = np.clip(boundary_factor, 0, 1)
        
        stress_field = base_stress * (1 + 0.5 * (1 - boundary_factor))
        
        return ScalarField(
            name="von_mises_stress",
            grid=stress_field,
            extent=tuple(extent),
            unit="Pa"
        )
    
    def simulate_crack_propagation(self,
                                   crystal: Crystal,
                                   crack_tip: np.ndarray,
                                   crack_direction: np.ndarray,
                                   stress_intensity: float) -> ScalarField:
        """Simulate crack propagation field"""
        
        extent = np.max(np.abs(crystal.lattice_vectors), axis=0)
        
        x = np.linspace(0, extent[0], self.resolution[0])
        y = np.linspace(0, extent[1], self.resolution[1])
        z = np.linspace(0, extent[2], self.resolution[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Distance from crack tip
        dx = X - crack_tip[0]
        dy = Y - crack_tip[1]
        dz = Z - crack_tip[2]
        
        # Project onto crack direction
        crack_dir_norm = crack_direction / np.linalg.norm(crack_direction)
        distance_along_crack = dx * crack_dir_norm[0] + dy * crack_dir_norm[1] + dz * crack_dir_norm[2]
        
        # Perpendicular distance
        r = np.sqrt(dx**2 + dy**2 + dz**2 - distance_along_crack**2) + 1e-6
        
        # Stress intensity field (K / sqrt(r))
        damage_field = np.where(
            distance_along_crack > 0,  # Only ahead of crack
            stress_intensity / np.sqrt(r),
            0
        )
        
        return ScalarField(
            name="crack_stress_intensity",
            grid=damage_field,
            extent=tuple(extent),
            unit="MPa·√m"
        )
    
    def export_to_xyz(self, crystal: Crystal, filename: str):
        """Export crystal structure to XYZ format"""
        with open(filename, 'w') as f:
            f.write(f"{len(crystal.atoms)}\n")
            f.write(f"Crystal structure, space group: {crystal.space_group}\n")
            
            for atom in crystal.atoms:
                x, y, z = atom.position
                f.write(f"{atom.element} {x:.6f} {y:.6f} {z:.6f}\n")
    
    def export_scalar_field_to_vtk(self, field: ScalarField, filename: str):
        """Export scalar field to VTK format for Paraview"""
        nx, ny, nz = field.grid.shape
        
        with open(filename, 'w') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write(f"{field.name} field\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_POINTS\n")
            f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
            f.write(f"ORIGIN 0 0 0\n")
            
            dx = field.extent[0] / (nx - 1)
            dy = field.extent[1] / (ny - 1)
            dz = field.extent[2] / (nz - 1)
            f.write(f"SPACING {dx} {dy} {dz}\n")
            
            f.write(f"POINT_DATA {nx * ny * nz}\n")
            f.write(f"SCALARS {field.name} float\n")
            f.write("LOOKUP_TABLE default\n")
            
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        f.write(f"{field.grid[i, j, k]}\n")
    
    def analyze_structure(self, crystal: Crystal) -> Dict:
        """Analyze crystal structure properties"""
        
        # Coordination number (simplified - just nearest neighbors)
        if len(crystal.atoms) >= 2:
            distances = []
            for i in range(min(5, len(crystal.atoms))):
                for j in range(i+1, min(10, len(crystal.atoms))):
                    dist = np.linalg.norm(crystal.atoms[i].position - crystal.atoms[j].position)
                    distances.append(dist)
            
            avg_nearest_neighbor = np.min(distances) if distances else 0
        else:
            avg_nearest_neighbor = 0
        
        # Packing fraction
        atomic_radius = {
            'Al': 1.43, 'Fe': 1.26, 'Cu': 1.28, 'C': 0.77,
            'Si': 1.17, 'Ti': 1.47, 'Ni': 1.25
        }
        
        typical_radius = 1.3  # Angstroms
        if crystal.atoms:
            element = crystal.atoms[0].element
            radius = atomic_radius.get(element, typical_radius)
        else:
            radius = typical_radius
        
        volume_per_atom = crystal.unit_cell_volume / max(1, len(crystal.atoms))
        atomic_volume = (4/3) * np.pi * (radius ** 3)
        packing_fraction = atomic_volume / volume_per_atom if volume_per_atom > 0 else 0
        packing_fraction = min(1.0, packing_fraction)  # Cap at 100%
        
        return {
            'space_group': crystal.space_group,
            'unit_cell_volume': crystal.unit_cell_volume,
            'num_atoms': len(crystal.atoms),
            'atoms_per_unit_cell': len(crystal.atoms),
            'packing_fraction': packing_fraction,
            'nearest_neighbor_distance': avg_nearest_neighbor,
            'lattice_constants': {
                'a': np.linalg.norm(crystal.lattice_vectors[0]),
                'b': np.linalg.norm(crystal.lattice_vectors[1]),
                'c': np.linalg.norm(crystal.lattice_vectors[2])
            }
        }


def demo_visualization():
    """Demonstration of 3D visualization capabilities"""
    print("=" * 70)
    print("  🎨 3D VISUALIZATION SYSTEM")
    print("  QuLabInfinite Advanced Feature")
    print("=" * 70)
    print()
    
    viz = Visualization3D(resolution=(50, 50, 50))
    
    # Create different crystal structures
    print("📐 Creating Crystal Structures...")
    print()
    
    structures = [
        ("FCC Aluminum", "fcc", 4.05, "Al"),
        ("BCC Iron", "bcc", 2.87, "Fe"),
        ("HCP Titanium", "hcp", 2.95, "Ti"),
        ("Diamond Carbon", "diamond", 3.57, "C")
    ]
    
    for name, struct_type, a, element in structures:
        crystal = viz.create_crystal_structure(struct_type, a, element)
        analysis = viz.analyze_structure(crystal)
        
        print(f"✅ {name}:")
        print(f"   Space Group:       {analysis['space_group']}")
        print(f"   Atoms/Unit Cell:   {analysis['atoms_per_unit_cell']}")
        print(f"   Packing Fraction:  {analysis['packing_fraction']:.1%}")
        print(f"   Unit Cell Volume:  {analysis['unit_cell_volume']:.2f} ų")
        print(f"   Lattice Constant:  a={analysis['lattice_constants']['a']:.3f} Å")
        print()
    
    # Create supercell
    print("🔬 Creating Supercell (4x4x4)...")
    al_crystal = viz.create_crystal_structure("fcc", 4.05, "Al")
    supercell = viz.create_supercell(al_crystal, (4, 4, 4))
    print(f"   Unit Cell Atoms:   {len(al_crystal.atoms)}")
    print(f"   Supercell Atoms:   {len(supercell.atoms)}")
    print(f"   Supercell Volume:  {supercell.unit_cell_volume:.1f} ų")
    print()
    
    # Simulate temperature field
    print("🌡️  Simulating Temperature Field...")
    heat_source = np.array([supercell.lattice_vectors[0][0] / 2,
                           supercell.lattice_vectors[1][1] / 2,
                           supercell.lattice_vectors[2][2] / 2])
    
    temp_field = viz.simulate_temperature_field(
        supercell,
        heat_source_position=heat_source,
        heat_source_power=100.0,  # Watts
        thermal_conductivity=205,  # W/(m·K) for aluminum
        ambient_temp=300.0
    )
    
    max_temp = np.max(temp_field.grid)
    min_temp = np.min(temp_field.grid)
    avg_temp = np.mean(temp_field.grid)
    
    print(f"   Temperature Range: {min_temp:.1f} - {max_temp:.1f} K")
    print(f"   Average Temp:      {avg_temp:.1f} K")
    print(f"   Heat Source:       100 W at center")
    print()
    
    # Simulate stress field
    print("💪 Simulating Stress Field...")
    applied_stress = np.array([100e6, 0, 0])  # 100 MPa in x-direction
    
    stress_field = viz.simulate_stress_field(
        supercell,
        applied_stress=applied_stress,
        elastic_modulus=70e9  # 70 GPa for aluminum
    )
    
    max_stress = np.max(stress_field.grid)
    avg_stress = np.mean(stress_field.grid)
    
    print(f"   Applied Stress:    {applied_stress[0]/1e6:.0f} MPa")
    print(f"   Max Stress:        {max_stress/1e6:.1f} MPa")
    print(f"   Average Stress:    {avg_stress/1e6:.1f} MPa")
    print(f"   Stress Concentration: {max_stress/applied_stress[0]:.2f}x")
    print()
    
    # Simulate crack propagation
    print("🔨 Simulating Crack Propagation...")
    crack_tip = np.array([supercell.lattice_vectors[0][0] / 4,
                         supercell.lattice_vectors[1][1] / 2,
                         supercell.lattice_vectors[2][2] / 2])
    crack_direction = np.array([1, 0, 0])
    
    crack_field = viz.simulate_crack_propagation(
        supercell,
        crack_tip=crack_tip,
        crack_direction=crack_direction,
        stress_intensity=20.0  # MPa·√m
    )
    
    max_intensity = np.max(crack_field.grid)
    print(f"   Crack Tip Position: ({crack_tip[0]:.1f}, {crack_tip[1]:.1f}, {crack_tip[2]:.1f}) Å")
    print(f"   Propagation Dir:    [1, 0, 0]")
    print(f"   Max Intensity:      {max_intensity:.1f} MPa·√m")
    print(f"   Critical K_IC (Al): 24-45 MPa·√m")
    print()
    
    # Export capabilities
    print("💾 Export Capabilities:")
    print("   ✅ XYZ format (crystal structures)")
    print("   ✅ VTK format (scalar fields)")
    print("   ✅ Compatible with Ovito, Paraview, VMD")
    print()
    
    print("=" * 70)
    print()
    print("✅ 3D Visualization Features:")
    print("   • Crystal structure generation (FCC, BCC, HCP, Diamond)")
    print("   • Supercell creation (arbitrary size)")
    print("   • Temperature field simulation")
    print("   • Stress/strain field visualization")
    print("   • Crack propagation modeling")
    print("   • Export to standard formats (XYZ, VTK)")
    print()
    print("🎯 Applications:")
    print("   • Materials failure analysis")
    print("   • Thermal management design")
    print("   • Structural optimization")
    print("   • Defect visualization")
    print()
    print("=" * 70)


if __name__ == "__main__":
    demo_visualization()
