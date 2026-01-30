"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

DEVELOPMENTAL BIOLOGY LAB - Production Ready
Advanced morphogenesis, pattern formation, and organogenesis modeling.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable
from scipy import ndimage, integrate, optimize
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from matplotlib import cm

@dataclass
class DevelopmentalBiologyLab:
    """Production-ready developmental biology simulation laboratory."""

    # Physical constants
    BOLTZMANN_CONSTANT: float = 1.38e-23  # J/K
    AVOGADRO: float = 6.022e23  # molecules/mol
    CELL_DIAMETER: float = 10.0  # micrometers
    TEMPERATURE: float = 310.15  # Kelvin (37°C)

    # Simulation parameters
    grid_size: Tuple[int, int, int] = (100, 100, 20)  # 3D tissue grid
    time_steps: int = 1000
    dt: float = 0.01  # Time step in hours
    diffusion_coefficient: float = 10.0  # μm²/s

    # Cell type definitions
    CELL_TYPES: Dict[str, int] = field(default_factory=lambda: {
        'stem': 0,
        'progenitor': 1,
        'neural': 2,
        'epithelial': 3,
        'mesenchymal': 4,
        'endothelial': 5,
        'muscle': 6,
        'bone': 7
    })

    def __post_init__(self):
        """Initialize tissue grid and morphogen fields."""
        self.tissue = np.zeros(self.grid_size, dtype=int)
        self.morphogen_fields = {}
        self.cell_positions = []
        self.lineage_tree = {}
        self.gene_expression = {}

    def initialize_embryo(self, initial_cells: int = 100) -> np.ndarray:
        """Initialize embryonic tissue with stem cells."""
        # Place cells in central sphere
        center = np.array(self.grid_size) // 2
        radius = min(self.grid_size) // 4

        positions = []
        for _ in range(initial_cells):
            # Random position within sphere
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            r = np.random.uniform(0, radius)

            x = int(center[0] + r * np.sin(phi) * np.cos(theta))
            y = int(center[1] + r * np.sin(phi) * np.sin(theta))
            z = int(center[2] + r * np.cos(phi))

            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1] and 0 <= z < self.grid_size[2]:
                self.tissue[x, y, z] = self.CELL_TYPES['stem']
                positions.append([x, y, z])

        self.cell_positions = positions
        return self.tissue

    def create_morphogen_gradient(self, source_position: Tuple[int, int, int],
                                 morphogen_name: str, production_rate: float = 1.0,
                                 decay_rate: float = 0.1) -> np.ndarray:
        """Create morphogen gradient from source."""
        field = np.zeros(self.grid_size)

        # Distance from source
        x, y, z = np.ogrid[:self.grid_size[0], :self.grid_size[1], :self.grid_size[2]]
        dist = np.sqrt((x - source_position[0])**2 +
                      (y - source_position[1])**2 +
                      (z - source_position[2])**2)

        # Exponential decay with distance
        field = production_rate * np.exp(-decay_rate * dist)

        # Add noise for biological variability
        field += np.random.normal(0, 0.01, field.shape)
        field = np.maximum(field, 0)  # Non-negative concentrations

        self.morphogen_fields[morphogen_name] = field
        return field

    def reaction_diffusion_step(self, u: np.ndarray, v: np.ndarray,
                               Du: float = 0.16, Dv: float = 0.08,
                               F: float = 0.035, k: float = 0.065) -> Tuple[np.ndarray, np.ndarray]:
        """Gray-Scott reaction-diffusion system for pattern formation."""
        # Laplacian using convolution
        kernel = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                          [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                          [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]) / 6.0

        Lu = ndimage.convolve(u, kernel, mode='wrap')
        Lv = ndimage.convolve(v, kernel, mode='wrap')

        # Reaction terms
        uvv = u * v * v

        # Update equations
        du = Du * Lu - uvv + F * (1 - u)
        dv = Dv * Lv + uvv - (F + k) * v

        u_new = u + self.dt * du
        v_new = v + self.dt * dv

        # Ensure values stay in valid range
        u_new = np.clip(u_new, 0, 1)
        v_new = np.clip(v_new, 0, 1)

        return u_new, v_new

    def turing_pattern_formation(self, iterations: int = 500) -> np.ndarray:
        """Generate Turing patterns for skin/fur patterns."""
        # Initialize with small random perturbations
        u = np.ones(self.grid_size[:2]) + np.random.normal(0, 0.01, self.grid_size[:2])
        v = np.zeros(self.grid_size[:2]) + np.random.normal(0, 0.01, self.grid_size[:2])

        # Add initial seeds
        center = (self.grid_size[0] // 2, self.grid_size[1] // 2)
        radius = 5
        y, x = np.ogrid[:self.grid_size[0], :self.grid_size[1]]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        v[mask] = 0.25

        # Run reaction-diffusion
        for _ in range(iterations):
            u, v = self.reaction_diffusion_step(u, v)

        return v

    def cell_differentiation(self, cell_position: Tuple[int, int, int],
                           morphogen_concentrations: Dict[str, float]) -> str:
        """Determine cell fate based on morphogen concentrations."""
        # Simplified differentiation rules based on morphogen thresholds
        wnt = morphogen_concentrations.get('wnt', 0)
        bmp = morphogen_concentrations.get('bmp', 0)
        shh = morphogen_concentrations.get('shh', 0)
        fgf = morphogen_concentrations.get('fgf', 0)

        # Neural differentiation (high SHH, low BMP)
        if shh > 0.7 and bmp < 0.3:
            return 'neural'

        # Epithelial (high WNT)
        elif wnt > 0.6:
            return 'epithelial'

        # Mesenchymal (high FGF)
        elif fgf > 0.6:
            return 'mesenchymal'

        # Muscle (medium levels of multiple factors)
        elif wnt > 0.4 and fgf > 0.4:
            return 'muscle'

        # Bone (high BMP)
        elif bmp > 0.7:
            return 'bone'

        # Endothelial (specific combination)
        elif fgf > 0.5 and wnt < 0.3:
            return 'endothelial'

        # Remain progenitor
        else:
            return 'progenitor'

    def simulate_gastrulation(self) -> np.ndarray:
        """Simulate gastrulation - formation of three germ layers."""
        # Create primitive streak
        streak_position = (self.grid_size[0] // 2, self.grid_size[1] // 2, 0)

        # Morphogen gradients for germ layer specification
        nodal = self.create_morphogen_gradient(streak_position, 'nodal', 1.0, 0.05)
        bmp = self.create_morphogen_gradient(
            (self.grid_size[0] // 4, self.grid_size[1] // 2, 0), 'bmp', 0.8, 0.08
        )

        # Assign germ layers based on morphogen levels
        germ_layers = np.zeros(self.grid_size, dtype=int)

        # Ectoderm (outer layer, low nodal)
        germ_layers[nodal < 0.3] = 1

        # Mesoderm (middle layer, medium nodal)
        germ_layers[(nodal >= 0.3) & (nodal < 0.7)] = 2

        # Endoderm (inner layer, high nodal)
        germ_layers[nodal >= 0.7] = 3

        return germ_layers

    def simulate_neurulation(self, ectoderm_mask: np.ndarray) -> np.ndarray:
        """Simulate neural tube formation from ectoderm."""
        neural_plate = np.zeros_like(ectoderm_mask)

        # Define neural plate region (dorsal ectoderm)
        midline_x = self.grid_size[0] // 2
        width = 10

        for z in range(self.grid_size[2]):
            neural_plate[midline_x-width:midline_x+width, :, z] = ectoderm_mask[midline_x-width:midline_x+width, :, z]

        # Simulate folding of neural plate
        neural_tube = np.zeros_like(neural_plate)
        fold_center = self.grid_size[1] // 2

        for x in range(midline_x - width, midline_x + width):
            for y in range(self.grid_size[1]):
                for z in range(self.grid_size[2]):
                    if neural_plate[x, y, z] > 0:
                        # Calculate new position after folding
                        dist_from_midline = abs(x - midline_x)
                        fold_angle = np.pi * (1 - dist_from_midline / width)
                        new_z = int(z + 5 * np.sin(fold_angle))

                        if new_z < self.grid_size[2]:
                            neural_tube[x, y, new_z] = self.CELL_TYPES['neural']

        return neural_tube

    def simulate_somitogenesis(self, num_somites: int = 30) -> List[np.ndarray]:
        """Simulate somite formation (segmentation clock)."""
        somites = []

        # Clock and wavefront model
        clock_period = 90  # minutes
        wavefront_speed = 0.1  # somites per hour

        # Position along anterior-posterior axis
        for i in range(num_somites):
            somite = np.zeros(self.grid_size)

            # Calculate somite position
            anterior_pos = int(self.grid_size[0] * 0.3 + i * self.grid_size[0] * 0.02)
            width = 5
            length = 15

            # Place somite
            if anterior_pos + width < self.grid_size[0]:
                midline_y = self.grid_size[1] // 2

                # Bilateral somites
                for side in [-1, 1]:
                    y_pos = midline_y + side * 10
                    somite[anterior_pos:anterior_pos+width,
                          y_pos-length//2:y_pos+length//2,
                          self.grid_size[2]//2-2:self.grid_size[2]//2+2] = self.CELL_TYPES['mesenchymal']

            somites.append(somite)

        return somites

    def simulate_angiogenesis(self, vegf_source: Tuple[int, int, int],
                            iterations: int = 100) -> np.ndarray:
        """Simulate blood vessel formation through sprouting angiogenesis."""
        vessels = np.zeros(self.grid_size, dtype=bool)

        # Create VEGF gradient
        vegf = self.create_morphogen_gradient(vegf_source, 'vegf', 1.0, 0.02)

        # Initialize tip cell
        tip_positions = [list(vegf_source)]

        for _ in range(iterations):
            new_tips = []

            for tip in tip_positions:
                x, y, z = tip

                # Sample VEGF gradient
                if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1] and 0 <= z < self.grid_size[2]:
                    local_vegf = vegf[x, y, z]

                    # Probability of branching
                    if np.random.random() < 0.1 * local_vegf:
                        # Branch
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            new_x, new_y = x + dx * 2, y + dy * 2
                            if 0 <= new_x < self.grid_size[0] and 0 <= new_y < self.grid_size[1]:
                                new_tips.append([new_x, new_y, z])

                    # Chemotaxis towards VEGF gradient
                    gradient = np.zeros(3)
                    for i, (dx, dy, dz) in enumerate([(-1, 0, 0), (1, 0, 0),
                                                       (0, -1, 0), (0, 1, 0),
                                                       (0, 0, -1), (0, 0, 1)]):
                        new_x, new_y, new_z = x + dx, y + dy, z + dz
                        if (0 <= new_x < self.grid_size[0] and
                            0 <= new_y < self.grid_size[1] and
                            0 <= new_z < self.grid_size[2]):
                            gradient[i % 3] += vegf[new_x, new_y, new_z] - vegf[x, y, z]

                    # Move tip cell
                    if np.linalg.norm(gradient) > 0:
                        gradient = gradient / np.linalg.norm(gradient)
                        new_pos = [
                            int(x + np.sign(gradient[0])),
                            int(y + np.sign(gradient[1])),
                            int(z + np.sign(gradient[2]))
                        ]

                        if (0 <= new_pos[0] < self.grid_size[0] and
                            0 <= new_pos[1] < self.grid_size[1] and
                            0 <= new_pos[2] < self.grid_size[2]):
                            vessels[new_pos[0], new_pos[1], new_pos[2]] = True
                            new_tips.append(new_pos)

            tip_positions = new_tips[:10]  # Limit number of active tips

        return vessels

    def simulate_organogenesis(self, organ_type: str = 'kidney') -> np.ndarray:
        """Simulate organ development."""
        organ = np.zeros(self.grid_size)

        if organ_type == 'kidney':
            # Ureteric bud branching
            bud_position = (self.grid_size[0] // 2, self.grid_size[1] // 2, self.grid_size[2] // 2)

            # GDNF gradient for branching
            gdnf = self.create_morphogen_gradient(bud_position, 'gdnf', 1.0, 0.05)

            # Branching morphogenesis
            branches = self._branching_morphogenesis(bud_position, gdnf, iterations=50)

            # Create nephrons around branches
            for branch_pos in branches:
                x, y, z = branch_pos
                # Glomerulus
                organ[x-1:x+2, y-1:y+2, z-1:z+2] = self.CELL_TYPES['epithelial']

        elif organ_type == 'lung':
            # Branching airways
            trachea_position = (self.grid_size[0] // 2, self.grid_size[1] // 2, 0)

            # FGF10 gradient for branching
            fgf10 = self.create_morphogen_gradient(trachea_position, 'fgf10', 1.0, 0.03)

            branches = self._branching_morphogenesis(trachea_position, fgf10, iterations=100)

            for branch_pos in branches:
                x, y, z = branch_pos
                organ[x, y, z] = self.CELL_TYPES['epithelial']

        elif organ_type == 'heart':
            # Heart tube formation and looping
            center = np.array(self.grid_size) // 2

            # Create heart tube
            for t in np.linspace(0, 2 * np.pi, 100):
                # Helical structure for looped heart
                x = int(center[0] + 10 * np.cos(t))
                y = int(center[1] + 10 * np.sin(t))
                z = int(center[2] + 5 * t / (2 * np.pi))

                if z < self.grid_size[2]:
                    organ[x-2:x+2, y-2:y+2, z] = self.CELL_TYPES['muscle']

        return organ

    def _branching_morphogenesis(self, start_pos: Tuple[int, int, int],
                                gradient_field: np.ndarray,
                                iterations: int = 50) -> List[Tuple[int, int, int]]:
        """Simulate branching morphogenesis."""
        branches = [start_pos]
        tips = [start_pos]

        for _ in range(iterations):
            new_tips = []

            for tip in tips:
                x, y, z = tip

                if (0 <= x < self.grid_size[0] and
                    0 <= y < self.grid_size[1] and
                    0 <= z < self.grid_size[2]):

                    # Branching probability based on gradient
                    if np.random.random() < 0.2 * gradient_field[x, y, z]:
                        # Create two branches
                        angle1 = np.random.uniform(0, 2 * np.pi)
                        angle2 = angle1 + np.pi / 3

                        for angle in [angle1, angle2]:
                            new_x = int(x + 3 * np.cos(angle))
                            new_y = int(y + 3 * np.sin(angle))
                            new_z = z + np.random.randint(-1, 2)

                            if (0 <= new_x < self.grid_size[0] and
                                0 <= new_y < self.grid_size[1] and
                                0 <= new_z < self.grid_size[2]):
                                new_pos = (new_x, new_y, new_z)
                                branches.append(new_pos)
                                new_tips.append(new_pos)
                    else:
                        # Extend current branch
                        dx = np.random.randint(-1, 2)
                        dy = np.random.randint(-1, 2)
                        dz = np.random.randint(0, 2)

                        new_x, new_y, new_z = x + dx, y + dy, z + dz

                        if (0 <= new_x < self.grid_size[0] and
                            0 <= new_y < self.grid_size[1] and
                            0 <= new_z < self.grid_size[2]):
                            new_pos = (new_x, new_y, new_z)
                            branches.append(new_pos)
                            new_tips.append(new_pos)

            tips = new_tips[:20]  # Limit active tips

        return branches

    def calculate_tissue_mechanics(self, cell_positions: np.ndarray,
                                  k_spring: float = 1.0,
                                  rest_length: float = 10.0) -> np.ndarray:
        """Calculate mechanical forces between cells."""
        n_cells = len(cell_positions)
        forces = np.zeros((n_cells, 3))

        # Calculate pairwise forces
        for i in range(n_cells):
            for j in range(i + 1, n_cells):
                # Vector from i to j
                r_ij = cell_positions[j] - cell_positions[i]
                distance = np.linalg.norm(r_ij)

                if distance > 0 and distance < 3 * rest_length:
                    # Spring force
                    force_magnitude = k_spring * (distance - rest_length)
                    force_direction = r_ij / distance

                    # Apply forces
                    forces[i] += force_magnitude * force_direction
                    forces[j] -= force_magnitude * force_direction

        return forces

    def simulate_morphogen_dynamics(self, production_sites: List[Tuple[int, int, int]],
                                   decay_rate: float = 0.1) -> np.ndarray:
        """Simulate morphogen diffusion and decay over time."""
        concentration = np.zeros(self.grid_size)

        # Add production at sources
        for site in production_sites:
            concentration[site] = 1.0

        # Diffusion parameters
        dx = 1.0  # Grid spacing in micrometers

        for _ in range(100):  # Time steps
            # Diffusion (using finite difference)
            laplacian = ndimage.laplace(concentration) / (dx * dx)
            concentration += self.dt * (self.diffusion_coefficient * laplacian - decay_rate * concentration)

            # Maintain production at sources
            for site in production_sites:
                concentration[site] = 1.0

            # Non-negative constraint
            concentration = np.maximum(concentration, 0)

        return concentration

    def track_cell_lineage(self, parent_id: int, daughter_ids: Tuple[int, int],
                          division_time: float) -> None:
        """Track cell division and lineage."""
        if parent_id not in self.lineage_tree:
            self.lineage_tree[parent_id] = {
                'parent': None,
                'daughters': [],
                'birth_time': 0,
                'division_time': None,
                'cell_type': 'stem'
            }

        self.lineage_tree[parent_id]['daughters'] = daughter_ids
        self.lineage_tree[parent_id]['division_time'] = division_time

        for daughter_id in daughter_ids:
            self.lineage_tree[daughter_id] = {
                'parent': parent_id,
                'daughters': [],
                'birth_time': division_time,
                'division_time': None,
                'cell_type': 'progenitor'
            }

    def gene_regulatory_network(self, current_expression: Dict[str, float],
                               signals: Dict[str, float]) -> Dict[str, float]:
        """Simulate gene regulatory network dynamics."""
        new_expression = current_expression.copy()

        # Simplified regulatory logic
        # Master regulators
        if signals.get('wnt', 0) > 0.5:
            new_expression['sox2'] = min(1.0, current_expression.get('sox2', 0) + 0.1)
            new_expression['nanog'] = min(1.0, current_expression.get('nanog', 0) + 0.1)

        if signals.get('bmp', 0) > 0.5:
            new_expression['gata6'] = min(1.0, current_expression.get('gata6', 0) + 0.1)
            new_expression['sox2'] = max(0.0, current_expression.get('sox2', 0) - 0.1)

        # Neural genes
        if new_expression.get('sox2', 0) > 0.7:
            new_expression['pax6'] = min(1.0, current_expression.get('pax6', 0) + 0.15)
            new_expression['neurog1'] = min(1.0, current_expression.get('neurog1', 0) + 0.1)

        # Muscle genes
        if signals.get('fgf', 0) > 0.5 and signals.get('wnt', 0) > 0.3:
            new_expression['myod'] = min(1.0, current_expression.get('myod', 0) + 0.1)
            new_expression['myf5'] = min(1.0, current_expression.get('myf5', 0) + 0.1)

        # Decay
        for gene in new_expression:
            new_expression[gene] *= 0.95  # 5% decay per time step

        return new_expression

    def run_comprehensive_simulation(self) -> Dict:
        """Run complete developmental biology simulation."""
        results = {}

        print("Initializing embryo...")
        self.initialize_embryo(100)
        results['initial_cells'] = len(self.cell_positions)

        print("Simulating gastrulation...")
        germ_layers = self.simulate_gastrulation()
        results['germ_layers_formed'] = len(np.unique(germ_layers))

        print("Simulating neurulation...")
        ectoderm = (germ_layers == 1)
        neural_tube = self.simulate_neurulation(ectoderm)
        results['neural_cells'] = np.sum(neural_tube == self.CELL_TYPES['neural'])

        print("Simulating somitogenesis...")
        somites = self.simulate_somitogenesis(20)
        results['somites_formed'] = len(somites)

        print("Simulating pattern formation...")
        pattern = self.turing_pattern_formation(iterations=200)
        results['pattern_complexity'] = np.std(pattern)

        print("Simulating organogenesis...")
        kidney = self.simulate_organogenesis('kidney')
        heart = self.simulate_organogenesis('heart')
        results['kidney_cells'] = np.sum(kidney > 0)
        results['heart_cells'] = np.sum(heart > 0)

        print("Simulating angiogenesis...")
        vessels = self.simulate_angiogenesis(
            (self.grid_size[0]//2, self.grid_size[1]//2, self.grid_size[2]//2),
            iterations=50
        )
        results['vessel_length'] = np.sum(vessels)

        return results


def run_demo():
    """Demonstrate comprehensive developmental biology capabilities."""
    print("=" * 80)
    print("DEVELOPMENTAL BIOLOGY LAB - Production Demo")
    print("=" * 80)

    lab = DevelopmentalBiologyLab(grid_size=(50, 50, 10))

    print("\n1. EMBRYO INITIALIZATION")
    print("-" * 40)
    tissue = lab.initialize_embryo(50)
    print(f"Initial embryo size: {np.sum(tissue > 0)} cells")
    print(f"Tissue dimensions: {tissue.shape}")

    print("\n2. MORPHOGEN GRADIENTS")
    print("-" * 40)
    source = (25, 25, 5)
    wnt_gradient = lab.create_morphogen_gradient(source, 'wnt', 1.0, 0.1)
    print(f"WNT gradient created")
    print(f"Max concentration: {np.max(wnt_gradient):.4f}")
    print(f"Min concentration: {np.min(wnt_gradient):.4f}")

    print("\n3. CELL DIFFERENTIATION")
    print("-" * 40)
    test_position = (25, 25, 5)
    morphogens = {
        'wnt': 0.7,
        'bmp': 0.2,
        'shh': 0.8,
        'fgf': 0.3
    }
    cell_fate = lab.cell_differentiation(test_position, morphogens)
    print(f"Cell fate decision: {cell_fate}")
    print(f"Based on morphogen levels: {morphogens}")

    print("\n4. PATTERN FORMATION (Turing)")
    print("-" * 40)
    print("Generating Turing pattern...")
    pattern = lab.turing_pattern_formation(iterations=100)
    print(f"Pattern generated: {pattern.shape}")
    print(f"Pattern variance: {np.var(pattern):.6f}")

    print("\n5. GASTRULATION")
    print("-" * 40)
    germ_layers = lab.simulate_gastrulation()
    unique_layers = np.unique(germ_layers)
    print(f"Germ layers formed: {len(unique_layers)}")
    for layer in unique_layers:
        count = np.sum(germ_layers == layer)
        if count > 0:
            layer_name = ['None', 'Ectoderm', 'Mesoderm', 'Endoderm'][min(layer, 3)]
            print(f"  {layer_name}: {count} cells")

    print("\n6. SOMITOGENESIS")
    print("-" * 40)
    somites = lab.simulate_somitogenesis(10)
    print(f"Somites generated: {len(somites)}")
    total_somite_cells = sum(np.sum(s > 0) for s in somites)
    print(f"Total somite cells: {total_somite_cells}")

    print("\n7. ORGANOGENESIS")
    print("-" * 40)
    kidney = lab.simulate_organogenesis('kidney')
    lung = lab.simulate_organogenesis('lung')
    heart = lab.simulate_organogenesis('heart')

    print(f"Kidney development: {np.sum(kidney > 0)} cells")
    print(f"Lung development: {np.sum(lung > 0)} cells")
    print(f"Heart development: {np.sum(heart > 0)} cells")

    print("\n8. ANGIOGENESIS")
    print("-" * 40)
    vegf_source = (25, 25, 5)
    vessels = lab.simulate_angiogenesis(vegf_source, iterations=30)
    print(f"Blood vessels formed: {np.sum(vessels)} voxels")

    print("\n9. CELL LINEAGE TRACKING")
    print("-" * 40)
    lab.track_cell_lineage(parent_id=1, daughter_ids=(2, 3), division_time=10.5)
    lab.track_cell_lineage(parent_id=2, daughter_ids=(4, 5), division_time=21.0)
    print(f"Lineage tree nodes: {len(lab.lineage_tree)}")
    for cell_id, info in list(lab.lineage_tree.items())[:3]:
        print(f"  Cell {cell_id}: parent={info['parent']}, daughters={info['daughters']}")

    print("\n10. GENE REGULATORY NETWORK")
    print("-" * 40)
    initial_expression = {'sox2': 0.5, 'nanog': 0.3, 'pax6': 0.0}
    signals = {'wnt': 0.8, 'bmp': 0.2, 'shh': 0.6}

    print("Initial expression:", initial_expression)
    new_expression = lab.gene_regulatory_network(initial_expression, signals)
    print("After regulation:", {k: f"{v:.3f}" for k, v in new_expression.items()})

    print("\n11. TISSUE MECHANICS")
    print("-" * 40)
    cell_positions = np.random.randn(20, 3) * 10 + 25
    forces = lab.calculate_tissue_mechanics(cell_positions, k_spring=1.0, rest_length=10.0)
    print(f"Mechanical forces calculated for {len(cell_positions)} cells")
    print(f"Mean force magnitude: {np.mean(np.linalg.norm(forces, axis=1)):.4f}")

    print("\n12. COMPREHENSIVE SIMULATION")
    print("-" * 40)
    print("Running full developmental simulation...")
    results = lab.run_comprehensive_simulation()

    print("\nSimulation Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 80)
    print("Developmental Biology Lab demonstration complete!")
    print("=" * 80)


if __name__ == '__main__':
    run_demo()