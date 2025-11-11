"""
Tumor growth and microenvironment simulation primitives.

The implementation provides a simplified, research-oriented approximation of
solid tumour dynamics. Parameters were sourced from publicly available cancer
biology references where possible and otherwise set to plausible defaults.
Results should be treated as qualitative guidance for experimentation only and
not as clinically validated predictions.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import time


class CellCyclePhase(Enum):
    """Cell cycle phases with realistic durations"""
    G0 = "G0"  # Quiescent (resting)
    G1 = "G1"  # Growth phase 1 (8-10 hours)
    S = "S"    # DNA synthesis (6-8 hours)
    G2 = "G2"  # Growth phase 2 (4-6 hours)
    M = "M"    # Mitosis (1-2 hours)
    APOPTOSIS = "APOPTOSIS"  # Cell death
    NECROSIS = "NECROSIS"    # Uncontrolled death


class TumorGrowthModel(Enum):
    """Validated tumor growth models from literature"""
    EXPONENTIAL = "exponential"          # Unrestricted growth (early stage)
    GOMPERTZIAN = "gompertzian"          # Most realistic for solid tumors
    LOGISTIC = "logistic"                # Resource-limited growth
    BERTALANFFY = "bertalanffy"          # Von Bertalanffy growth
    POWER_LAW = "power_law"              # Fractal growth


@dataclass
class CancerCell:
    """
    Individual cancer cell with realistic biological parameters
    Based on average human cell: ~10-20 μm diameter, 1-2 pg mass
    """
    # Identity
    cell_id: int
    position: np.ndarray  # 3D position in tumor (μm)

    # Cell cycle
    phase: CellCyclePhase = CellCyclePhase.G1
    phase_time: float = 0.0  # Hours in current phase
    division_time: float = 24.0  # Hours (typical cancer cell: 18-24h)

    # Viability
    is_alive: bool = True
    apoptosis_threshold: float = 0.5  # 0-1 scale
    necrosis_threshold: float = 0.3   # 0-1 scale
    viability_score: float = 1.0      # 1.0 = fully healthy
    time_since_death: float = 0.0     # Hours since death (for cleanup)

    # Metabolism (The 10 Fields at cellular level)
    local_ph: float = 7.4
    local_oxygen: float = 21.0  # % O2
    local_glucose: float = 5.5   # mM
    local_lactate: float = 1.0   # mM
    local_temperature: float = 37.0  # Celsius
    local_ros: float = 0.1       # μM H2O2 equivalent
    local_glutamine: float = 0.6  # mM
    local_calcium: float = 100.0  # μM
    atp_adp_ratio: float = 10.0
    cytokine_exposure: float = 5.0  # pg/mL

    # Genetics
    mutation_count: int = 0
    resistance_mutations: Dict[str, bool] = field(default_factory=dict)
    oncogenes_active: List[str] = field(default_factory=lambda: ['MYC', 'RAS'])
    tumor_suppressors_lost: List[str] = field(default_factory=lambda: ['TP53', 'RB1'])

    # Microenvironment
    distance_to_vasculature: float = 100.0  # μm
    nutrient_access: float = 1.0  # 0-1 scale
    waste_accumulation: float = 0.0  # 0-1 scale

    # Drug exposure
    drug_concentrations: Dict[str, float] = field(default_factory=dict)
    drug_resistance_factors: Dict[str, float] = field(default_factory=dict)

    def calculate_proliferation_rate(self) -> float:
        """
        Calculate cell proliferation rate based on microenvironment
        Returns: divisions per day (normal cancer cell: ~1.0)
        """
        base_rate = 1.0  # 1 division per day

        # Oxygen effect (hypoxia slows division)
        if self.local_oxygen < 5.0:
            oxygen_factor = self.local_oxygen / 5.0
        else:
            oxygen_factor = 1.0

        # Glucose effect (Warburg - high glucose increases rate)
        if self.local_glucose > 7.0:
            glucose_factor = 1.2
        elif self.local_glucose < 3.0:
            glucose_factor = 0.5
        else:
            glucose_factor = 1.0

        # pH effect (acidic environment)
        if self.local_ph < 7.0:
            ph_factor = 0.8  # Slight slowdown but more invasive
        else:
            ph_factor = 1.0

        # Nutrient access
        nutrient_factor = self.nutrient_access

        # ATP/ADP (energy availability)
        energy_factor = min(1.5, self.atp_adp_ratio / 10.0)

        proliferation_rate = base_rate * oxygen_factor * glucose_factor * ph_factor * nutrient_factor * energy_factor

        return max(0.0, proliferation_rate)

    def calculate_apoptosis_probability(self, dt: float) -> float:
        """
        Calculate probability of apoptosis in time step dt
        Based on cellular stress and damage
        """
        base_apoptosis = 0.001 * dt  # 0.1% per hour baseline

        # Stress factors that increase apoptosis
        hypoxia_stress = max(0, (5.0 - self.local_oxygen) / 5.0) * 0.01 * dt
        ros_stress = max(0, (self.local_ros - 2.0) / 10.0) * 0.02 * dt
        nutrient_stress = max(0, (1.0 - self.nutrient_access)) * 0.015 * dt
        low_atp_stress = max(0, (8.0 - self.atp_adp_ratio) / 8.0) * 0.01 * dt

        # Drug-induced apoptosis
        drug_stress = sum(self.drug_concentrations.values()) * 0.005 * dt

        # Protection from oncogenes
        oncogene_protection = 0.5 if 'BCL2' in self.oncogenes_active else 1.0

        total_apoptosis_prob = (base_apoptosis + hypoxia_stress + ros_stress +
                               nutrient_stress + low_atp_stress + drug_stress) * oncogene_protection

        return min(1.0, total_apoptosis_prob)

    def calculate_necrosis_probability(self, dt: float) -> float:
        """
        Calculate probability of necrosis (uncontrolled death)
        Usually due to severe hypoxia or lack of nutrients
        """
        base_necrosis = 0.0

        # Severe hypoxia (< 1% O2)
        if self.local_oxygen < 1.0:
            base_necrosis += 0.05 * dt

        # Severe nutrient deprivation
        if self.nutrient_access < 0.2:
            base_necrosis += 0.03 * dt

        # Severe waste accumulation
        if self.waste_accumulation > 0.8:
            base_necrosis += 0.02 * dt

        # Extreme pH
        if self.local_ph < 6.5:
            base_necrosis += 0.015 * dt

        return min(1.0, base_necrosis)

    def update_viability(self, dt: float):
        """Update cell viability and determine if cell dies"""
        # Calculate death probabilities
        apoptosis_prob = self.calculate_apoptosis_probability(dt)
        necrosis_prob = self.calculate_necrosis_probability(dt)

        # Update viability score
        stress = (apoptosis_prob + necrosis_prob) / dt
        self.viability_score = max(0.0, self.viability_score - stress * dt * 0.1)

        # Determine death
        if np.random.random() < necrosis_prob:
            self.is_alive = False
            self.phase = CellCyclePhase.NECROSIS
            self.time_since_death = 0.0
            return False

        if np.random.random() < apoptosis_prob:
            self.is_alive = False
            self.phase = CellCyclePhase.APOPTOSIS
            self.time_since_death = 0.0
            return False

        # Surviving cells reset the counter
        self.time_since_death = 0.0
        return True

    def advance_cell_cycle(self, dt: float):
        """Advance cell through cell cycle phases"""
        if not self.is_alive or self.phase in [CellCyclePhase.APOPTOSIS, CellCyclePhase.NECROSIS]:
            return None

        self.phase_time += dt

        # Phase durations (hours) - realistic values for cancer cells
        phase_durations = {
            CellCyclePhase.G0: float('inf'),  # Indefinite
            CellCyclePhase.G1: 10.0,
            CellCyclePhase.S: 8.0,
            CellCyclePhase.G2: 4.0,
            CellCyclePhase.M: 1.0,
        }

        if self.phase_time >= phase_durations.get(self.phase, float('inf')):
            # Transition to next phase
            transitions = {
                CellCyclePhase.G0: CellCyclePhase.G1,
                CellCyclePhase.G1: CellCyclePhase.S,
                CellCyclePhase.S: CellCyclePhase.G2,
                CellCyclePhase.G2: CellCyclePhase.M,
                CellCyclePhase.M: CellCyclePhase.G1,  # Division creates new cell
            }

            if self.phase == CellCyclePhase.M:
                # Cell division!
                self.phase = CellCyclePhase.G1
                self.phase_time = 0.0
                return "DIVIDE"
            else:
                self.phase = transitions.get(self.phase, self.phase)
                self.phase_time = 0.0

        return None


@dataclass
class TumorMicroenvironment:
    """
    3D spatial tumor microenvironment
    Simulates gradients of nutrients, oxygen, waste
    """
    # Grid dimensions (in micrometers)
    grid_size: Tuple[int, int, int] = (200, 200, 200)  # 200μm x 200μm x 200μm
    resolution: float = 10.0  # μm per grid point

    # 3D fields (the 10 fields spatially distributed)
    ph_field: np.ndarray = None
    oxygen_field: np.ndarray = None
    glucose_field: np.ndarray = None
    lactate_field: np.ndarray = None
    temperature_field: np.ndarray = None
    ros_field: np.ndarray = None
    glutamine_field: np.ndarray = None
    calcium_field: np.ndarray = None
    atp_field: np.ndarray = None
    cytokine_field: np.ndarray = None

    # Vasculature (blood vessel locations)
    vessel_locations: List[np.ndarray] = field(default_factory=list)

    # Diffusion coefficients (μm²/s) - from literature
    oxygen_diffusion: float = 2000.0     # O2 diffuses quickly
    glucose_diffusion: float = 600.0     # Smaller molecules
    lactate_diffusion: float = 500.0
    glutamine_diffusion: float = 400.0

    def __post_init__(self):
        """Initialize 3D fields"""
        if self.ph_field is None:
            self.ph_field = np.full(self.grid_size, 7.4, dtype=np.float32)
            self.oxygen_field = np.full(self.grid_size, 21.0, dtype=np.float32)
            self.glucose_field = np.full(self.grid_size, 5.5, dtype=np.float32)
            self.lactate_field = np.full(self.grid_size, 1.0, dtype=np.float32)
            self.temperature_field = np.full(self.grid_size, 37.0, dtype=np.float32)
            self.ros_field = np.full(self.grid_size, 0.1, dtype=np.float32)
            self.glutamine_field = np.full(self.grid_size, 0.6, dtype=np.float32)
            self.calcium_field = np.full(self.grid_size, 100.0, dtype=np.float32)
            self.atp_field = np.full(self.grid_size, 10.0, dtype=np.float32)
            self.cytokine_field = np.full(self.grid_size, 5.0, dtype=np.float32)

    def add_blood_vessel(self, position: np.ndarray):
        """Add a blood vessel at position (source of nutrients)"""
        self.vessel_locations.append(position)

    def diffuse_field(self, field: np.ndarray, diffusion_coeff: float, consumption_rate: np.ndarray, dt: float) -> np.ndarray:
        """
        Solve diffusion equation with consumption
        ∂C/∂t = D∇²C - k*C
        Using simple finite difference method
        """
        # Laplacian (simplified 3D)
        laplacian = (
            np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
            np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) +
            np.roll(field, 1, axis=2) + np.roll(field, -1, axis=2) -
            6 * field
        ) / (self.resolution ** 2)

        # Update field
        new_field = field + dt * (diffusion_coeff * laplacian - consumption_rate)

        # Clamp to non-negative
        new_field = np.maximum(0, new_field)

        return new_field

    def update_microenvironment(self, cell_positions: np.ndarray, cell_consumption: Dict[str, np.ndarray], dt: float):
        """Update all fields based on diffusion and cell consumption"""
        # Update oxygen (fast diffusion, high consumption)
        self.oxygen_field = self.diffuse_field(
            self.oxygen_field, self.oxygen_diffusion,
            cell_consumption.get('oxygen', np.zeros(self.grid_size)), dt
        )

        # Update glucose
        self.glucose_field = self.diffuse_field(
            self.glucose_field, self.glucose_diffusion,
            cell_consumption.get('glucose', np.zeros(self.grid_size)), dt
        )

        # Update lactate (produced by cells, so negative consumption)
        self.lactate_field = self.diffuse_field(
            self.lactate_field, self.lactate_diffusion,
            -cell_consumption.get('lactate_production', np.zeros(self.grid_size)), dt
        )

        # Lactate acidifies pH
        self.ph_field = 7.4 - (self.lactate_field - 1.0) * 0.05

        # ROS increases in hypoxic regions
        hypoxic_mask = self.oxygen_field < 5.0
        self.ros_field[hypoxic_mask] += 0.01 * dt
        self.ros_field[~hypoxic_mask] = np.maximum(0.1, self.ros_field[~hypoxic_mask] - 0.005 * dt)


class TumorSimulator:
    """
    Main tumour simulator built on well-studied growth models.

    Tracks individual cells and aggregates their behaviour to approximate tumour
    dynamics. Parameter defaults come from published sources when available and
    otherwise from reasonable heuristics.
    """

    def __init__(self,
                 tumor_type: str = "solid_tumor",
                 growth_model: TumorGrowthModel = TumorGrowthModel.GOMPERTZIAN,
                 initial_cells: int = 100):
        """
        Initialize tumor simulator

        Args:
            tumor_type: Type of tumor (affects parameters)
            growth_model: Growth model to use
            initial_cells: Number of cells to start with
        """
        self.tumor_type = tumor_type
        self.growth_model = growth_model

        # Cell population
        self.cells: List[CancerCell] = []
        self.next_cell_id = 0

        # Microenvironment
        self.microenvironment = TumorMicroenvironment()
        self.field_overrides: Dict[str, float] = {}
        self.dead_cell_retention_hours = 24.0

        # Growth parameters (from literature)
        self.carrying_capacity = 1e9  # ~1 billion cells (~1 cm³ tumor)
        self.intrinsic_growth_rate = 0.03  # per hour (doubling time ~23 hours)
        self.gompertz_retardation = 0.001  # Gompertzian retardation coefficient

        # Time tracking
        self.time = 0.0  # hours
        self.dt = 0.1  # time step (hours)

        # Statistics
        self.history = {
            'time': [],
            'cell_count': [],
            'alive_count': [],
            'apoptotic_count': [],
            'necrotic_count': [],
            'average_viability': [],
            'tumor_volume': [],  # mm³
        }

        # Initialize cells
        self._initialize_tumor(initial_cells)

    def apply_field_overrides(self, field_values: Dict[str, float]):
        """
        Use global field overrides supplied by higher-level controllers (e.g.,
        TenFieldController) so that each cell samples the conditioned
        microenvironment instead of the default lattice.
        """
        self.field_overrides = field_values.copy()

    def _get_field_value(self, field_name: str, grid_pos: np.ndarray) -> float:
        """
        Retrieve a microenvironment value for the given lattice position,
        honouring any uniform overrides supplied by the controller.
        """
        if field_name in self.field_overrides:
            return self.field_overrides[field_name]

        field_map = {
            'ph_level': 'ph_field',
            'oxygen_percent': 'oxygen_field',
            'glucose_mm': 'glucose_field',
            'lactate_mm': 'lactate_field',
            'temperature_c': 'temperature_field',
            'ros_um': 'ros_field',
            'glutamine_mm': 'glutamine_field',
            'calcium_um': 'calcium_field',
            'atp_adp_ratio': 'atp_field',
            'cytokine_pg_ml': 'cytokine_field',
        }

        lattice_name = field_map.get(field_name)
        if lattice_name is None:
            raise KeyError(f"Unknown field '{field_name}'")

        lattice = getattr(self.microenvironment, lattice_name)
        return lattice[tuple(grid_pos)]

    def _initialize_tumor(self, n_cells: int):
        """Initialize tumor with n_cells in a spherical cluster"""
        center = np.array(self.microenvironment.grid_size) * self.microenvironment.resolution / 2.0

        for i in range(n_cells):
            # Random position near center (sphere)
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            r = np.random.uniform(0, 50.0)  # Within 50 μm radius

            position = center + r * np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])

            cell = CancerCell(
                cell_id=self.next_cell_id,
                position=position,
                phase=CellCyclePhase.G1
            )
            self.cells.append(cell)
            self.next_cell_id += 1

        # Add blood vessels at periphery
        for i in range(8):
            angle = i * 2 * np.pi / 8
            vessel_pos = center + 150 * np.array([np.cos(angle), np.sin(angle), 0])
            self.microenvironment.add_blood_vessel(vessel_pos)

    def get_cell_count(self) -> int:
        """Total number of cells"""
        return len(self.cells)

    def get_alive_count(self) -> int:
        """Number of living cells"""
        return sum(1 for cell in self.cells if cell.is_alive)

    def get_tumor_volume_mm3(self) -> float:
        """
        Estimate tumor volume in mm³
        Assumes spherical tumor, calculates radius from cell positions
        """
        if not self.cells:
            return 0.0

        positions = np.array([cell.position for cell in self.cells if cell.is_alive])
        if len(positions) == 0:
            return 0.0

        center = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        radius_um = np.max(distances)
        radius_mm = radius_um / 1000.0
        volume_mm3 = (4/3) * np.pi * (radius_mm ** 3)

        return volume_mm3

    def step(self, dt: Optional[float] = None):
        """
        Advance simulation by one time step

        Args:
            dt: Time step in hours (default: self.dt)
        """
        if dt is None:
            dt = self.dt

        # Update each cell
        new_cells = []
        apoptotic_count = 0
        necrotic_count = 0

        for cell in self.cells:
            if not cell.is_alive:
                cell.time_since_death += dt
                if cell.phase == CellCyclePhase.APOPTOSIS:
                    apoptotic_count += 1
                elif cell.phase == CellCyclePhase.NECROSIS:
                    necrotic_count += 1
                continue

            # Update local microenvironment for this cell
            grid_pos = (cell.position / self.microenvironment.resolution).astype(int)
            grid_pos = np.clip(grid_pos, 0, np.array(self.microenvironment.grid_size) - 1)

            cell.local_ph = self._get_field_value('ph_level', grid_pos)
            cell.local_oxygen = self._get_field_value('oxygen_percent', grid_pos)
            cell.local_glucose = self._get_field_value('glucose_mm', grid_pos)
            cell.local_lactate = self._get_field_value('lactate_mm', grid_pos)
            cell.local_temperature = self._get_field_value('temperature_c', grid_pos)
            cell.local_ros = self._get_field_value('ros_um', grid_pos)
            cell.local_glutamine = self._get_field_value('glutamine_mm', grid_pos)
            cell.local_calcium = self._get_field_value('calcium_um', grid_pos)
            cell.atp_adp_ratio = self._get_field_value('atp_adp_ratio', grid_pos)
            cell.cytokine_exposure = self._get_field_value('cytokine_pg_ml', grid_pos)

            # Calculate nutrient access based on distance to nearest vessel
            distances_to_vessels = [
                np.linalg.norm(cell.position - vessel)
                for vessel in self.microenvironment.vessel_locations
            ]
            if distances_to_vessels:
                min_distance = min(distances_to_vessels)
                # Nutrient access decays exponentially with distance (diffusion limit ~150 μm)
                cell.nutrient_access = np.exp(-min_distance / 150.0)

            # Update viability
            cell.update_viability(dt)

            # Advance cell cycle
            if cell.is_alive:
                event = cell.advance_cell_cycle(dt)

                if event == "DIVIDE":
                    # Create daughter cell
                    daughter = CancerCell(
                        cell_id=self.next_cell_id,
                        position=cell.position + np.random.randn(3) * 10.0,  # 10 μm offset
                        phase=CellCyclePhase.G1,
                        division_time=cell.division_time * np.random.uniform(0.9, 1.1),
                        mutation_count=cell.mutation_count + np.random.poisson(0.1),  # Mutations accumulate
                    )
                    new_cells.append(daughter)
                    self.next_cell_id += 1

        # Add new cells from division
        self.cells.extend(new_cells)

        # Remove dead cells (garbage collection after 24 hours)
        self.cells = [
            cell for cell in self.cells
            if cell.is_alive or cell.time_since_death < self.dead_cell_retention_hours
        ]

        # Update microenvironment
        # (Simplified - would need proper spatial mapping for full realism)

        # Update time
        self.time += dt

        # Record statistics
        self.history['time'].append(self.time)
        self.history['cell_count'].append(self.get_cell_count())
        self.history['alive_count'].append(self.get_alive_count())
        self.history['apoptotic_count'].append(apoptotic_count)
        self.history['necrotic_count'].append(necrotic_count)

        alive_cells = [c for c in self.cells if c.is_alive]
        avg_viability = np.mean([c.viability_score for c in alive_cells]) if alive_cells else 0.0
        self.history['average_viability'].append(avg_viability)
        self.history['tumor_volume'].append(self.get_tumor_volume_mm3())

    def simulate(self, duration_hours: float, progress_callback=None):
        """
        Run simulation for specified duration

        Args:
            duration_hours: How long to simulate (hours)
            progress_callback: Optional function(time, cell_count) for progress updates
        """
        steps = int(duration_hours / self.dt)

        for i in range(steps):
            self.step()

            if progress_callback and i % 10 == 0:
                progress_callback(self.time, self.get_alive_count())

    def get_statistics(self) -> Dict:
        """Get current tumor statistics"""
        alive_cells = [c for c in self.cells if c.is_alive]

        if not alive_cells:
            return {
                'total_cells': len(self.cells),
                'alive_cells': 0,
                'dead_cells': len(self.cells),
                'tumor_volume_mm3': 0.0,
                'average_viability': 0.0,
                'time_hours': self.time,
            }

        # Cell cycle distribution
        cycle_distribution = {}
        for phase in CellCyclePhase:
            cycle_distribution[phase.value] = sum(1 for c in alive_cells if c.phase == phase)

        # Metabolic stats (average of 10 fields)
        metabolic_stats = {
            'avg_ph': np.mean([c.local_ph for c in alive_cells]),
            'avg_oxygen': np.mean([c.local_oxygen for c in alive_cells]),
            'avg_glucose': np.mean([c.local_glucose for c in alive_cells]),
            'avg_lactate': np.mean([c.local_lactate for c in alive_cells]),
            'avg_ros': np.mean([c.local_ros for c in alive_cells]),
            'avg_atp_adp': np.mean([c.atp_adp_ratio for c in alive_cells]),
        }

        return {
            'total_cells': len(self.cells),
            'alive_cells': len(alive_cells),
            'dead_cells': len(self.cells) - len(alive_cells),
            'tumor_volume_mm3': self.get_tumor_volume_mm3(),
            'average_viability': np.mean([c.viability_score for c in alive_cells]),
            'cell_cycle_distribution': cycle_distribution,
            'metabolic_stats': metabolic_stats,
            'time_hours': self.time,
            'time_days': self.time / 24.0,
        }
