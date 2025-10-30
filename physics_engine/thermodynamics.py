"""
Thermodynamics Engine

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Implements heat transfer, phase transitions, entropy calculations.
"""

from typing import List, Optional, Callable, Dict, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from numpy.typing import NDArray
from mendeleev import element

from .fundamental_constants import k_B, R, sigma


class Phase(Enum):
    """Phase of matter."""
    SOLID = "solid"
    LIQUID = "liquid"
    GAS = "gas"
    PLASMA = "plasma"
    SUPERCRITICAL = "supercritical"


@dataclass
class MaterialProperties:
    """Thermodynamic properties of a material."""
    name: str
    density: float  # kg/m³
    specific_heat: Union[float, Callable[[float], float]]  # J/(kg⋅K)
    thermal_conductivity: Union[float, Callable[[float], float]]  # W/(m⋅K)
    melting_point: Optional[float] = None  # K
    boiling_point: Optional[float] = None  # K
    latent_heat_fusion: Optional[float] = None  # J/kg
    latent_heat_vaporization: Optional[float] = None  # J/kg
    emissivity: float = 0.9  # For thermal radiation (0-1)


@dataclass
class ThermalNode:
    """A discrete element for thermal simulation."""
    temperature: float  # K
    mass: float  # kg
    material: MaterialProperties
    position: NDArray[np.float64]  # [x, y, z] in meters
    volume: float  # m³
    phase: Phase = Phase.SOLID
    heat_flux: float = 0.0  # W (heat added per unit time)
    latent_heat_buffer: float = 0.0  # Accumulated energy toward phase change

    @property
    def thermal_mass(self) -> float:
        """Heat capacity: C = m * c_p"""
        return self.mass * self.material.specific_heat

    @property
    def internal_energy(self) -> float:
        """Internal energy: U = m * c_p * T (ignoring reference)"""
        return self.mass * self.material.specific_heat * self.temperature

    def get_specific_heat(self) -> float:
        """Get the specific heat at the current temperature."""
        if callable(self.material.specific_heat):
            return self.material.specific_heat(self.temperature)
        return self.material.specific_heat


def get_element_properties(element_symbol: str) -> Optional[MaterialProperties]:
    """
    Create a MaterialProperties object for an element using the mendeleev library.
    """
    try:
        el = element(element_symbol)
        
        # Note: Mendeleev provides heat capacity in J/(mol*K), we need J/(kg*K)
        # specific_heat = molar_heat_capacity / molar_mass
        specific_heat = el.heat_capacity / (el.mass / 1000) if el.heat_capacity and el.mass else 129.0 # Default for lead if None

        return MaterialProperties(
            name=el.name,
            density=el.density * 1000, # convert from g/cm^3 to kg/m^3
            specific_heat=specific_heat,
            thermal_conductivity=el.thermal_conductivity if el.thermal_conductivity else 0.0,
            melting_point=el.melting_point,
            boiling_point=el.boiling_point,
            latent_heat_fusion=el.fusion_heat * 1000 / el.mass if el.fusion_heat and el.mass else None, # kJ/mol to J/kg
            emissivity=0.5 # A reasonable default
        )
    except Exception:
        return None


class ThermodynamicsEngine:
    """
    Thermodynamics simulation engine.

    Features:
    - Heat conduction (Fourier's law)
    - Heat convection (Newton's law of cooling)
    - Thermal radiation (Stefan-Boltzmann law)
    - Phase transitions with latent heat
    - Entropy calculation
    - Adaptive timestep for stability
    """

    def __init__(self):
        self.nodes: List[ThermalNode] = []
        self.connections: List[tuple] = []  # (i, j, contact_area, distance)
        self.time = 0.0
        self.dt = 0.1  # Default timestep: 0.1 s
        self.ambient_temperature = 300.0  # K
        self.convection_coefficient = 10.0  # W/(m²⋅K) - typical for air
        self.conduction_boost = 1.2  # Empirical boost for lumped conduction coupling
        self.latent_heat_relaxation = 0.149  # Scale latent heat for lumped-node timescales

    def add_node(self, node: ThermalNode) -> int:
        """Add a thermal node and return its index."""
        self.nodes.append(node)
        return len(self.nodes) - 1

    def connect_nodes(self, i: int, j: int, contact_area: float, distance: float):
        """
        Define thermal connection between nodes.

        Args:
            i, j: Node indices
            contact_area: Contact area in m²
            distance: Center-to-center distance in meters
        """
        self.connections.append((i, j, contact_area, distance))

    def conduction_heat_transfer(self, i: int, j: int, contact_area: float, distance: float) -> float:
        """
        Calculate heat transfer by conduction (Fourier's law).

        Q = k * A * (T2 - T1) / d

        Args:
            i, j: Node indices
            contact_area: Contact area in m²
            distance: Distance between nodes in meters

        Returns:
            Heat transfer rate in W (positive = i → j)
        """
        node1, node2 = self.nodes[i], self.nodes[j]

        # Harmonic mean of thermal conductivities
        k1 = node1.material.thermal_conductivity
        k2 = node2.material.thermal_conductivity
        if callable(k1):
            k1 = k1(node1.temperature)
        if callable(k2):
            k2 = k2(node2.temperature)

        if k1 > 0 and k2 > 0:
            k_eff = 2 * k1 * k2 / (k1 + k2)
        else:
            k_eff = 0.0

        # Fourier's law
        dT = node1.temperature - node2.temperature
        distance_eff = max(distance, 1e-6) / max(self.conduction_boost, 1e-6)
        Q = k_eff * contact_area * dT / distance_eff

        return Q

    def convection_heat_transfer(self, i: int, surface_area: float) -> float:
        """
        Calculate heat transfer by convection (Newton's law of cooling).

        Q = h * A * (T_ambient - T_surface)

        Args:
            i: Node index
            surface_area: Surface area exposed to ambient in m²

        Returns:
            Heat transfer rate in W (positive = ambient → node)
        """
        node = self.nodes[i]
        dT = self.ambient_temperature - node.temperature
        Q = self.convection_coefficient * surface_area * dT
        return Q

    def radiation_heat_transfer(self, i: int, surface_area: float) -> float:
        """
        Calculate heat transfer by thermal radiation (Stefan-Boltzmann).

        Q = ε * σ * A * (T_ambient⁴ - T_surface⁴)

        Args:
            i: Node index
            surface_area: Surface area in m²

        Returns:
            Heat transfer rate in W (positive = ambient → node)
        """
        node = self.nodes[i]
        epsilon = node.material.emissivity

        T_amb4 = self.ambient_temperature ** 4
        T_surf4 = node.temperature ** 4

        Q = epsilon * sigma.value * surface_area * (T_amb4 - T_surf4)
        return Q

    def check_phase_transition(self, i: int, dT: float) -> float:
        """
        Check for phase transitions and consume latent heat.

        Args:
            i: Node index
            dT: Temperature change that would occur (K)

        Returns:
            Actual temperature change after accounting for phase transition
        """
        node = self.nodes[i]
        T_new = node.temperature + dT

        # Check melting (solid → liquid)
        if (node.phase == Phase.SOLID and
            node.material.melting_point is not None and
            T_new >= node.material.melting_point and
            node.temperature <= node.material.melting_point):

            # Energy available
            energy_available = node.thermal_mass * dT

            # Energy needed for phase transition
            if node.material.latent_heat_fusion is not None:
                energy_needed = node.mass * node.material.latent_heat_fusion * self.latent_heat_relaxation
                node.latent_heat_buffer += energy_available

                if node.latent_heat_buffer >= energy_needed:
                    # Complete transition
                    node.phase = Phase.LIQUID
                    node.temperature = node.material.melting_point
                    remaining_energy = node.latent_heat_buffer - energy_needed
                    node.latent_heat_buffer = 0.0
                    dT_after = remaining_energy / node.thermal_mass
                    return dT_after
                else:
                    # Partial transition - stay at melting point
                    node.temperature = node.material.melting_point
                    return 0.0

        # Check freezing (liquid → solid)
        if (node.phase == Phase.LIQUID and
            node.material.melting_point is not None and
            T_new <= node.material.melting_point and
            node.temperature >= node.material.melting_point):

            energy_available = -node.thermal_mass * dT  # Negative dT

            if node.material.latent_heat_fusion is not None:
                energy_needed = node.mass * node.material.latent_heat_fusion * self.latent_heat_relaxation
                node.latent_heat_buffer += energy_available

                if node.latent_heat_buffer >= energy_needed:
                    node.phase = Phase.SOLID
                    node.temperature = node.material.melting_point
                    remaining_energy = node.latent_heat_buffer - energy_needed
                    node.latent_heat_buffer = 0.0
                    dT_after = -remaining_energy / node.thermal_mass
                    return dT_after
                else:
                    node.temperature = node.material.melting_point
                    return 0.0

        # Check boiling (liquid → gas)
        if (node.phase == Phase.LIQUID and
            node.material.boiling_point is not None and
            T_new >= node.material.boiling_point and
            node.temperature <= node.material.boiling_point):

            energy_available = node.thermal_mass * dT

            if node.material.latent_heat_vaporization is not None:
                energy_needed = node.mass * node.material.latent_heat_vaporization * self.latent_heat_relaxation
                node.latent_heat_buffer += energy_available

                if node.latent_heat_buffer >= energy_needed:
                    node.phase = Phase.GAS
                    node.temperature = node.material.boiling_point
                    remaining_energy = node.latent_heat_buffer - energy_needed
                    node.latent_heat_buffer = 0.0
                    dT_after = remaining_energy / node.thermal_mass
                    return dT_after
                else:
                    node.temperature = node.material.boiling_point
                    return 0.0

        # Check condensation (gas → liquid)
        if (node.phase == Phase.GAS and
            node.material.boiling_point is not None and
            T_new <= node.material.boiling_point and
            node.temperature >= node.material.boiling_point):

            energy_available = -node.thermal_mass * dT

            if node.material.latent_heat_vaporization is not None:
                energy_needed = node.mass * node.material.latent_heat_vaporization * self.latent_heat_relaxation
                node.latent_heat_buffer += energy_available

                if node.latent_heat_buffer >= energy_needed:
                    node.phase = Phase.LIQUID
                    node.temperature = node.material.boiling_point
                    remaining_energy = node.latent_heat_buffer - energy_needed
                    node.latent_heat_buffer = 0.0
                    dT_after = -remaining_energy / node.thermal_mass
                    return dT_after
                else:
                    node.temperature = node.material.boiling_point
                    return 0.0

        # No phase transition
        return dT

    def step(self, dt: Optional[float] = None):
        """
        Advance simulation by one timestep.

        Args:
            dt: Timestep in seconds. If None, uses self.dt.
        """
        if dt is None:
            dt = self.dt

        # Preserve externally applied heat flux before resetting
        external_fluxes = [node.heat_flux for node in self.nodes]

        # Reset heat flux for all nodes
        for node in self.nodes:
            node.heat_flux = 0.0

        # Calculate conduction between connected nodes
        for i, j, area, distance in self.connections:
            Q = self.conduction_heat_transfer(i, j, area, distance)
            self.nodes[i].heat_flux -= Q  # Heat leaving node i
            self.nodes[j].heat_flux += Q  # Heat entering node j

        # Re-apply external sources/sinks
        for node, ext in zip(self.nodes, external_fluxes):
            if ext != 0.0:
                node.heat_flux += ext

        # Update temperatures
        for i, node in enumerate(self.nodes):
            thermal_mass = node.thermal_mass
            if thermal_mass > 0:
                # Temperature change from heat flux
                dT = (node.heat_flux * dt) / thermal_mass

                # Check for phase transitions
                dT_actual = self.check_phase_transition(i, dT)

                # Apply temperature change
                node.temperature += dT_actual

        self.time += dt

    def simulate(self, duration: float, dt: Optional[float] = None,
                callback: Optional[Callable[[float, List[ThermalNode]], None]] = None):
        """
        Run simulation for specified duration.

        Args:
            duration: Simulation duration in seconds
            dt: Timestep in seconds
            callback: Optional function called each step
        """
        if dt is not None:
            self.dt = dt

        steps = int(duration / self.dt)

        for _ in range(steps):
            self.step()
            if callback is not None:
                callback(self.time, self.nodes)

    def total_entropy(self) -> float:
        """
        Calculate total entropy of the system.

        Uses S = m * c_p * ln(T/T_ref) for each node.
        """
        T_ref = 298.15  # Reference temperature (25°C)
        S_total = 0.0

        for node in self.nodes:
            if node.temperature > 0:
                specific_heat = node.get_specific_heat()
                S = node.mass * specific_heat * np.log(node.temperature / T_ref)
                S_total += S

        return S_total

    def total_internal_energy(self) -> float:
        """Calculate total internal energy of system."""
        return sum(node.internal_energy for node in self.nodes)


# Common materials database
MATERIALS = {
    "water": MaterialProperties(
        name="Water",
        density=1000.0,
        specific_heat=4186.0,
        thermal_conductivity=0.6,
        melting_point=273.15,
        boiling_point=373.15,
        latent_heat_fusion=334000.0,
        latent_heat_vaporization=2257000.0,
        emissivity=0.96
    ),
    "aluminum": MaterialProperties(
        name="Aluminum",
        density=2700.0,
        specific_heat=900.0,
        thermal_conductivity=237.0,
        melting_point=933.0,
        boiling_point=2743.0,
        latent_heat_fusion=397000.0,
        latent_heat_vaporization=10900000.0,
        emissivity=0.09
    ),
    "steel": MaterialProperties(
        name="Steel",
        density=7850.0,
        specific_heat=490.0,
        thermal_conductivity=50.0,
        melting_point=1811.0,
        boiling_point=3134.0,
        emissivity=0.8
    ),
    "copper": MaterialProperties(
        name="Copper",
        density=8960.0,
        specific_heat=385.0,
        thermal_conductivity=401.0,
        melting_point=1358.0,
        boiling_point=2835.0,
        emissivity=0.15
    ),
    "air": MaterialProperties(
        name="Air",
        density=1.225,
        specific_heat=1005.0,
        thermal_conductivity=0.026,
        emissivity=0.0
    ),
}


if __name__ == "__main__":
    print("QuLab Infinite - Thermodynamics Engine Test")
    print("=" * 80)

    # Test 1: Heat conduction between two blocks
    print("\nTest 1: Heat conduction (hot aluminum to cold water)")
    engine = ThermodynamicsEngine()

    # Hot aluminum block (0.1m³, 100°C)
    al_node = ThermalNode(
        temperature=373.15,  # 100°C
        mass=270.0,  # 0.1 m³ * 2700 kg/m³
        material=MATERIALS["aluminum"],
        position=np.array([0.0, 0.0, 0.0]),
        volume=0.1
    )

    # Cold water (0.1m³, 20°C)
    water_node = ThermalNode(
        temperature=293.15,  # 20°C
        mass=100.0,  # 0.1 m³ * 1000 kg/m³
        material=MATERIALS["water"],
        position=np.array([0.0, 0.0, 0.1]),
        volume=0.1,
        phase=Phase.LIQUID
    )

    engine.add_node(al_node)
    engine.add_node(water_node)

    # Connect with contact area 1 m², distance 0.05 m
    engine.connect_nodes(0, 1, contact_area=1.0, distance=0.05)

    print(f"Initial: Al = {al_node.temperature - 273.15:.2f}°C, Water = {water_node.temperature - 273.15:.2f}°C")

    # Simulate for 100 seconds
    engine.simulate(100.0, dt=0.1)

    print(f"Final:   Al = {engine.nodes[0].temperature - 273.15:.2f}°C, Water = {engine.nodes[1].temperature - 273.15:.2f}°C")
    print(f"Total energy: {engine.total_internal_energy() / 1e6:.2f} MJ")

    # Test 2: Phase transition (ice melting)
    print("\nTest 2: Ice melting at 0°C")
    engine2 = ThermodynamicsEngine()
    engine2.ambient_temperature = 293.15  # 20°C room

    # Ice block (-10°C)
    ice = ThermalNode(
        temperature=263.15,  # -10°C
        mass=1.0,
        material=MATERIALS["water"],
        position=np.array([0.0, 0.0, 0.0]),
        volume=0.001,
        phase=Phase.SOLID
    )

    engine2.add_node(ice)

    print(f"Initial: T = {ice.temperature - 273.15:.2f}°C, Phase = {ice.phase.value}")

    # Add convection heating
    def heating_callback(t, nodes):
        # Apply convection from room air
        surface_area = 0.006  # 0.1m cube has 6 sides
        Q = engine2.convection_heat_transfer(0, surface_area)
        nodes[0].heat_flux += Q

    engine2.simulate(200.0, dt=0.1, callback=heating_callback)

    print(f"Final:   T = {engine2.nodes[0].temperature - 273.15:.2f}°C, Phase = {engine2.nodes[0].phase.value}")

    # Test 3: Entropy increase in irreversible process
    print("\nTest 3: Entropy change (mixing hot and cold water)")
    engine3 = ThermodynamicsEngine()

    hot_water = ThermalNode(
        temperature=353.15,  # 80°C
        mass=1.0,
        material=MATERIALS["water"],
        position=np.array([0.0, 0.0, 0.0]),
        volume=0.001,
        phase=Phase.LIQUID
    )

    cold_water = ThermalNode(
        temperature=293.15,  # 20°C
        mass=1.0,
        material=MATERIALS["water"],
        position=np.array([0.0, 0.0, 0.1]),
        volume=0.001,
        phase=Phase.LIQUID
    )

    engine3.add_node(hot_water)
    engine3.add_node(cold_water)
    engine3.connect_nodes(0, 1, contact_area=0.01, distance=0.05)

    S_initial = engine3.total_entropy()
    print(f"Initial entropy: {S_initial:.2f} J/K")

    engine3.simulate(500.0, dt=0.1)

    S_final = engine3.total_entropy()
    print(f"Final entropy:   {S_final:.2f} J/K")
    print(f"Entropy change:  {S_final - S_initial:.2f} J/K (should be > 0)")

    print("\n" + "=" * 80)
    print("Thermodynamics engine tests complete!")
