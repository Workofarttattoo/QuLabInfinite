"""
QuLab Infinite Physics Engine

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Comprehensive physics simulation with real-world accuracy.
Multi-scale support from quantum to macroscopic phenomena.
"""

from .physics_core import (
    PhysicsCore,
    SimulationConfig,
    SimulationScale,
    create_benchmark_simulation
)

from .mechanics import (
    MechanicsEngine,
    Particle,
    RigidBody,
    spring_force,
    damped_spring_force
)

from .thermodynamics import (
    ThermodynamicsEngine,
    ThermalNode,
    MaterialProperties,
    Phase,
    MATERIALS
)

from .fluid_dynamics import (
    FluidDynamicsEngine,
    NavierStokesSolver,
    FluidType,
    BoundaryCondition
)

from .electromagnetism import (
    ElectromagnetismEngine,
    Charge,
    Current,
    coulomb_force,
    lorentz_force,
    biot_savart
)

from .quantum_mechanics import (
    SchrodingerSolver,
    TimeDepSchrodingerSolver,
    QuantumState,
    particle_in_box,
    harmonic_oscillator,
    hydrogen_coulomb
)

from .fundamental_constants import (
    c, h, hbar, k_B, G, e, m_e, m_p, m_n,
    N_A, R, sigma, epsilon_0, mu_0,
    alpha, R_inf, a_0, g_0, atm, T_0, u, F
)

from .units_system import (
    convert,
    Quantity,
    ALL_UNITS
)

__version__ = "1.0.0"
__author__ = "Joshua Hendricks Cole"
__copyright__ = "Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light)"
__license__ = "PATENT PENDING"

__all__ = [
    # Core
    "PhysicsCore",
    "SimulationConfig",
    "SimulationScale",
    "create_benchmark_simulation",

    # Mechanics
    "MechanicsEngine",
    "Particle",
    "RigidBody",
    "spring_force",
    "damped_spring_force",

    # Thermodynamics
    "ThermodynamicsEngine",
    "ThermalNode",
    "MaterialProperties",
    "Phase",
    "MATERIALS",

    # Fluids
    "FluidDynamicsEngine",
    "NavierStokesSolver",
    "FluidType",
    "BoundaryCondition",

    # Electromagnetism
    "ElectromagnetismEngine",
    "Charge",
    "Current",
    "coulomb_force",
    "lorentz_force",
    "biot_savart",

    # Quantum
    "SchrodingerSolver",
    "TimeDepSchrodingerSolver",
    "QuantumState",
    "particle_in_box",
    "harmonic_oscillator",
    "hydrogen_coulomb",

    # Constants
    "c", "h", "hbar", "k_B", "G", "e", "m_e", "m_p", "m_n",
    "N_A", "R", "sigma", "epsilon_0", "mu_0",
    "alpha", "R_inf", "a_0", "g_0", "atm", "T_0", "u", "F",

    # Units
    "convert",
    "Quantity",
    "ALL_UNITS",
]


def test_installation():
    """Quick test to verify physics engine is working."""
    print("QuLab Infinite Physics Engine")
    print(f"Version: {__version__}")
    print(f"{__copyright__}")
    print()

    # Test mechanics
    print("Testing mechanics engine...")
    engine = MechanicsEngine()
    p = Particle(mass=1.0, position=np.array([0.0, 0.0, 1.0]),
                velocity=np.zeros(3), force=np.zeros(3), radius=0.1)
    engine.add_particle(p)
    engine.step(0.001)
    print("  ✓ Mechanics engine working")

    # Test thermodynamics
    print("Testing thermodynamics engine...")
    thermo = ThermodynamicsEngine()
    node = ThermalNode(temperature=300.0, mass=1.0, material=MATERIALS["water"],
                      position=np.array([0.0, 0.0, 0.0]), volume=0.001)
    thermo.add_node(node)
    thermo.step(0.1)
    print("  ✓ Thermodynamics engine working")

    # Test fluids
    print("Testing fluid dynamics engine...")
    fluids = FluidDynamicsEngine((50, 50), dx=0.01, dt=0.001, viscosity=0.01)
    fluids.initialize_flow(density=1.0)
    fluids.step()
    print("  ✓ Fluid dynamics engine working")

    # Test EM
    print("Testing electromagnetism engine...")
    em = ElectromagnetismEngine((50, 50, 50), dx=0.01)
    em.step()
    print("  ✓ Electromagnetism engine working")

    # Test quantum
    print("Testing quantum mechanics engine...")
    x = np.linspace(0, 1e-9, 100)
    V = particle_in_box(x, 1e-9)
    qm = SchrodingerSolver(x, V)
    states = qm.solve_eigenstates(n_states=2)
    print("  ✓ Quantum mechanics engine working")

    # Test core integration
    print("Testing physics core integration...")
    core = create_benchmark_simulation("free_fall")
    core.step()
    print("  ✓ Physics core integration working")

    print()
    print("All engines operational!")
    return True


# Import numpy for test_installation
import numpy as np


if __name__ == "__main__":
    test_installation()
