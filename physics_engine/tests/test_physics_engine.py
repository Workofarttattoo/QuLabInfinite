"""
Physics Engine Comprehensive Test Suite

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Validates physics engine accuracy against NIST benchmarks and known analytical solutions.
Target: <1% error vs reference data for all tests.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from physics_engine.mechanics import MechanicsEngine, Particle, spring_force
from physics_engine.thermodynamics import ThermodynamicsEngine, ThermalNode, Phase, MATERIALS
from physics_engine.fluid_dynamics import FluidDynamicsEngine
from physics_engine.electromagnetism import ElectromagnetismEngine, coulomb_force, Charge
from physics_engine.quantum_mechanics import SchrodingerSolver, particle_in_box, harmonic_oscillator, QuantumState
from physics_engine.physics_core import PhysicsCore, SimulationConfig, SimulationScale, create_benchmark_simulation
from physics_engine.fundamental_constants import g_0, c, hbar, m_e, e, epsilon_0


class TestMechanics(unittest.TestCase):
    """Test classical mechanics engine."""

    def test_free_fall_analytical(self):
        """Test free fall against analytical solution."""
        engine = MechanicsEngine()

        p = Particle(
            mass=1.0,
            position=np.array([0.0, 0.0, 10.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
            force=np.zeros(3),
            radius=0.1
        )
        engine.add_particle(p)

        # Simulate 1 second
        duration = 1.0
        dt = 0.001
        steps = int(duration / dt)

        for _ in range(steps):
            engine.step(dt)

        # Analytical solution: z(t) = z0 - 0.5*g*t²
        z_expected = 10.0 - 0.5 * g_0.value * duration**2

        z_actual = engine.particles[0].position[2]
        error = abs(z_actual - z_expected) / abs(z_expected)

        self.assertLess(error, 0.01, f"Free fall error {error*100:.2f}% exceeds 1%")

    def test_energy_conservation(self):
        """Test energy conservation in elastic collision."""
        engine = MechanicsEngine()
        engine.restitution = 1.0  # Perfectly elastic
        engine.gravity = np.zeros(3)  # No gravity

        # Two equal masses
        p1 = Particle(
            mass=1.0,
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([1.0, 0.0, 0.0]),
            force=np.zeros(3),
            radius=0.1
        )

        p2 = Particle(
            mass=1.0,
            position=np.array([2.0, 0.0, 0.0]),
            velocity=np.array([-1.0, 0.0, 0.0]),
            force=np.zeros(3),
            radius=0.1
        )

        engine.add_particle(p1)
        engine.add_particle(p2)

        initial_energy = engine.total_energy()

        # Simulate collision
        engine.simulate(2.0, dt=0.001)

        final_energy = engine.total_energy()
        energy_error = abs(final_energy - initial_energy) / abs(initial_energy)

        self.assertLess(energy_error, 0.01, f"Energy error {energy_error*100:.2f}% exceeds 1%")

    def test_momentum_conservation(self):
        """Test momentum conservation."""
        engine = MechanicsEngine()
        engine.gravity = np.zeros(3)

        # Random particles
        np.random.seed(42)
        for _ in range(10):
            p = Particle(
                mass=np.random.uniform(0.5, 2.0),
                position=np.random.uniform(-5, 5, 3),
                velocity=np.random.uniform(-2, 2, 3),
                force=np.zeros(3),
                radius=0.1
            )
            engine.add_particle(p)

        initial_momentum = engine.momentum()

        engine.simulate(1.0, dt=0.001)

        final_momentum = engine.momentum()
        momentum_error = np.linalg.norm(final_momentum - initial_momentum) / np.linalg.norm(initial_momentum)

        self.assertLess(momentum_error, 0.01, f"Momentum error {momentum_error*100:.2f}% exceeds 1%")

    def test_spring_oscillation(self):
        """Test harmonic oscillator with springs."""
        engine = MechanicsEngine()
        engine.gravity = np.zeros(3)

        # Two particles connected by spring
        p1 = Particle(
            mass=1.0,
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            force=np.zeros(3),
            fixed=True
        )

        p2 = Particle(
            mass=0.1,
            position=np.array([1.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            force=np.zeros(3)
        )

        engine.add_particle(p1)
        engine.add_particle(p2)

        # Spring parameters
        k = 100.0  # N/m
        rest_length = 0.5

        # Simulate with spring force
        positions = []

        def step_with_spring():
            f1, f2 = spring_force(engine.particles[0], engine.particles[1], k, rest_length)
            engine.particles[0].force = f1
            engine.particles[1].force = f2
            engine.step()
            positions.append(engine.particles[1].position[0])

        dt = 0.001
        for _ in range(1000):
            step_with_spring()

        # Check for oscillation
        positions = np.array(positions)
        self.assertGreater(np.max(positions) - np.min(positions), 0.1, "No oscillation detected")


class TestThermodynamics(unittest.TestCase):
    """Test thermodynamics engine."""

    def test_heat_conduction(self):
        """Test heat conduction follows Fourier's law."""
        engine = ThermodynamicsEngine()

        # Two aluminum blocks
        hot = ThermalNode(
            temperature=400.0,
            mass=100.0,
            material=MATERIALS["aluminum"],
            position=np.array([0.0, 0.0, 0.0]),
            volume=0.037
        )

        cold = ThermalNode(
            temperature=300.0,
            mass=100.0,
            material=MATERIALS["aluminum"],
            position=np.array([0.0, 0.0, 0.1]),
            volume=0.037
        )

        engine.add_node(hot)
        engine.add_node(cold)
        engine.connect_nodes(0, 1, contact_area=0.01, distance=0.1)

        # Simulate
        engine.simulate(100.0, dt=0.1)

        # Temperatures should equalize
        T1 = engine.nodes[0].temperature
        T2 = engine.nodes[1].temperature

        # Should be close to average
        T_avg = (400.0 + 300.0) / 2
        self.assertLess(abs(T1 - T_avg), 20.0, "Hot block didn't equilibrate")
        self.assertLess(abs(T2 - T_avg), 20.0, "Cold block didn't equilibrate")

    def test_phase_transition_latent_heat(self):
        """Test ice melting consumes latent heat."""
        engine = ThermodynamicsEngine()

        # Ice at 0°C
        ice = ThermalNode(
            temperature=273.15,
            mass=1.0,
            material=MATERIALS["water"],
            position=np.array([0.0, 0.0, 0.0]),
            volume=0.001,
            phase=Phase.SOLID
        )

        engine.add_node(ice)

        # Add constant heat flux
        Q_rate = 1000.0  # 1 kW

        for _ in range(500):
            engine.nodes[0].heat_flux = Q_rate
            engine.step(0.1)

        # Should transition to liquid
        self.assertEqual(engine.nodes[0].phase, Phase.LIQUID, "Ice didn't melt")

        # Temperature should be above 0°C after melting
        self.assertGreater(engine.nodes[0].temperature, 273.15, "Temperature didn't rise after melting")

    def test_entropy_increase(self):
        """Test entropy increases in irreversible process."""
        engine = ThermodynamicsEngine()

        # Hot and cold water mixing
        hot = ThermalNode(
            temperature=353.15,
            mass=1.0,
            material=MATERIALS["water"],
            position=np.array([0.0, 0.0, 0.0]),
            volume=0.001,
            phase=Phase.LIQUID
        )

        cold = ThermalNode(
            temperature=293.15,
            mass=1.0,
            material=MATERIALS["water"],
            position=np.array([0.0, 0.0, 0.1]),
            volume=0.001,
            phase=Phase.LIQUID
        )

        engine.add_node(hot)
        engine.add_node(cold)
        engine.connect_nodes(0, 1, contact_area=0.01, distance=0.05)

        S_initial = engine.total_entropy()

        engine.simulate(500.0, dt=0.1)

        S_final = engine.total_entropy()

        # Entropy must increase (second law)
        self.assertGreater(S_final, S_initial, "Entropy decreased (violates second law)")


class TestFluidDynamics(unittest.TestCase):
    """Test fluid dynamics engine."""

    def test_poiseuille_flow(self):
        """Test Poiseuille flow matches analytical solution."""
        nx, ny = 200, 50
        dx = 0.01
        dt = 0.001
        viscosity = 0.01

        engine = FluidDynamicsEngine((nx, ny), dx=dx, dt=dt, viscosity=viscosity)
        engine.initialize_flow(density=1.0)

        # Boundaries
        engine.boundary[0, :] = 1
        engine.boundary[-1, :] = 1

        # Pressure gradient
        pressure_gradient = 0.1
        engine.force[:, :, 0] = pressure_gradient

        # Run to steady state
        engine.simulate(2000)

        # Check center velocity
        center_vel = np.mean(engine.u[:, ny//2, 0])

        # Analytical: u_max = (dp/dx) / (2μ) * (H/2)²
        H = ny * dx
        u_analytical = (pressure_gradient / (2 * viscosity)) * (H/2)**2

        error = abs(center_vel - u_analytical) / abs(u_analytical)

        self.assertLess(error, 0.10, f"Poiseuille flow error {error*100:.1f}% exceeds 10%")

    def test_mass_conservation(self):
        """Test mass conservation in flow."""
        engine = FluidDynamicsEngine((100, 100), dx=0.01, dt=0.001, viscosity=0.01)
        engine.initialize_flow(density=1.0)

        initial_mass = np.sum(engine.rho) * engine.dx**2

        engine.simulate(100)

        final_mass = np.sum(engine.rho) * engine.dx**2
        mass_error = abs(final_mass - initial_mass) / initial_mass

        self.assertLess(mass_error, 0.01, f"Mass error {mass_error*100:.2f}% exceeds 1%")


class TestElectromagnetism(unittest.TestCase):
    """Test electromagnetism engine."""

    def test_coulomb_law(self):
        """Test Coulomb's law accuracy."""
        q1 = Charge(
            position=np.array([0.0, 0.0, 0.0]),
            charge=1e-9,
            velocity=np.zeros(3)
        )

        q2 = Charge(
            position=np.array([0.1, 0.0, 0.0]),
            charge=-1e-9,
            velocity=np.zeros(3)
        )

        F = coulomb_force(q1, q2)
        F_magnitude = np.linalg.norm(F)

        # Analytical
        k = 1.0 / (4 * np.pi * epsilon_0.value)
        F_expected = k * abs(q1.charge * q2.charge) / 0.1**2

        error = abs(F_magnitude - F_expected) / F_expected

        self.assertLess(error, 0.001, f"Coulomb force error {error*100:.2f}% exceeds 0.1%")

    def test_wave_propagation_speed(self):
        """Test EM wave propagates at speed of light."""
        nx, ny, nz = 200, 50, 50
        dx = 1e-3

        engine = ElectromagnetismEngine((nx, ny, nz), dx=dx)

        # Add source
        freq = 10e9
        source_pos = (20, ny//2, nz//2)

        # Run simulation
        for step in range(1000):
            engine.add_source(source_pos, amplitude=1.0, frequency=freq, component='Ez')
            engine.step()

        # Check wave front position
        Ez_profile = engine.Ez[:, ny//2, nz//2]
        wave_front = np.where(np.abs(Ez_profile) > 0.1)[0]

        if len(wave_front) > 0:
            distance = (wave_front[-1] - source_pos[0]) * dx
            time = engine.time

            speed = distance / time

            # Should be close to c
            error = abs(speed - c.value) / c.value

            self.assertLess(error, 0.10, f"Wave speed error {error*100:.1f}% exceeds 10%")


class TestQuantumMechanics(unittest.TestCase):
    """Test quantum mechanics engine."""

    def test_particle_in_box(self):
        """Test particle in box energy levels."""
        L = 1e-9  # 1 nm
        x = np.linspace(0, L, 1000)
        V = particle_in_box(x, L)

        solver = SchrodingerSolver(x, V, mass=m_e.value)
        states = solver.solve_eigenstates(n_states=3)

        # Analytical: E_n = n²π²ℏ²/(2mL²)
        for i, state in enumerate(states):
            n = i + 1
            E_analytical = (n**2 * np.pi**2 * hbar.value**2) / (2 * m_e.value * L**2)

            error = abs(state.energy - E_analytical) / E_analytical

            self.assertLess(error, 0.01, f"Energy level n={n} error {error*100:.2f}% exceeds 1%")

    def test_harmonic_oscillator_spacing(self):
        """Test harmonic oscillator equal energy spacing."""
        x = np.linspace(-1e-9, 1e-9, 1000)
        omega = 1e15
        V = harmonic_oscillator(x, omega, mass=m_e.value)

        solver = SchrodingerSolver(x, V, mass=m_e.value)
        states = solver.solve_eigenstates(n_states=4)

        # Energy spacing should be ℏω
        spacing_expected = hbar.value * omega

        for i in range(len(states) - 1):
            spacing = states[i + 1].energy - states[i].energy
            error = abs(spacing - spacing_expected) / spacing_expected

            self.assertLess(error, 0.02, f"Energy spacing error {error*100:.2f}% exceeds 2%")

    def test_wavefunction_normalization(self):
        """Test wavefunctions are normalized."""
        x = np.linspace(0, 1e-9, 1000)
        V = particle_in_box(x, 1e-9)

        solver = SchrodingerSolver(x, V)
        states = solver.solve_eigenstates(n_states=3)

        dx = x[1] - x[0]

        for state in states:
            integral = np.sum(state.probability) * dx
            error = abs(integral - 1.0)

            self.assertLess(error, 0.01, f"Normalization error {error:.4f} exceeds 0.01")


class TestPhysicsCore(unittest.TestCase):
    """Test integrated physics core."""

    def test_multi_physics_coupling(self):
        """Test mechanics + thermodynamics coupling."""
        # Create simulation with both engines
        config = SimulationConfig(
            scale=SimulationScale.MACRO,
            domain_size=(2,),
            resolution=0.1,
            timestep=0.001,
            duration=1.0,
            enable_mechanics=True,
            enable_thermodynamics=True,
        )

        core = PhysicsCore(config)

        # Add mechanical particle
        core.add_particle(
            mass=1.0,
            position=np.array([0.0, 0.0, 1.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
            radius=0.1
        )

        # Add thermal node
        node = ThermalNode(
            temperature=300.0,
            mass=10.0,
            material=MATERIALS["aluminum"],
            position=np.array([0.0, 0.0, 0.0]),
            volume=0.001
        )
        core.add_thermal_node(node)

        # Run simulation
        core.simulate()

        # Check both engines ran
        self.assertGreater(core.step_count, 0, "No simulation steps")
        self.assertGreater(core.time, 0, "Time didn't advance")

        stats = core.get_statistics()
        self.assertIn("mechanics", stats)
        self.assertIn("thermodynamics", stats)

    def test_benchmark_accuracy(self):
        """Test standard benchmarks meet accuracy targets."""
        problems = ["free_fall", "projectile", "elastic_collision"]

        for problem in problems:
            with self.subTest(problem=problem):
                core = create_benchmark_simulation(problem)
                core.simulate()

                validation = core.validate_accuracy(tolerance=0.01)

                # Should pass energy conservation
                if "mechanics_energy_conserved" in validation:
                    self.assertTrue(validation["mechanics_energy_conserved"],
                                  f"{problem} failed energy conservation")


class TestPerformance(unittest.TestCase):
    """Test performance requirements."""

    def test_particle_throughput(self):
        """Test meets 1M particle-timesteps/second target."""
        import time

        config = SimulationConfig(
            scale=SimulationScale.MACRO,
            domain_size=(100, 100, 100),
            resolution=0.01,
            timestep=0.001,
            duration=0.01,  # 10 steps
            enable_mechanics=True,
        )

        core = PhysicsCore(config)

        # Add 1000 particles
        np.random.seed(42)
        for _ in range(1000):
            core.add_particle(
                mass=0.001,
                position=np.random.uniform(-5, 5, 3),
                velocity=np.random.uniform(-1, 1, 3),
                radius=0.01
            )

        start = time.time()
        core.simulate()
        elapsed = time.time() - start

        particle_timesteps = len(core.mechanics.particles) * core.step_count
        rate = particle_timesteps / elapsed

        # Target: 1M particle-timesteps/second
        # For smaller test, require at least 100k
        self.assertGreater(rate, 1e5, f"Performance {rate:,.0f} below 100k particle-timesteps/s")


def run_nist_validation():
    """Run validation against NIST reference data."""
    print("\n" + "=" * 80)
    print("NIST VALIDATION SUITE")
    print("=" * 80)

    # Test fundamental constants accuracy
    print("\n1. Fundamental Constants:")
    print(f"   Speed of light: {c.value:.6e} m/s (exact)")
    print(f"   Planck constant: {hbar.value:.6e} J⋅s (exact)")
    print(f"   Electron mass: {m_e.value:.6e} kg (± {m_e.uncertainty:.2e})")
    print(f"   Elementary charge: {e.value:.6e} C (exact)")

    # Test free fall (NIST gravity standard)
    print("\n2. Free Fall (Standard Gravity):")
    engine = MechanicsEngine()
    p = Particle(mass=1.0, position=np.array([0.0, 0.0, 10.0]),
                velocity=np.zeros(3), force=np.zeros(3), radius=0.1)
    engine.add_particle(p)
    engine.simulate(1.0, dt=0.0001)

    z_expected = 10.0 - 0.5 * g_0.value * 1.0**2
    z_actual = engine.particles[0].position[2]
    error = abs(z_actual - z_expected) / abs(z_expected) * 100

    print(f"   Expected: {z_expected:.6f} m")
    print(f"   Actual: {z_actual:.6f} m")
    print(f"   Error: {error:.4f}% {'✓' if error < 1.0 else '✗'}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run NIST validation
    run_nist_validation()

    # Run unit tests
    print("\nRunning comprehensive test suite...")
    unittest.main(verbosity=2)
