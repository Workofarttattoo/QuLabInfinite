"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

QUANTUM MECHANICS LAB
Advanced quantum mechanical simulations including Schrodinger equation solvers,
harmonic oscillators, hydrogen atom wavefunctions, tunneling, and perturbation theory.
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from scipy import constants, special, integrate, linalg
from scipy.sparse import diags
from typing import Tuple, Callable, Optional, List
import warnings

# Physical constants
HBAR = constants.hbar  # Reduced Planck constant (J·s)
ME = constants.m_e  # Electron mass (kg)
E_CHARGE = constants.e  # Elementary charge (C)
EPSILON_0 = constants.epsilon_0  # Vacuum permittivity (F/m)
A_BOHR = 4 * np.pi * EPSILON_0 * HBAR**2 / (ME * E_CHARGE**2)  # Bohr radius (m)
E_H = ME * E_CHARGE**4 / (32 * np.pi**2 * EPSILON_0**2 * HBAR**2)  # Hartree energy (J)


class QuantumMechanicsLab:
    """Advanced quantum mechanics simulation laboratory."""

    def __init__(self):
        """Initialize the quantum mechanics lab."""
        self.name = "Quantum Mechanics Laboratory"
        self.version = "2.0.0"

    def solve_schrodinger_1d(self, potential: Callable, x_range: Tuple[float, float],
                             n_points: int = 1000, n_states: int = 5,
                             mass: float = ME) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the time-independent Schrodinger equation in 1D.

        Args:
            potential: Potential energy function V(x)
            x_range: Tuple of (x_min, x_max)
            n_points: Number of spatial grid points
            n_states: Number of energy eigenstates to find
            mass: Particle mass (default: electron mass)

        Returns:
            Tuple of (x_grid, energies, wavefunctions)
        """
        x = np.linspace(x_range[0], x_range[1], n_points)
        dx = x[1] - x[0]

        # Construct the Hamiltonian matrix using finite differences
        kinetic = -HBAR**2 / (2 * mass * dx**2) * (
            np.diag(np.ones(n_points - 1), 1) -
            2 * np.diag(np.ones(n_points), 0) +
            np.diag(np.ones(n_points - 1), -1)
        )

        # Add potential energy
        V = np.diag(potential(x))
        H = kinetic + V

        # Solve eigenvalue problem
        eigenvalues, eigenvectors = linalg.eigh(H)

        # Select lowest n_states
        energies = eigenvalues[:n_states]
        wavefunctions = eigenvectors[:, :n_states].T

        # Normalize wavefunctions
        for i in range(n_states):
            norm = np.sqrt(np.trapz(np.abs(wavefunctions[i])**2, x))
            wavefunctions[i] /= norm

        return x, energies, wavefunctions

    def solve_schrodinger_2d(self, potential: Callable, x_range: Tuple[float, float],
                             y_range: Tuple[float, float], n_points: int = 50,
                             n_states: int = 3, mass: float = ME) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the time-independent Schrodinger equation in 2D.

        Args:
            potential: Potential energy function V(x, y)
            x_range: Tuple of (x_min, x_max)
            y_range: Tuple of (y_min, y_max)
            n_points: Number of grid points in each dimension
            n_states: Number of energy eigenstates to find
            mass: Particle mass

        Returns:
            Tuple of ((X, Y) meshgrid, energies, wavefunctions)
        """
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = np.linspace(y_range[0], y_range[1], n_points)
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        X, Y = np.meshgrid(x, y)

        # Build 2D Laplacian using Kronecker products
        Ix = np.eye(n_points)
        Iy = np.eye(n_points)

        # 1D second derivative operators
        Dx = (np.diag(np.ones(n_points - 1), 1) -
              2 * np.diag(np.ones(n_points), 0) +
              np.diag(np.ones(n_points - 1), -1)) / dx**2

        Dy = (np.diag(np.ones(n_points - 1), 1) -
              2 * np.diag(np.ones(n_points), 0) +
              np.diag(np.ones(n_points - 1), -1)) / dy**2

        # 2D Laplacian
        L2D = np.kron(Ix, Dy) + np.kron(Dx, Iy)

        # Kinetic energy operator
        T = -HBAR**2 / (2 * mass) * L2D

        # Potential energy
        V_flat = potential(X, Y).flatten()
        V = np.diag(V_flat)

        # Total Hamiltonian
        H = T + V

        # Solve for lowest eigenvalues
        eigenvalues, eigenvectors = linalg.eigh(H)

        energies = eigenvalues[:n_states]
        wavefunctions = eigenvectors[:, :n_states].reshape(n_points, n_points, n_states)

        return (X, Y), energies, wavefunctions

    def harmonic_oscillator_eigenstates(self, n_max: int = 10, omega: float = 1e15,
                                       mass: float = ME, x_range: Tuple[float, float] = (-5e-9, 5e-9),
                                       n_points: int = 500) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate quantum harmonic oscillator eigenstates analytically.

        Args:
            n_max: Maximum quantum number
            omega: Angular frequency (rad/s)
            mass: Particle mass
            x_range: Spatial range for wavefunction evaluation
            n_points: Number of spatial points

        Returns:
            Tuple of (x_grid, energies, wavefunctions)
        """
        x = np.linspace(x_range[0], x_range[1], n_points)

        # Characteristic length scale
        x0 = np.sqrt(HBAR / (mass * omega))

        # Dimensionless position
        xi = x / x0

        energies = np.zeros(n_max + 1)
        wavefunctions = np.zeros((n_max + 1, n_points))

        for n in range(n_max + 1):
            # Energy eigenvalue
            energies[n] = HBAR * omega * (n + 0.5)

            # Hermite polynomial
            Hn = special.hermite(n)

            # Wavefunction
            psi = (1 / (2**n * special.factorial(n) * np.sqrt(np.pi * x0)))**0.5 * \
                  np.exp(-xi**2 / 2) * Hn(xi)

            wavefunctions[n] = psi

        return x, energies, wavefunctions

    def hydrogen_wavefunction(self, n: int, l: int, m: int,
                            r_max: float = 20 * A_BOHR,
                            n_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate hydrogen atom wavefunctions.

        Args:
            n: Principal quantum number (n >= 1)
            l: Orbital angular momentum quantum number (0 <= l < n)
            m: Magnetic quantum number (-l <= m <= l)
            r_max: Maximum radius
            n_points: Number of grid points in each dimension

        Returns:
            Tuple of ((R, Theta, Phi) grids, wavefunction)
        """
        if not (n >= 1 and 0 <= l < n and -l <= m <= l):
            raise ValueError("Invalid quantum numbers")

        # Create spherical coordinate grids
        r = np.linspace(0, r_max, n_points)
        theta = np.linspace(0, np.pi, n_points)
        phi = np.linspace(0, 2 * np.pi, n_points)

        R, Theta, Phi = np.meshgrid(r, theta, phi, indexing='ij')

        # Radial part using associated Laguerre polynomials
        rho = 2 * R / (n * A_BOHR)

        # Normalization factor for radial part
        norm_R = np.sqrt((2 / (n * A_BOHR))**3 *
                        special.factorial(n - l - 1) /
                        (2 * n * special.factorial(n + l)**3))

        # Associated Laguerre polynomial
        L = special.genlaguerre(n - l - 1, 2 * l + 1)

        # Radial wavefunction
        R_nl = norm_R * np.exp(-rho / 2) * rho**l * L(rho)

        # Angular part (spherical harmonics)
        Y_lm = special.sph_harm(m, l, Phi, Theta)

        # Total wavefunction
        psi = R_nl * Y_lm

        return (R, Theta, Phi), psi

    def quantum_tunneling_probability(self, barrier_height: float, barrier_width: float,
                                     particle_energy: float, mass: float = ME) -> float:
        """
        Calculate transmission probability through a rectangular potential barrier.

        Args:
            barrier_height: Height of potential barrier (J)
            barrier_width: Width of barrier (m)
            particle_energy: Energy of incident particle (J)
            mass: Particle mass

        Returns:
            Transmission probability (0 to 1)
        """
        if particle_energy >= barrier_height:
            # Classical case - particle goes over barrier
            return 1.0

        # Wave vector outside barrier
        k1 = np.sqrt(2 * mass * particle_energy) / HBAR

        # Decay constant inside barrier
        k2 = np.sqrt(2 * mass * (barrier_height - particle_energy)) / HBAR

        # Transmission coefficient using WKB approximation
        if k2 * barrier_width >> 1:
            # Thick barrier approximation
            T = 16 * (particle_energy / barrier_height) * \
                (1 - particle_energy / barrier_height) * \
                np.exp(-2 * k2 * barrier_width)
        else:
            # Exact solution
            sinh_term = np.sinh(k2 * barrier_width)**2
            T = 1 / (1 + (k2**2 + k1**2)**2 * sinh_term / (4 * k1**2 * k2**2))

        return T

    def perturbation_theory_first_order(self, H0: np.ndarray, V: np.ndarray,
                                       n_states: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate first-order perturbation theory corrections.

        Args:
            H0: Unperturbed Hamiltonian matrix
            V: Perturbation potential matrix
            n_states: Number of states to calculate

        Returns:
            Tuple of (energy corrections, state corrections)
        """
        # Solve unperturbed problem
        E0, psi0 = linalg.eigh(H0)

        # Select lowest n_states
        E0 = E0[:n_states]
        psi0 = psi0[:, :n_states]

        # First-order energy corrections
        E1 = np.zeros(n_states)
        for n in range(n_states):
            E1[n] = np.real(psi0[:, n].conj() @ V @ psi0[:, n])

        # First-order state corrections
        psi1 = np.zeros_like(psi0, dtype=complex)
        for n in range(n_states):
            for m in range(len(H0)):
                if m != n and m < n_states:
                    if abs(E0[n] - E0[m]) > 1e-10:  # Avoid division by zero
                        V_mn = psi0[:, m].conj() @ V @ psi0[:, n]
                        psi1[:, n] += V_mn / (E0[n] - E0[m]) * psi0[:, m]

        return E1, psi1

    def time_evolution_schrodinger(self, psi0: np.ndarray, H: np.ndarray,
                                  t_max: float, n_steps: int = 1000) -> np.ndarray:
        """
        Evolve a quantum state in time using the time-dependent Schrodinger equation.

        Args:
            psi0: Initial state vector
            H: Hamiltonian matrix
            t_max: Maximum evolution time (s)
            n_steps: Number of time steps

        Returns:
            Array of state vectors at each time step
        """
        t = np.linspace(0, t_max, n_steps)
        dt = t[1] - t[0]

        # Time evolution operator (using matrix exponential)
        U = linalg.expm(-1j * H * dt / HBAR)

        # Store states at each time
        states = np.zeros((n_steps, len(psi0)), dtype=complex)
        states[0] = psi0

        # Evolve state
        for i in range(1, n_steps):
            states[i] = U @ states[i-1]
            # Normalize to account for numerical errors
            states[i] /= np.linalg.norm(states[i])

        return states

    def expectation_value(self, psi: np.ndarray, operator: np.ndarray) -> float:
        """
        Calculate expectation value of an operator in a given state.

        Args:
            psi: Quantum state vector
            operator: Operator matrix

        Returns:
            Expectation value <psi|operator|psi>
        """
        return np.real(psi.conj() @ operator @ psi)

    def uncertainty_relation(self, psi: np.ndarray, A: np.ndarray,
                           B: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate uncertainty relation for two operators.

        Args:
            psi: Quantum state vector
            A: First operator
            B: Second operator

        Returns:
            Tuple of (ΔA, ΔB, ΔA·ΔB)
        """
        # Expectation values
        A_exp = self.expectation_value(psi, A)
        B_exp = self.expectation_value(psi, B)

        # Variances
        A2_exp = self.expectation_value(psi, A @ A)
        B2_exp = self.expectation_value(psi, B @ B)

        delta_A = np.sqrt(A2_exp - A_exp**2)
        delta_B = np.sqrt(B2_exp - B_exp**2)

        return delta_A, delta_B, delta_A * delta_B

    def coherent_state(self, alpha: complex, n_max: int = 50) -> np.ndarray:
        """
        Create a coherent state (quantum harmonic oscillator).

        Args:
            alpha: Complex amplitude
            n_max: Maximum Fock state number

        Returns:
            Coherent state vector in Fock basis
        """
        state = np.zeros(n_max + 1, dtype=complex)

        for n in range(n_max + 1):
            state[n] = np.exp(-0.5 * abs(alpha)**2) * \
                      alpha**n / np.sqrt(special.factorial(n))

        return state / np.linalg.norm(state)

    def berry_phase(self, H_func: Callable, params_path: np.ndarray,
                   state_idx: int = 0) -> float:
        """
        Calculate geometric (Berry) phase for adiabatic evolution.

        Args:
            H_func: Function that returns Hamiltonian for given parameter
            params_path: Array of parameters defining closed path
            state_idx: Which eigenstate to track (0 = ground state)

        Returns:
            Berry phase (radians)
        """
        n_points = len(params_path)
        phase = 0.0

        # Get initial eigenstate
        H0 = H_func(params_path[0])
        _, psi0 = linalg.eigh(H0)
        psi_prev = psi0[:, state_idx]

        # Evolve around closed path
        for i in range(1, n_points):
            H = H_func(params_path[i])
            _, psi = linalg.eigh(H)
            psi_curr = psi[:, state_idx]

            # Ensure continuity of phase
            overlap = np.vdot(psi_prev, psi_curr)
            psi_curr *= np.exp(1j * np.angle(overlap))

            # Accumulate Berry connection
            if i < n_points - 1:
                phase += np.angle(np.vdot(psi_prev, psi_curr))

            psi_prev = psi_curr

        # Close the loop
        phase += np.angle(np.vdot(psi_prev, psi0[:, state_idx]))

        return phase


def run_demo():
    """Demonstrate quantum mechanics calculations."""
    lab = QuantumMechanicsLab()
    print(f"Initializing {lab.name} v{lab.version}")
    print("=" * 60)

    # 1. Solve 1D Schrodinger equation for particle in a box
    print("\n1. Particle in a Box (1D):")
    L = 1e-9  # 1 nm box
    potential_box = lambda x: np.where((x > 0) & (x < L), 0, 1e10)
    x, energies, wavefunctions = lab.solve_schrodinger_1d(
        potential_box, (0, L), n_points=500, n_states=3
    )
    print(f"   First 3 energy levels (eV):")
    for i, E in enumerate(energies):
        print(f"   n={i+1}: {E / constants.eV:.3f} eV")

    # 2. Harmonic oscillator
    print("\n2. Quantum Harmonic Oscillator:")
    x_ho, E_ho, psi_ho = lab.harmonic_oscillator_eigenstates(
        n_max=5, omega=1e15, x_range=(-5e-9, 5e-9)
    )
    print(f"   First 6 energy levels (eV):")
    for i, E in enumerate(E_ho):
        print(f"   n={i}: {E / constants.eV:.6f} eV")

    # 3. Hydrogen atom
    print("\n3. Hydrogen Atom Wavefunctions:")
    print("   Calculating 2p orbital (n=2, l=1, m=0)...")
    coords, psi_H = lab.hydrogen_wavefunction(n=2, l=1, m=0, n_points=30)
    print(f"   Wavefunction shape: {psi_H.shape}")
    print(f"   Maximum probability density: {np.max(np.abs(psi_H)**2):.2e}")

    # 4. Quantum tunneling
    print("\n4. Quantum Tunneling:")
    barrier_height = 1.0 * constants.eV  # 1 eV barrier
    barrier_width = 1e-9  # 1 nm wide
    particle_energy = 0.5 * constants.eV  # 0.5 eV particle
    T = lab.quantum_tunneling_probability(
        barrier_height, barrier_width, particle_energy
    )
    print(f"   Barrier: {barrier_height/constants.eV:.1f} eV, {barrier_width*1e9:.1f} nm")
    print(f"   Particle energy: {particle_energy/constants.eV:.1f} eV")
    print(f"   Transmission probability: {T:.2e}")

    # 5. Perturbation theory
    print("\n5. Perturbation Theory:")
    # Unperturbed harmonic oscillator
    n_basis = 10
    H0 = np.diag(np.arange(n_basis) + 0.5)
    # Small perturbation
    V = 0.1 * np.random.randn(n_basis, n_basis)
    V = (V + V.T) / 2  # Make Hermitian
    E1, psi1 = lab.perturbation_theory_first_order(H0, V, n_states=3)
    print(f"   First-order energy corrections:")
    for i, dE in enumerate(E1):
        print(f"   State {i}: ΔE = {dE:.4f}")

    # 6. Coherent state
    print("\n6. Coherent State:")
    alpha = 2.0 + 1.0j
    coherent = lab.coherent_state(alpha, n_max=20)
    print(f"   α = {alpha}")
    print(f"   Average photon number: {np.sum(np.arange(len(coherent)) * np.abs(coherent)**2):.2f}")

    print("\n" + "=" * 60)
    print("Quantum Mechanics Lab demonstration complete!")


if __name__ == "__main__":
    run_demo()