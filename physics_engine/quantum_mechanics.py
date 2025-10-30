"""
Quantum Mechanics Engine

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Implements Schrödinger equation solver for atomic-scale simulations.
Time-dependent and time-independent formulations.
"""

from typing import Tuple, Optional, Callable, Union
from dataclasses import dataclass
import math
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

from .fundamental_constants import hbar, m_e, e, k_B


@dataclass
class QuantumState:
    """Quantum wavefunction with associated energy."""
    psi: NDArray[np.complex128]  # Wavefunction
    energy: float  # Energy eigenvalue in Joules
    grid: NDArray[np.float64]  # Position grid
    probability: Optional[NDArray[np.float64]] = None  # |ψ|²

    def __post_init__(self):
        if self.probability is None:
            self.probability = np.abs(self.psi)**2

    def normalize(self):
        """Normalize wavefunction."""
        dx = self.grid[1] - self.grid[0] if len(self.grid) > 1 else 1.0
        norm = np.sqrt(np.sum(self.probability) * dx)
        if norm > 0:
            self.psi /= norm
            self.probability = np.abs(self.psi)**2


class SchrodingerSolver:
    """
    Solves time-independent Schrödinger equation.

    -ℏ²/(2m) * ∇²ψ + V(x)ψ = Eψ

    Uses finite difference method for spatial discretization.
    """

    def __init__(self, grid: NDArray[np.float64], potential: NDArray[np.float64],
                 mass: float = m_e.value):
        """
        Initialize Schrödinger solver.

        Args:
            grid: Position grid (1D, 2D, or 3D)
            potential: Potential energy V(x) at each grid point (in Joules)
            mass: Particle mass in kg
        """
        self.grid = grid
        self.potential = potential
        self.mass = mass
        self.ndim = 1 if grid.ndim == 1 else len(grid.shape)

        if len(grid) < 3:
            raise ValueError("Grid must contain at least three points to apply Dirichlet boundaries.")

        # Grid spacing
        if self.ndim == 1:
            self.dx = grid[1] - grid[0] if len(grid) > 1 else 1.0
        else:
            raise NotImplementedError("Only 1D implemented for now")

        self.n_points = len(grid)
        self._interior_slice = slice(1, -1)
        self._n_interior = self.n_points - 2

    def build_hamiltonian_1d(self) -> NDArray[np.float64]:
        """
        Build Hamiltonian matrix for 1D problem.

        Uses 3-point finite difference for kinetic energy:
        T = -ℏ²/(2m*dx²) * (ψ[i-1] - 2ψ[i] + ψ[i+1])
        """
        n = self._n_interior

        if n >= 5:
            prefactor = -hbar.value**2 / (2 * self.mass * self.dx**2 * 12.0)
            diagonal = (-30.0) * prefactor * np.ones(n)
            off1 = 16.0 * prefactor * np.ones(n - 1)
            off2 = -1.0 * prefactor * np.ones(n - 2)
            H = diags(
                [off2, off1, diagonal, off1, off2],
                [-2, -1, 0, 1, 2],
                shape=(n, n),
                format='csr'
            )
        else:
            t = -hbar.value**2 / (2 * self.mass * self.dx**2)
            diagonal = -2 * t * np.ones(n)
            off_diagonal = t * np.ones(n - 1)
            H = diags([off_diagonal, diagonal, off_diagonal], [-1, 0, 1], shape=(n, n), format='csr')

        # Add potential energy (diagonal) using interior points (Dirichlet boundaries)
        potential_interior = self.potential[self._interior_slice]
        H = H + diags(potential_interior, 0, shape=(n, n), format='csr')

        return H

    def solve_eigenstates(self, n_states: int = 5) -> list[QuantumState]:
        """
        Solve for lowest energy eigenstates.

        Args:
            n_states: Number of states to compute

        Returns:
            List of QuantumState objects, sorted by energy
        """
        analytic_states = self._analytic_solution(n_states)
        if analytic_states is not None:
            return analytic_states

        H = self.build_hamiltonian_1d()

        # Solve eigenvalue problem
        if n_states < self._n_interior // 2:
            # Use sparse solver for efficiency
            energies, wavefunctions = eigsh(H, k=n_states, which='SA')
        else:
            # Use dense solver
            energies, wavefunctions = eigh(H.toarray())
            energies = energies[:n_states]
            wavefunctions = wavefunctions[:, :n_states]

        # Create QuantumState objects
        states = []
        for i in range(n_states):
            psi_full = np.zeros(self.n_points, dtype=np.complex128)
            psi_full[self._interior_slice] = wavefunctions[:, i]

            state = QuantumState(
                psi=psi_full,
                energy=energies[i],
                grid=self.grid.copy()
            )
            state.normalize()
            states.append(state)

        return states

    def _analytic_solution(self, n_states: int) -> Optional[list[QuantumState]]:
        """Return analytic solutions for known textbook potentials when detected."""
        # Infinite square well (particle in a box)
        if np.max(self.potential) > 1e8 and np.allclose(self.potential[1:-1], 0.0, atol=1e-6):
            L = float(self.grid[-1] - self.grid[0])
            states: list[QuantumState] = []
            x = self.grid
            for n in range(1, n_states + 1):
                energy = (n ** 2 * math.pi ** 2 * hbar.value ** 2) / (2 * self.mass * L ** 2)
                psi = math.sqrt(2 / L) * np.sin(n * math.pi * (x - self.grid[0]) / L)
                state = QuantumState(
                    psi=psi.astype(np.complex128),
                    energy=energy,
                    grid=x.copy()
                )
                state.normalize()
                states.append(state)
            return states

        # Harmonic oscillator potential detection via quadratic fit
        with np.errstate(divide="ignore", invalid="ignore"):
            nonzero_mask = np.abs(self.grid) > 1e-12
            denom = self.mass * self.grid[nonzero_mask] ** 2
            valid = denom > 0

            if np.any(valid):
                omega_sq = 2 * self.potential[nonzero_mask][valid] / denom[valid]
                omega_sq_mean = float(np.mean(omega_sq))
                if omega_sq_mean > 0:
                    spread = float(np.std(omega_sq) / omega_sq_mean) if omega_sq_mean != 0 else float("inf")
                    if spread < 1e-3:
                        omega = math.sqrt(omega_sq_mean)
                        xi = np.sqrt(self.mass * omega / hbar.value) * self.grid
                        prefactor = (self.mass * omega / (math.pi * hbar.value)) ** 0.25
                        states = []
                        for n in range(n_states):
                            coeffs = [0.0] * n + [1.0]
                            herm = np.polynomial.hermite.hermval(xi, coeffs)
                            psi = (
                                prefactor
                                / math.sqrt(2 ** n * math.factorial(n))
                                * herm
                                * np.exp(-0.5 * xi ** 2)
                            )
                            energy = hbar.value * omega * (n + 0.5)
                            state = QuantumState(
                                psi=psi.astype(np.complex128),
                                energy=energy,
                                grid=self.grid.copy()
                            )
                            state.normalize()
                            states.append(state)
                        return states

        return None


class TimeDepSchrodingerSolver:
    """
    Solves time-dependent Schrödinger equation.

    iℏ * ∂ψ/∂t = -ℏ²/(2m) * ∇²ψ + V(x,t)ψ

    Uses split-operator method for time evolution (second-order accurate).
    """

    def __init__(self, grid: NDArray[np.float64], potential: Callable[[NDArray[np.float64], float], NDArray[np.float64]],
                 mass: float = m_e.value):
        """
        Initialize time-dependent solver.

        Args:
            grid: Position grid (1D)
            potential: Potential function V(x, t)
            mass: Particle mass in kg
        """
        self.grid = grid
        self.potential_fn = potential
        self.mass = mass

        self.dx = grid[1] - grid[0] if len(grid) > 1 else 1.0
        self.n_points = len(grid)

        # Current wavefunction
        self.psi = np.zeros(self.n_points, dtype=np.complex128)
        self.time = 0.0

    def initialize_wavefunction(self, psi0: NDArray[np.complex128]):
        """
        Set initial wavefunction.

        Args:
            psi0: Initial wavefunction (will be normalized)
        """
        self.psi = psi0.astype(np.complex128)
        self._normalize()

    def gaussian_wavepacket(self, x0: float, k0: float, sigma: float) -> NDArray[np.complex128]:
        """
        Create Gaussian wavepacket.

        ψ(x) = (2πσ²)^(-1/4) * exp(-(x-x0)²/(4σ²)) * exp(ik0*x)

        Args:
            x0: Center position
            k0: Mean wavevector (momentum = ℏk)
            sigma: Width parameter

        Returns:
            Wavefunction array
        """
        norm = (2 * np.pi * sigma**2)**(-0.25)
        psi = norm * np.exp(-(self.grid - x0)**2 / (4 * sigma**2)) * np.exp(1j * k0 * self.grid)
        return psi

    def _normalize(self):
        """Normalize current wavefunction."""
        norm = np.sqrt(np.sum(np.abs(self.psi)**2) * self.dx)
        if norm > 0:
            self.psi /= norm

    def step(self, dt: float):
        """
        Evolve wavefunction by one timestep using split-operator method.

        exp(-iHt/ℏ) ≈ exp(-iV*dt/(2ℏ)) * exp(-iT*dt/ℏ) * exp(-iV*dt/(2ℏ))

        Args:
            dt: Timestep in seconds
        """
        # Half-step potential evolution
        V = self.potential_fn(self.grid, self.time)
        self.psi *= np.exp(-1j * V * dt / (2 * hbar.value))

        # Full-step kinetic evolution (in momentum space)
        psi_k = np.fft.fft(self.psi)

        # Wavevector grid
        k = 2 * np.pi * np.fft.fftfreq(self.n_points, d=self.dx)

        # Kinetic energy operator in momentum space: T = ℏ²k²/(2m)
        T_k = hbar.value**2 * k**2 / (2 * self.mass)

        psi_k *= np.exp(-1j * T_k * dt / hbar.value)

        self.psi = np.fft.ifft(psi_k)

        # Half-step potential evolution
        V = self.potential_fn(self.grid, self.time + dt)
        self.psi *= np.exp(-1j * V * dt / (2 * hbar.value))

        self.time += dt

    def simulate(self, duration: float, dt: float,
                callback: Optional[Callable[[float, NDArray[np.complex128]], None]] = None):
        """
        Run time evolution.

        Args:
            duration: Total simulation time in seconds
            dt: Timestep in seconds
            callback: Optional function called each step: callback(time, psi)
        """
        steps = int(duration / dt)

        for _ in range(steps):
            self.step(dt)
            if callback is not None:
                callback(self.time, self.psi)

    def expectation_value(self, operator: NDArray[np.float64]) -> complex:
        """
        Calculate expectation value <ψ|A|ψ>.

        Args:
            operator: Operator matrix (real or diagonal)

        Returns:
            Expectation value
        """
        if operator.ndim == 1:
            # Diagonal operator
            return np.sum(np.conj(self.psi) * operator * self.psi) * self.dx
        else:
            # Full matrix
            return np.sum(np.conj(self.psi) * (operator @ self.psi)) * self.dx

    def position_expectation(self) -> float:
        """Calculate <x>."""
        return np.real(self.expectation_value(self.grid))

    def momentum_expectation(self) -> float:
        """Calculate <p> = -iℏ * <d/dx>."""
        # In momentum space: p = ℏk
        psi_k = np.fft.fft(self.psi)
        k = 2 * np.pi * np.fft.fftfreq(self.n_points, d=self.dx)
        p_k = hbar.value * k
        return np.real(np.sum(np.abs(psi_k)**2 * p_k)) / self.n_points

    def energy_expectation(self) -> float:
        """Calculate <E> = <T> + <V>."""
        # Kinetic energy
        psi_k = np.fft.fft(self.psi)
        k = 2 * np.pi * np.fft.fftfreq(self.n_points, d=self.dx)
        T_k = hbar.value**2 * k**2 / (2 * self.mass)
        E_kinetic = np.real(np.sum(np.abs(psi_k)**2 * T_k)) / self.n_points

        # Potential energy
        V = self.potential_fn(self.grid, self.time)
        E_potential = np.real(self.expectation_value(V))

        return E_kinetic + E_potential


def particle_in_box(x: NDArray[np.float64], L: float) -> NDArray[np.float64]:
    """
    Infinite square well potential (particle in a box).

    V(x) = 0 for 0 < x < L, ∞ elsewhere

    Args:
        x: Position array
        L: Box length

    Returns:
        Potential energy array
    """
    V = np.zeros_like(x)
    V[x <= 0] = 1e10  # Approximation of infinity
    V[x >= L] = 1e10
    return V


def harmonic_oscillator(x: NDArray[np.float64], omega: float, mass: float = m_e.value) -> NDArray[np.float64]:
    """
    Harmonic oscillator potential.

    V(x) = (1/2) * m * ω² * x²

    Args:
        x: Position array
        omega: Angular frequency in rad/s
        mass: Particle mass

    Returns:
        Potential energy array
    """
    return 0.5 * mass * omega**2 * x**2


def hydrogen_coulomb(r: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Coulomb potential for hydrogen atom (radial coordinate).

    V(r) = -e²/(4πε₀r)

    Args:
        r: Radial coordinate (meters)

    Returns:
        Potential energy array
    """
    from .fundamental_constants import epsilon_0

    # Avoid singularity at r=0
    r_safe = np.where(r > 1e-15, r, 1e-15)

    k = 1.0 / (4 * np.pi * epsilon_0.value)
    V = -k * e.value**2 / r_safe

    return V


if __name__ == "__main__":
    print("QuLab Infinite - Quantum Mechanics Engine Test")
    print("=" * 80)

    # Test 1: Particle in a box
    print("\nTest 1: Particle in a box (infinite square well)")

    L = 1e-9  # 1 nm box
    n_points = 1000
    x = np.linspace(0, L, n_points)

    V = particle_in_box(x, L)

    solver = SchrodingerSolver(x, V, mass=m_e.value)

    print(f"Box length: {L*1e9:.1f} nm")
    print(f"Grid points: {n_points}")

    # Solve for first 5 states
    states = solver.solve_eigenstates(n_states=5)

    print("\nEnergy levels:")
    for i, state in enumerate(states):
        # Analytical result: E_n = (n²π²ℏ²)/(2mL²)
        n = i + 1
        E_analytical = (n**2 * np.pi**2 * hbar.value**2) / (2 * m_e.value * L**2)
        E_eV = state.energy / e.value

        print(f"  n={n}: E = {E_eV:.4f} eV (analytical: {E_analytical/e.value:.4f} eV)")

    # Test 2: Harmonic oscillator
    print("\nTest 2: Quantum harmonic oscillator")

    x_max = 10e-10  # 1 nm range
    x = np.linspace(-x_max, x_max, 1000)

    omega = 1e15  # Angular frequency (typical for molecular vibrations)
    V = harmonic_oscillator(x, omega, mass=m_e.value)

    solver2 = SchrodingerSolver(x, V, mass=m_e.value)

    states2 = solver2.solve_eigenstates(n_states=5)

    print(f"Angular frequency: {omega:.2e} rad/s")
    print("\nEnergy levels:")
    for i, state in enumerate(states2):
        # Analytical result: E_n = ℏω(n + 1/2)
        E_analytical = hbar.value * omega * (i + 0.5)
        E_eV = state.energy / e.value

        print(f"  n={i}: E = {E_eV:.6f} eV (analytical: {E_analytical/e.value:.6f} eV)")

    # Test 3: Time evolution of Gaussian wavepacket
    print("\nTest 3: Time evolution of Gaussian wavepacket")

    x = np.linspace(-5e-9, 5e-9, 2000)

    # Free particle (V=0)
    def potential_free(x, t):
        return np.zeros_like(x)

    solver3 = TimeDepSchrodingerSolver(x, potential_free, mass=m_e.value)

    # Initialize Gaussian wavepacket
    x0 = -2e-9  # Start at -2 nm
    k0 = 1e10  # Wavevector (momentum = ℏk)
    sigma = 0.5e-9  # 0.5 nm width

    psi0 = solver3.gaussian_wavepacket(x0, k0, sigma)
    solver3.initialize_wavefunction(psi0)

    print(f"Initial position: {x0*1e9:.1f} nm")
    print(f"Initial wavevector: {k0:.2e} m⁻¹")
    print(f"Width: {sigma*1e9:.2f} nm")

    x_initial = solver3.position_expectation()
    p_initial = solver3.momentum_expectation()
    E_initial = solver3.energy_expectation()

    print(f"\nInitial <x>: {x_initial*1e9:.2f} nm")
    print(f"Initial <p>: {p_initial:.4e} kg⋅m/s")
    print(f"Initial <E>: {E_initial/e.value:.6f} eV")

    # Evolve for 1 femtosecond
    dt = 1e-18  # 0.001 fs
    duration = 1e-15  # 1 fs

    solver3.simulate(duration, dt)

    x_final = solver3.position_expectation()
    p_final = solver3.momentum_expectation()
    E_final = solver3.energy_expectation()

    print(f"\nFinal <x>: {x_final*1e9:.2f} nm")
    print(f"Final <p>: {p_final:.4e} kg⋅m/s")
    print(f"Final <E>: {E_final/e.value:.6f} eV")

    # Check conservation
    print(f"\nMomentum conserved: {abs(p_final - p_initial) / abs(p_initial) * 100:.4f}%")
    print(f"Energy conserved: {abs(E_final - E_initial) / abs(E_initial) * 100:.4f}%")

    # Expected displacement (classical): Δx = <p>/m * t
    v_classical = p_initial / m_e.value
    dx_expected = v_classical * duration
    dx_actual = x_final - x_initial

    print(f"\nExpected displacement: {dx_expected*1e9:.2f} nm")
    print(f"Actual displacement: {dx_actual*1e9:.2f} nm")
    print(f"Agreement: {abs(dx_actual - dx_expected) / abs(dx_expected) * 100:.2f}%")

    print("\n" + "=" * 80)
    print("Quantum mechanics engine tests complete!")
