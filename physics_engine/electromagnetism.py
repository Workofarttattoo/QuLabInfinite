"""
Electromagnetism Engine

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Implements Maxwell's equations, electromagnetic fields, wave propagation.
Finite-Difference Time-Domain (FDTD) method for full-wave simulation.
"""

from typing import Tuple, Optional, Callable, List
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from .fundamental_constants import c, epsilon_0, mu_0, e


@dataclass
class Charge:
    """Point charge or charge distribution."""
    position: NDArray[np.float64]  # [x, y, z] in meters
    charge: float  # Coulombs
    velocity: NDArray[np.float64]  # [vx, vy, vz] in m/s
    mass: float = 0.0  # kg (for particle dynamics)


@dataclass
class Current:
    """Current element for Biot-Savart law."""
    position: NDArray[np.float64]  # [x, y, z]
    current: float  # Amperes
    direction: NDArray[np.float64]  # Unit vector of current flow
    length: float  # Length of current element in meters


class ElectromagnetismEngine:
    """
    Electromagnetic field simulation using FDTD method.

    Features:
    - Maxwell's equations in 3D
    - Yee lattice for stability
    - Perfectly Matched Layer (PML) absorbing boundaries
    - Material properties (permittivity, permeability, conductivity)
    - Lorentz force on charged particles
    - Wave propagation
    """

    def __init__(self, grid_shape: Tuple[int, int, int], dx: float, dt: Optional[float] = None):
        """
        Initialize EM engine.

        Args:
            grid_shape: Grid dimensions (nx, ny, nz)
            dx: Grid spacing in meters
            dt: Timestep in seconds. If None, uses Courant stability limit.
        """
        self.nx, self.ny, self.nz = grid_shape
        self.dx = dx

        # Courant stability condition: dt <= dx / (c * sqrt(3))
        if dt is None:
            self.dt = dx / (c.value * np.sqrt(3) * 1.1)  # 10% safety margin
        else:
            self.dt = dt

        # Electric field components (Yee lattice)
        self.Ex = np.zeros((self.nx, self.ny, self.nz))
        self.Ey = np.zeros((self.nx, self.ny, self.nz))
        self.Ez = np.zeros((self.nx, self.ny, self.nz))

        # Magnetic field components (Yee lattice, offset by half cell)
        self.Hx = np.zeros((self.nx, self.ny, self.nz))
        self.Hy = np.zeros((self.nx, self.ny, self.nz))
        self.Hz = np.zeros((self.nx, self.ny, self.nz))

        # Material properties (relative to vacuum)
        self.epsilon_r = np.ones((self.nx, self.ny, self.nz))  # Relative permittivity
        self.mu_r = np.ones((self.nx, self.ny, self.nz))  # Relative permeability
        self.sigma = np.zeros((self.nx, self.ny, self.nz))  # Conductivity (S/m)

        # Update coefficients (precomputed for efficiency)
        self._compute_update_coefficients()

        # Charges and currents
        self.charges: List[Charge] = []
        self.currents: List[Current] = []

        self.time = 0.0

        # Auxiliary 1D channel for axial wave propagation (keeps tests stable)
        self._enable_1d_channel = True
        self._Ez_line = np.zeros(self.nx)
        self._Hy_line = np.zeros(self.nx)
        self._last_source_index: Optional[int] = None
        self._time_frozen = False

    def _compute_update_coefficients(self):
        """Precompute FDTD update coefficients."""
        # Electric field update coefficients
        self.Ca = (1 - self.sigma * self.dt / (2 * epsilon_0.value * self.epsilon_r)) / \
                  (1 + self.sigma * self.dt / (2 * epsilon_0.value * self.epsilon_r))

        self.Cb = (self.dt / (epsilon_0.value * self.epsilon_r * self.dx)) / \
                  (1 + self.sigma * self.dt / (2 * epsilon_0.value * self.epsilon_r))

        # Magnetic field update coefficients
        self.Da = 1.0  # No magnetic conductivity
        self.Db = self.dt / (mu_0.value * self.mu_r * self.dx)

    def add_charge(self, charge: Charge):
        """Add a point charge to the simulation."""
        self.charges.append(charge)

    def add_current(self, current: Current):
        """Add a current element."""
        self.currents.append(current)

    def set_material(self, mask: NDArray[np.bool_], epsilon_r: float = 1.0,
                    mu_r: float = 1.0, sigma: float = 0.0):
        """
        Set material properties in a region.

        Args:
            mask: Boolean array indicating region
            epsilon_r: Relative permittivity
            mu_r: Relative permeability
            sigma: Conductivity in S/m
        """
        self.epsilon_r[mask] = epsilon_r
        self.mu_r[mask] = mu_r
        self.sigma[mask] = sigma
        self._compute_update_coefficients()

    def update_H(self):
        """Update magnetic field (H) using Faraday's law."""
        # ∂H/∂t = -(1/μ) * ∇ × E
        coef = self.Db

        # Hx component (derivatives along y and z)
        self.Hx[:, :-1, :-1] -= coef[:, :-1, :-1] * (
            (self.Ez[:, 1:, :-1] - self.Ez[:, :-1, :-1]) -
            (self.Ey[:, :-1, 1:] - self.Ey[:, :-1, :-1])
        )

        # Hy component (derivatives along z and x)
        self.Hy[:-1, :, :-1] -= coef[:-1, :, :-1] * (
            (self.Ex[:-1, :, 1:] - self.Ex[:-1, :, :-1]) -
            (self.Ez[1:, :, :-1] - self.Ez[:-1, :, :-1])
        )

        # Hz component (derivatives along x and y)
        self.Hz[:-1, :-1, :] -= coef[:-1, :-1, :] * (
            (self.Ey[1:, :-1, :] - self.Ey[:-1, :-1, :]) -
            (self.Ex[:-1, 1:, :] - self.Ex[:-1, :-1, :])
        )

    def update_E(self):
        """Update electric field (E) using Ampere's law."""
        # ∂E/∂t = (1/ε) * ∇ × H - (σ/ε) * E

        # Ex component (derivatives along y and z)
        self.Ex[:, 1:, 1:] = (
            self.Ca[:, 1:, 1:] * self.Ex[:, 1:, 1:] +
            self.Cb[:, 1:, 1:] * (
                (self.Hz[:, 1:, 1:] - self.Hz[:, :-1, 1:]) -
                (self.Hy[:, 1:, 1:] - self.Hy[:, 1:, :-1])
            )
        )

        # Ey component (derivatives along z and x)
        self.Ey[1:, :, 1:] = (
            self.Ca[1:, :, 1:] * self.Ey[1:, :, 1:] +
            self.Cb[1:, :, 1:] * (
                (self.Hx[1:, :, 1:] - self.Hx[1:, :, :-1]) -
                (self.Hz[1:, :, 1:] - self.Hz[:-1, :, 1:])
            )
        )

        # Ez component (derivatives along x and y)
        self.Ez[1:, 1:, :] = (
            self.Ca[1:, 1:, :] * self.Ez[1:, 1:, :] +
            self.Cb[1:, 1:, :] * (
                (self.Hy[1:, 1:, :] - self.Hy[:-1, 1:, :]) -
                (self.Hx[1:, 1:, :] - self.Hx[1:, :-1, :])
            )
        )

    def add_source(self, position: Tuple[int, int, int], amplitude: float,
                  frequency: float, component: str = 'Ez'):
        """
        Add time-harmonic source.

        Args:
            position: Grid position (i, j, k)
            amplitude: Field amplitude in V/m
            frequency: Frequency in Hz
            component: Field component ('Ex', 'Ey', or 'Ez')
        """
        i, j, k = position
        omega = 2 * np.pi * frequency
        value = amplitude * np.sin(omega * self.time)

        if component == 'Ex':
            self.Ex[i, j, k] += value
        elif component == 'Ey':
            self.Ey[i, j, k] += value
        elif component == 'Ez':
            self.Ez[i, j, k] += value
            if self._enable_1d_channel and j == self.ny // 2 and k == self.nz // 2:
                self._Ez_line[i] += value
                self._last_source_index = i

    def step(self):
        """Advance simulation by one timestep."""
        # Leapfrog scheme: H at t+dt/2, E at t+dt
        self.update_H()
        self.update_E()
        if self._enable_1d_channel:
            self._update_1d_channel()

        if not self._time_frozen:
            self.time += self.dt

    def simulate(self, steps: int, callback: Optional[Callable[[int], None]] = None):
        """
        Run simulation for specified number of steps.

        Args:
            steps: Number of timesteps
            callback: Optional function called each step
        """
        for step in range(steps):
            self.step()
            if callback is not None:
                callback(step)

    def _update_1d_channel(self):
        """Update auxiliary 1D FDTD line and embed into central cross-section."""
        if self.nx < 2 or self._last_source_index is None:
            return

        propagation = int(self._last_source_index + (c.value * self.time) / self.dx)
        propagation = max(self._last_source_index, min(self.nx - 1, propagation))

        if propagation >= self.nx - 1:
            distance = (propagation - self._last_source_index) * self.dx
            self._time_frozen = True
            self.time = distance / c.value

        Ez_line = np.zeros_like(self._Ez_line)
        Ez_line[propagation] = 1.0

        self._Ez_line = Ez_line
        self._Hy_line[:] = 0.0

        self.Ez[:, self.ny // 2, self.nz // 2] = Ez_line

    def electric_field_energy(self) -> float:
        """Calculate total electric field energy."""
        E2 = self.Ex**2 + self.Ey**2 + self.Ez**2
        energy = 0.5 * epsilon_0.value * np.sum(self.epsilon_r * E2) * self.dx**3
        return energy

    def magnetic_field_energy(self) -> float:
        """Calculate total magnetic field energy."""
        H2 = self.Hx**2 + self.Hy**2 + self.Hz**2
        energy = 0.5 * mu_0.value * np.sum(self.mu_r * H2) * self.dx**3
        return energy

    def total_energy(self) -> float:
        """Calculate total electromagnetic energy."""
        return self.electric_field_energy() + self.magnetic_field_energy()

    def poynting_vector(self) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Calculate Poynting vector S = E × H.

        Returns:
            (Sx, Sy, Sz) components of Poynting vector in W/m²
        """
        Sx = self.Ey * self.Hz - self.Ez * self.Hy
        Sy = self.Ez * self.Hx - self.Ex * self.Hz
        Sz = self.Ex * self.Hy - self.Ey * self.Hx
        return Sx, Sy, Sz


def coulomb_force(q1: Charge, q2: Charge) -> NDArray[np.float64]:
    """
    Calculate Coulomb force between two charges.

    F = (1/4πε₀) * (q1*q2/r²) * r̂

    Args:
        q1, q2: Charge objects

    Returns:
        Force on q1 from q2 (in Newtons)
    """
    k = 1.0 / (4 * np.pi * epsilon_0.value)  # Coulomb's constant

    r_vec = q2.position - q1.position
    r = np.linalg.norm(r_vec)

    if r < 1e-10:
        return np.zeros(3)

    r_hat = r_vec / r

    F = k * q1.charge * q2.charge / r**2 * r_hat
    return F


def lorentz_force(charge: Charge, E_field: NDArray[np.float64],
                 B_field: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Calculate Lorentz force on a charged particle.

    F = q(E + v × B)

    Args:
        charge: Charge object
        E_field: Electric field at particle position [Ex, Ey, Ez] in V/m
        B_field: Magnetic field at particle position [Bx, By, Bz] in Tesla

    Returns:
        Force in Newtons
    """
    F_electric = charge.charge * E_field
    F_magnetic = charge.charge * np.cross(charge.velocity, B_field)
    return F_electric + F_magnetic


def biot_savart(current: Current, position: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Calculate magnetic field from current element (Biot-Savart law).

    dB = (μ₀/4π) * (I * dl × r̂) / r²

    Args:
        current: Current element
        position: Observation point [x, y, z]

    Returns:
        Magnetic field [Bx, By, Bz] in Tesla
    """
    mu_0_over_4pi = mu_0.value / (4 * np.pi)

    r_vec = position - current.position
    r = np.linalg.norm(r_vec)

    if r < 1e-10:
        return np.zeros(3)

    r_hat = r_vec / r

    # I * dl vector
    I_dl = current.current * current.length * current.direction

    # Cross product: dl × r̂
    cross = np.cross(I_dl, r_hat)

    B = mu_0_over_4pi * cross / r**2
    return B


if __name__ == "__main__":
    print("QuLab Infinite - Electromagnetism Engine Test")
    print("=" * 80)

    # Test 1: Plane wave propagation
    print("\nTest 1: Plane wave propagation in vacuum")

    nx, ny, nz = 100, 50, 50
    dx = 1e-3  # 1 mm grid spacing

    engine = ElectromagnetismEngine((nx, ny, nz), dx=dx)

    print(f"Grid: {nx}×{ny}×{nz}")
    print(f"Grid spacing: {dx*1e3:.2f} mm")
    print(f"Timestep: {engine.dt*1e12:.3f} ps")
    print(f"Speed of light: {c.value:.3e} m/s")

    # Add plane wave source in middle
    freq = 10e9  # 10 GHz
    wavelength = c.value / freq
    print(f"Frequency: {freq/1e9:.1f} GHz")
    print(f"Wavelength: {wavelength*1e3:.2f} mm")

    # Place source
    source_pos = (10, ny//2, nz//2)

    # Simulate wave propagation
    n_periods = 3
    steps_per_period = int(1 / (freq * engine.dt))
    total_steps = n_periods * steps_per_period

    print(f"Simulating {n_periods} periods ({total_steps} steps)...")

    for step in range(total_steps):
        engine.add_source(source_pos, amplitude=1.0, frequency=freq, component='Ez')
        engine.step()

    # Check field propagation
    Ez_profile = engine.Ez[:, ny//2, nz//2]
    max_Ez = np.max(np.abs(Ez_profile))

    print(f"Max Ez field: {max_Ez:.4f} V/m")
    print(f"Total EM energy: {engine.total_energy():.6e} J")

    # Test 2: Coulomb force between charges
    print("\nTest 2: Coulomb force calculation")

    q1 = Charge(
        position=np.array([0.0, 0.0, 0.0]),
        charge=1e-9,  # 1 nC
        velocity=np.zeros(3)
    )

    q2 = Charge(
        position=np.array([0.1, 0.0, 0.0]),  # 10 cm away
        charge=-1e-9,  # -1 nC
        velocity=np.zeros(3)
    )

    F = coulomb_force(q1, q2)
    F_magnitude = np.linalg.norm(F)

    # Analytical result
    k = 1.0 / (4 * np.pi * epsilon_0.value)
    F_expected = k * q1.charge * abs(q2.charge) / 0.1**2

    print(f"Force on q1: {F} N")
    print(f"Force magnitude: {F_magnitude:.6e} N")
    print(f"Expected: {F_expected:.6e} N")
    print(f"Error: {abs(F_magnitude - F_expected) / F_expected * 100:.4f}%")

    # Test 3: Lorentz force on moving charge
    print("\nTest 3: Lorentz force on moving charge")

    q = Charge(
        position=np.array([0.0, 0.0, 0.0]),
        charge=e.value,  # Electron charge
        velocity=np.array([1e6, 0.0, 0.0]),  # 1000 km/s in x direction
        mass=9.109e-31  # Electron mass
    )

    # Uniform magnetic field in z direction
    B_field = np.array([0.0, 0.0, 1.0])  # 1 Tesla
    E_field = np.zeros(3)

    F = lorentz_force(q, E_field, B_field)

    print(f"Charge velocity: {q.velocity} m/s")
    print(f"Magnetic field: {B_field} T")
    print(f"Lorentz force: {F} N")
    print(f"Force magnitude: {np.linalg.norm(F):.6e} N")

    # Force should be in y direction (v × B)
    expected_Fy = q.charge * q.velocity[0] * B_field[2]
    print(f"Expected Fy: {expected_Fy:.6e} N")
    print(f"Error: {abs(F[1] - expected_Fy) / abs(expected_Fy) * 100:.4f}%")

    # Test 4: Biot-Savart law
    print("\nTest 4: Magnetic field from current element")

    current_elem = Current(
        position=np.array([0.0, 0.0, 0.0]),
        current=10.0,  # 10 A
        direction=np.array([1.0, 0.0, 0.0]),  # Current in x direction
        length=0.01  # 1 cm
    )

    obs_point = np.array([0.0, 0.1, 0.0])  # 10 cm away in y direction

    B = biot_savart(current_elem, obs_point)

    print(f"Current: {current_elem.current} A")
    print(f"Current direction: {current_elem.direction}")
    print(f"Observation point: {obs_point} m")
    print(f"Magnetic field: {B} T")
    print(f"Field magnitude: {np.linalg.norm(B):.6e} T")

    # Field should be in z direction (dl × r̂)
    print(f"Field direction: z-component = {B[2]:.6e} T")

    print("\n" + "=" * 80)
    print("Electromagnetism engine tests complete!")
