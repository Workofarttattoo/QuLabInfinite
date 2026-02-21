"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

CONDENSED MATTER PHYSICS LAB
Advanced condensed matter simulations including band structure calculations,
tight-binding models, BCS superconductivity, phonon dispersion, and transport properties.
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from scipy import constants, linalg, special, integrate, optimize
from typing import Tuple, List, Dict, Optional, Callable
from dataclasses import dataclass, field


# Physical constants
KB = constants.k  # Boltzmann constant (J/K)
HBAR = constants.hbar  # Reduced Planck constant (J·s)
ME = constants.m_e  # Electron mass (kg)
E_CHARGE = constants.e  # Elementary charge (C)
MU_0 = constants.mu_0  # Vacuum permeability (H/m)
EPSILON_0 = constants.epsilon_0  # Vacuum permittivity (F/m)
A_BOHR = constants.physical_constants['Bohr radius'][0]  # Bohr radius (m)
EV_TO_J = constants.eV  # Electron volt to Joules
RY_TO_EV = 13.6056980659  # Rydberg to eV


@dataclass
class Crystal:
    """Represents a crystal structure."""
    name: str
    lattice_vectors: np.ndarray  # 3x3 matrix, rows are lattice vectors
    basis_atoms: List[Dict]  # List of {'element': str, 'position': np.array}
    lattice_constant: float = 1.0  # Lattice parameter in Angstroms


class CondensedMatterPhysicsLab:
    """Advanced condensed matter physics simulation laboratory."""

    def __init__(self):
        """Initialize the condensed matter physics lab."""
        self.name = "Condensed Matter Physics Laboratory"
        self.version = "2.0.0"

    def create_crystal_structure(self, structure_type: str,
                                lattice_constant: float = 5.43) -> Crystal:
        """
        Create common crystal structures.

        Args:
            structure_type: Type of structure ('fcc', 'bcc', 'diamond', 'graphene', etc.)
            lattice_constant: Lattice parameter in Angstroms

        Returns:
            Crystal object
        """
        a = lattice_constant * 1e-10  # Convert to meters

        if structure_type == 'fcc':
            # Face-centered cubic
            lattice = np.array([
                [0, 0.5, 0.5],
                [0.5, 0, 0.5],
                [0.5, 0.5, 0]
            ]) * a
            basis = [{'element': 'Cu', 'position': np.array([0, 0, 0])}]

        elif structure_type == 'bcc':
            # Body-centered cubic
            lattice = np.array([
                [-0.5, 0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, -0.5]
            ]) * a
            basis = [{'element': 'Fe', 'position': np.array([0, 0, 0])}]

        elif structure_type == 'diamond':
            # Diamond structure
            lattice = np.array([
                [0, 0.5, 0.5],
                [0.5, 0, 0.5],
                [0.5, 0.5, 0]
            ]) * a
            basis = [
                {'element': 'C', 'position': np.array([0, 0, 0])},
                {'element': 'C', 'position': np.array([0.25, 0.25, 0.25]) * a}
            ]

        elif structure_type == 'graphene':
            # Hexagonal lattice (2D)
            lattice = np.array([
                [1, 0, 0],
                [0.5, np.sqrt(3)/2, 0],
                [0, 0, 10]  # Large z-spacing for 2D
            ]) * a
            basis = [
                {'element': 'C', 'position': np.array([0, 0, 0])},
                {'element': 'C', 'position': np.array([1/3, 1/3, 0]) * a}
            ]

        elif structure_type == 'simple_cubic':
            lattice = np.eye(3) * a
            basis = [{'element': 'Na', 'position': np.array([0, 0, 0])}]

        else:
            # Default to simple cubic
            lattice = np.eye(3) * a
            basis = [{'element': 'X', 'position': np.array([0, 0, 0])}]

        return Crystal(structure_type, lattice, basis, lattice_constant)

    def reciprocal_lattice(self, crystal: Crystal) -> np.ndarray:
        """
        Calculate reciprocal lattice vectors.

        Args:
            crystal: Crystal structure

        Returns:
            3x3 matrix of reciprocal lattice vectors
        """
        a1, a2, a3 = crystal.lattice_vectors

        # Volume of unit cell
        V = np.dot(a1, np.cross(a2, a3))

        # Reciprocal lattice vectors
        b1 = 2 * np.pi * np.cross(a2, a3) / V
        b2 = 2 * np.pi * np.cross(a3, a1) / V
        b3 = 2 * np.pi * np.cross(a1, a2) / V

        return np.array([b1, b2, b3])

    def brillouin_zone_path(self, crystal_type: str,
                          n_points: int = 100) -> Tuple[np.ndarray, List[str]]:
        """
        Generate k-points along high-symmetry path in Brillouin zone.

        Args:
            crystal_type: Type of crystal ('fcc', 'bcc', etc.)
            n_points: Number of points along path

        Returns:
            Tuple of (k-points array, symmetry labels)
        """
        if crystal_type == 'fcc':
            # FCC high-symmetry points (in units of 2π/a)
            points = {
                'Γ': np.array([0, 0, 0]),
                'X': np.array([0, 1, 0]),
                'W': np.array([0.5, 1, 0]),
                'K': np.array([0.75, 0.75, 0]),
                'L': np.array([0.5, 0.5, 0.5]),
                'U': np.array([0.25, 1, 0.25])
            }
            path = ['Γ', 'X', 'W', 'K', 'Γ', 'L', 'U', 'W', 'L', 'K']

        elif crystal_type == 'bcc':
            points = {
                'Γ': np.array([0, 0, 0]),
                'H': np.array([0, 0, 1]),
                'P': np.array([0.5, 0.5, 0.5]),
                'N': np.array([0, 0.5, 0.5])
            }
            path = ['Γ', 'H', 'N', 'Γ', 'P', 'N']

        else:
            # Simple cubic
            points = {
                'Γ': np.array([0, 0, 0]),
                'X': np.array([0.5, 0, 0]),
                'M': np.array([0.5, 0.5, 0]),
                'R': np.array([0.5, 0.5, 0.5])
            }
            path = ['Γ', 'X', 'M', 'Γ', 'R', 'M']

        # Generate k-points along path
        k_points = []
        labels = []
        segment_length = n_points // (len(path) - 1)

        for i in range(len(path) - 1):
            start = points[path[i]]
            end = points[path[i + 1]]

            for j in range(segment_length):
                t = j / segment_length
                k = start + t * (end - start)
                k_points.append(k)

            labels.append(path[i])

        labels.append(path[-1])
        k_points.append(points[path[-1]])

        return np.array(k_points), labels

    def tight_binding_hamiltonian(self, k: np.ndarray, crystal: Crystal,
                                 t: float = 1.0, t_nn: float = 0.0) -> np.ndarray:
        """
        Construct tight-binding Hamiltonian for given k-point.

        Args:
            k: Wave vector
            crystal: Crystal structure
            t: Nearest-neighbor hopping parameter (eV)
            t_nn: Next-nearest-neighbor hopping (eV)

        Returns:
            Hamiltonian matrix
        """
        n_orbitals = len(crystal.basis_atoms)
        H = np.zeros((n_orbitals, n_orbitals), dtype=complex)

        # On-site energies (diagonal)
        for i in range(n_orbitals):
            H[i, i] = 0  # Can add site-dependent energies

        # Hopping terms
        for i in range(n_orbitals):
            for j in range(n_orbitals):
                if i != j:
                    # Position difference
                    r_ij = (crystal.basis_atoms[j]['position'] -
                           crystal.basis_atoms[i]['position'])

                    # Nearest-neighbor hopping
                    phase = np.exp(1j * np.dot(k, r_ij))
                    H[i, j] = -t * phase

        # Add lattice periodicity
        for n1 in [-1, 0, 1]:
            for n2 in [-1, 0, 1]:
                for n3 in [-1, 0, 1]:
                    if n1 == 0 and n2 == 0 and n3 == 0:
                        continue

                    R = (n1 * crystal.lattice_vectors[0] +
                         n2 * crystal.lattice_vectors[1] +
                         n3 * crystal.lattice_vectors[2])

                    phase = np.exp(1j * np.dot(k, R))

                    # Check if R connects nearest neighbors
                    R_norm = np.linalg.norm(R)
                    min_lattice = min(np.linalg.norm(crystal.lattice_vectors[i])
                                    for i in range(3))

                    if abs(R_norm - min_lattice) < 0.1 * min_lattice:
                        # Nearest neighbor
                        for i in range(n_orbitals):
                            H[i, i] += -t * phase

        return H

    def calculate_band_structure(self, crystal: Crystal, k_points: np.ndarray,
                                hopping_params: Dict = None) -> np.ndarray:
        """
        Calculate electronic band structure.

        Args:
            crystal: Crystal structure
            k_points: Array of k-points
            hopping_params: Dictionary of hopping parameters

        Returns:
            Band energies (n_kpoints x n_bands)
        """
        if hopping_params is None:
            hopping_params = {'t': 1.0, 't_nn': 0.0}

        n_kpoints = len(k_points)
        n_bands = len(crystal.basis_atoms)

        energies = np.zeros((n_kpoints, n_bands))

        for i, k in enumerate(k_points):
            H = self.tight_binding_hamiltonian(k, crystal, **hopping_params)
            eigenvalues = linalg.eigvalsh(H)
            energies[i] = sorted(eigenvalues)

        return energies

    def density_of_states(self, energies: np.ndarray,
                         energy_range: Tuple[float, float] = None,
                         n_bins: int = 200,
                         broadening: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate density of states from band structure.

        Args:
            energies: Band energies
            energy_range: (E_min, E_max) in eV
            n_bins: Number of energy bins
            broadening: Gaussian broadening in eV

        Returns:
            Tuple of (energy array, DOS array)
        """
        if energy_range is None:
            E_min = np.min(energies) - 1
            E_max = np.max(energies) + 1
        else:
            E_min, E_max = energy_range

        E = np.linspace(E_min, E_max, n_bins)
        dos = np.zeros(n_bins)

        # Flatten all energies
        all_energies = energies.flatten()

        # Calculate DOS with Gaussian broadening
        for energy in all_energies:
            dos += np.exp(-(E - energy)**2 / (2 * broadening**2))

        # Normalize
        dos /= (np.sqrt(2 * np.pi) * broadening * len(all_energies))

        return E, dos

    def fermi_dirac(self, E: np.ndarray, E_F: float, T: float) -> np.ndarray:
        """
        Fermi-Dirac distribution function.

        Args:
            E: Energy array (eV)
            E_F: Fermi energy (eV)
            T: Temperature (K)

        Returns:
            Occupation probability
        """
        if T == 0:
            return np.where(E < E_F, 1.0, 0.0)

        beta = 1 / (KB * T / EV_TO_J)
        return 1 / (1 + np.exp(beta * (E - E_F)))

    def bcs_gap_equation(self, T: float, V: float, omega_D: float,
                        DOS_EF: float) -> float:
        """
        Solve BCS gap equation for superconductivity.

        Args:
            T: Temperature (K)
            V: Pairing potential (eV)
            omega_D: Debye frequency (eV)
            DOS_EF: Density of states at Fermi level (1/eV)

        Returns:
            Superconducting gap (eV)
        """
        if T == 0:
            # Zero temperature gap
            if V * DOS_EF < 0.2:  # Weak coupling
                Delta_0 = 2 * omega_D * np.exp(-1 / (V * DOS_EF))
            else:
                Delta_0 = omega_D * V * DOS_EF
            return Delta_0

        # Self-consistent solution at finite T
        def gap_equation(Delta):
            if Delta <= 0:
                return -1

            # Integration over energy
            def integrand(xi):
                E = np.sqrt(xi**2 + Delta**2)
                return np.tanh(E / (2 * KB * T / EV_TO_J)) / E

            integral, _ = integrate.quad(integrand, 0, omega_D)
            return 1 - V * DOS_EF * integral

        # Find solution
        if V * DOS_EF > 0.1:
            try:
                sol = optimize.brentq(gap_equation, 1e-6, omega_D)
                return sol
            except:
                return 0.0
        else:
            return 0.0

    def bcs_coherence_length(self, Delta: float, v_F: float) -> float:
        """
        Calculate BCS coherence length.

        Args:
            Delta: Superconducting gap (eV)
            v_F: Fermi velocity (m/s)

        Returns:
            Coherence length in meters
        """
        if Delta == 0:
            return np.inf

        xi = HBAR * v_F / (np.pi * Delta * EV_TO_J)
        return xi

    def phonon_dispersion(self, k: np.ndarray, M: float,
                         K: float, K_nn: float = 0) -> np.ndarray:
        """
        Calculate phonon dispersion relation (1D chain model).

        Args:
            k: Wave vector array
            M: Atomic mass (kg)
            K: Spring constant (N/m)
            K_nn: Next-nearest neighbor spring constant

        Returns:
            Phonon frequencies (rad/s)
        """
        omega = np.sqrt(2 * K / M * (1 - np.cos(k)) +
                       2 * K_nn / M * (1 - np.cos(2 * k)))
        return omega

    def debye_model(self, T: float, theta_D: float) -> Dict:
        """
        Calculate thermodynamic properties using Debye model.

        Args:
            T: Temperature (K)
            theta_D: Debye temperature (K)

        Returns:
            Dictionary of thermodynamic properties
        """
        x = theta_D / T

        # Debye function D_3(x)
        if x > 20:
            # High temperature limit
            D3 = 1 - 3 * x / 8 + x**2 / 20
        else:
            # Numerical integration
            def integrand(t):
                return t**3 / (np.exp(t) - 1)

            integral, _ = integrate.quad(integrand, 0, x)
            D3 = 3 / x**3 * integral

        # Heat capacity
        C_V = 9 * KB * D3

        # Internal energy
        U = 9 * KB * T * D3

        # Low/high temperature limits for specific heat
        if T < theta_D / 50:
            # Low-T limit (T³ law)
            C_V = 12 * np.pi**4 * KB / 5 * (T / theta_D)**3
        elif T > 2 * theta_D:
            # High-T limit (Dulong-Petit)
            C_V = 3 * KB

        return {
            'heat_capacity': C_V,
            'internal_energy': U,
            'debye_function': D3
        }

    def drude_conductivity(self, omega: np.ndarray, n: float,
                          tau: float, m: float = ME) -> np.ndarray:
        """
        Calculate Drude optical conductivity.

        Args:
            omega: Frequency array (rad/s)
            n: Electron density (m^-3)
            tau: Scattering time (s)
            m: Electron mass (kg)

        Returns:
            Complex conductivity
        """
        omega_p = np.sqrt(n * E_CHARGE**2 / (EPSILON_0 * m))  # Plasma frequency

        sigma = (n * E_CHARGE**2 * tau / m) / (1 - 1j * omega * tau)

        return sigma

    def hall_coefficient(self, carrier_type: str, n: float) -> float:
        """
        Calculate Hall coefficient.

        Args:
            carrier_type: 'electron' or 'hole'
            n: Carrier density (m^-3)

        Returns:
            Hall coefficient (m³/C)
        """
        if carrier_type == 'electron':
            R_H = -1 / (n * E_CHARGE)
        else:  # hole
            R_H = 1 / (n * E_CHARGE)

        return R_H

    def thermal_conductivity_electrons(self, sigma: float, T: float) -> float:
        """
        Calculate electronic thermal conductivity (Wiedemann-Franz law).

        Args:
            sigma: Electrical conductivity (S/m)
            T: Temperature (K)

        Returns:
            Thermal conductivity (W/m/K)
        """
        L = (np.pi**2 / 3) * (KB / E_CHARGE)**2  # Lorenz number
        kappa = L * sigma * T
        return kappa

    def bloch_theorem(self, k: np.ndarray, r: np.ndarray,
                     u_k: Callable) -> complex:
        """
        Bloch wavefunction.

        Args:
            k: Wave vector
            r: Position vector
            u_k: Periodic function u_k(r)

        Returns:
            Bloch wavefunction value
        """
        return u_k(r) * np.exp(1j * np.dot(k, r))

    def wannier_function(self, n: int, R: np.ndarray,
                        band_structure: np.ndarray,
                        k_points: np.ndarray) -> np.ndarray:
        """
        Calculate Wannier function (simplified).

        Args:
            n: Band index
            R: Lattice vector
            band_structure: Band energies
            k_points: k-point mesh

        Returns:
            Wannier function values
        """
        N_k = len(k_points)
        w_n = np.zeros(N_k, dtype=complex)

        for i, k in enumerate(k_points):
            # Simplified - normally would use actual Bloch functions
            w_n[i] = np.exp(-1j * np.dot(k, R)) / np.sqrt(N_k)

        return w_n

    def anderson_localization(self, L: int, W: float, t: float = 1.0) -> np.ndarray:
        """
        Anderson localization in 1D disordered system.

        Args:
            L: System size
            W: Disorder strength
            t: Hopping parameter

        Returns:
            Eigenvalues and participation ratios
        """
        # Random on-site energies
        epsilon = np.random.uniform(-W/2, W/2, L)

        # Construct Hamiltonian
        H = np.diag(epsilon)
        for i in range(L - 1):
            H[i, i+1] = -t
            H[i+1, i] = -t

        # Add periodic boundary conditions
        H[0, L-1] = -t
        H[L-1, 0] = -t

        # Solve for eigenvalues and eigenvectors
        eigenvalues, eigenvectors = linalg.eigh(H)

        # Calculate participation ratio
        participation = np.zeros(L)
        for i in range(L):
            psi = eigenvectors[:, i]
            participation[i] = 1 / np.sum(np.abs(psi)**4) / L

        return eigenvalues, participation

    def topological_invariant_1d(self, H_k: Callable, k_points: np.ndarray) -> float:
        """
        Calculate 1D topological invariant (winding number).

        Args:
            H_k: Function that returns Hamiltonian for given k
            k_points: k-points around Brillouin zone

        Returns:
            Winding number
        """
        winding = 0
        n_k = len(k_points)

        for i in range(n_k):
            k1 = k_points[i]
            k2 = k_points[(i + 1) % n_k]

            # Get ground state at each k
            _, v1 = linalg.eigh(H_k(k1))
            _, v2 = linalg.eigh(H_k(k2))

            # Ground state overlap
            overlap = np.vdot(v1[:, 0], v2[:, 0])
            winding += np.angle(overlap)

        winding /= (2 * np.pi)
        return np.round(winding)

    def quantum_hall_conductance(self, B: float, n: int) -> float:
        """
        Calculate quantum Hall conductance.

        Args:
            B: Magnetic field (T)
            n: Landau level filling factor

        Returns:
            Hall conductance (e²/h units)
        """
        # Quantized Hall conductance
        sigma_xy = n * E_CHARGE**2 / (2 * np.pi * HBAR)
        return sigma_xy

    def magnetic_susceptibility_pauli(self, DOS_EF: float) -> float:
        """
        Calculate Pauli paramagnetic susceptibility.

        Args:
            DOS_EF: Density of states at Fermi level (1/eV/m³)

        Returns:
            Magnetic susceptibility (SI units)
        """
        mu_B = constants.physical_constants['Bohr magneton'][0]
        chi = mu_B**2 * MU_0 * DOS_EF * EV_TO_J
        return chi


def run_demo():
    """Demonstrate condensed matter physics calculations."""
    lab = CondensedMatterPhysicsLab()
    print(f"Initializing {lab.name} v{lab.version}")
    print("=" * 60)

    # 1. Crystal structure
    print("\n1. Crystal Structure:")
    crystal = lab.create_crystal_structure('fcc', lattice_constant=3.61)  # Cu
    print(f"   Crystal type: FCC (Copper)")
    print(f"   Lattice constant: 3.61 Å")

    reciprocal = lab.reciprocal_lattice(crystal)
    print(f"   Reciprocal lattice vector magnitude: {np.linalg.norm(reciprocal[0]):.3f} 1/m")

    # 2. Band structure
    print("\n2. Electronic Band Structure:")
    k_points, labels = lab.brillouin_zone_path('fcc', n_points=100)
    print(f"   High-symmetry path: {' → '.join(labels[::10])}")

    # Calculate bands (simplified tight-binding)
    bands = lab.calculate_band_structure(crystal, k_points, {'t': 1.0, 't_nn': 0.1})
    print(f"   Number of bands: {bands.shape[1]}")
    print(f"   Band width: {np.max(bands) - np.min(bands):.2f} eV")

    # 3. Density of states
    print("\n3. Density of States:")
    E, dos = lab.density_of_states(bands, broadening=0.05)
    E_F_index = np.argmax(dos)
    E_F = E[E_F_index]
    DOS_EF = dos[E_F_index]
    print(f"   Fermi energy (estimated): {E_F:.2f} eV")
    print(f"   DOS at Fermi level: {DOS_EF:.2f} states/eV")

    # 4. BCS Superconductivity
    print("\n4. BCS Superconductivity:")
    T = 1.0  # K
    V = 0.3  # eV
    omega_D = 0.03  # eV (typical for metals)
    DOS_EF_sc = 2.0  # states/eV

    Delta = lab.bcs_gap_equation(T, V, omega_D, DOS_EF_sc)
    print(f"   Temperature: {T} K")
    print(f"   Pairing potential: {V} eV")
    print(f"   Debye frequency: {omega_D * 1000:.1f} meV")
    print(f"   Superconducting gap: {Delta * 1000:.3f} meV")

    if Delta > 0:
        T_c = Delta / (1.76 * KB / EV_TO_J)  # BCS relation
        print(f"   Critical temperature (BCS): {T_c:.2f} K")

        v_F = 1e6  # Fermi velocity (m/s)
        xi = lab.bcs_coherence_length(Delta, v_F)
        print(f"   Coherence length: {xi * 1e9:.1f} nm")

    # 5. Phonons and Debye model
    print("\n5. Phonon Properties:")
    M = 63.5 * constants.atomic_mass  # Cu atomic mass
    K = 50  # N/m (spring constant)

    k_phonon = np.linspace(-np.pi, np.pi, 100)
    omega_phonon = lab.phonon_dispersion(k_phonon, M, K, K_nn=K*0.1)
    max_omega = np.max(omega_phonon)
    print(f"   Maximum phonon frequency: {max_omega / (2 * np.pi) / 1e12:.1f} THz")

    # Debye model
    theta_D = 343  # K (Debye temperature for Cu)
    T_debye = 300  # K

    debye_props = lab.debye_model(T_debye, theta_D)
    print(f"\n   Debye model at {T_debye} K:")
    print(f"   Heat capacity: {debye_props['heat_capacity'] / KB:.3f} kB per atom")
    print(f"   Debye temperature: {theta_D} K")

    # 6. Transport properties
    print("\n6. Transport Properties:")

    # Drude model
    n_electrons = 8.45e28  # m^-3 (Cu)
    tau = 2.5e-14  # s (scattering time)
    omega_range = np.logspace(12, 16, 100)  # rad/s

    sigma_dc = n_electrons * E_CHARGE**2 * tau / ME
    print(f"   DC conductivity: {sigma_dc:.2e} S/m")

    # Hall coefficient
    R_H = lab.hall_coefficient('electron', n_electrons)
    print(f"   Hall coefficient: {R_H * 1e10:.3f} × 10^-10 m³/C")

    # Thermal conductivity
    kappa = lab.thermal_conductivity_electrons(sigma_dc, T_debye)
    print(f"   Electronic thermal conductivity: {kappa:.1f} W/m/K")

    # 7. Anderson localization
    print("\n7. Anderson Localization (1D):")
    L = 100  # System size
    W = 3.0  # Disorder strength

    energies, participation = lab.anderson_localization(L, W)

    # Localization length
    avg_participation = np.mean(participation)
    print(f"   System size: {L} sites")
    print(f"   Disorder strength: W/t = {W}")
    print(f"   Average participation ratio: {avg_participation:.3f}")

    if avg_participation < 0.3:
        print(f"   System is LOCALIZED")
    else:
        print(f"   System is EXTENDED")

    # 8. Topological properties
    print("\n8. Topological Properties:")

    # SSH model (1D topological insulator)
    def ssh_hamiltonian(k, v=0.5, w=1.0):
        """Su-Schrieffer-Heeger model."""
        H = np.array([
            [0, v + w * np.exp(-1j * k)],
            [v + w * np.exp(1j * k), 0]
        ])
        return H

    k_topo = np.linspace(-np.pi, np.pi, 100)
    winding = lab.topological_invariant_1d(ssh_hamiltonian, k_topo)
    print(f"   SSH model winding number: {int(winding)}")

    if winding != 0:
        print(f"   System is TOPOLOGICALLY NON-TRIVIAL")
    else:
        print(f"   System is TOPOLOGICALLY TRIVIAL")

    # 9. Quantum Hall effect
    print("\n9. Quantum Hall Effect:")
    B_hall = 10  # Tesla
    n_landau = 2  # Filling factor

    sigma_xy = lab.quantum_hall_conductance(B_hall, n_landau)
    conductance_quantum = E_CHARGE**2 / (2 * np.pi * HBAR)
    print(f"   Magnetic field: {B_hall} T")
    print(f"   Filling factor: {n_landau}")
    print(f"   Hall conductance: {sigma_xy / conductance_quantum:.1f} × (e²/h)")

    # 10. Magnetic properties
    print("\n10. Magnetic Properties:")
    chi_pauli = lab.magnetic_susceptibility_pauli(DOS_EF_sc / EV_TO_J)
    print(f"   Pauli susceptibility: {chi_pauli:.2e} (SI units)")

    # Van Vleck susceptibility (order of magnitude estimate)
    chi_vv = chi_pauli * 0.1
    print(f"   Van Vleck contribution: ~{chi_vv:.2e} (SI units)")

    print("\n" + "=" * 60)
    print("Condensed Matter Physics Lab demonstration complete!")


if __name__ == "__main__":
    run_demo()