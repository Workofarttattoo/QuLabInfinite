"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

PARTICLE PHYSICS LAB
Advanced particle physics simulations including cross-sections, decay rates,
Feynman diagrams, Mandelstam variables, and PDG particle data.
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from scipy import constants, special, integrate
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import math


# Physical constants
C = constants.c  # Speed of light (m/s)
HBAR = constants.hbar  # Reduced Planck constant (J·s)
ALPHA = constants.fine_structure  # Fine structure constant
ME = constants.m_e  # Electron mass (kg)
MP = constants.m_p  # Proton mass (kg)
MN = constants.m_n  # Neutron mass (kg)
E_CHARGE = constants.e  # Elementary charge (C)
GF = 1.1663787e-5  # Fermi coupling constant (GeV^-2)
MW = 80.379  # W boson mass (GeV/c²)
MZ = 91.1876  # Z boson mass (GeV/c²)
MH = 125.10  # Higgs boson mass (GeV/c²)

# Conversion factors
GEV_TO_J = 1.60217663e-10  # GeV to Joules
FM_TO_M = 1e-15  # Fermi to meters
MBARN_TO_M2 = 1e-31  # Millibarn to m²

# CKM matrix elements (approximate)
V_UD = 0.97370
V_US = 0.2245
V_UB = 0.00382


@dataclass
class Particle:
    """Represents a fundamental particle with PDG properties."""
    name: str
    pdg_id: int
    mass: float  # GeV/c²
    charge: float  # Elementary charge units
    spin: float  # ℏ units
    isospin: float
    color_charge: int = 0  # 0 for colorless, 3 for quarks
    lifetime: Optional[float] = None  # seconds
    width: Optional[float] = None  # GeV


@dataclass
class FourVector:
    """Relativistic four-vector for particle kinematics."""
    E: float  # Energy (GeV)
    px: float  # Momentum x-component (GeV/c)
    py: float  # Momentum y-component (GeV/c)
    pz: float  # Momentum z-component (GeV/c)

    def __add__(self, other):
        return FourVector(
            self.E + other.E,
            self.px + other.px,
            self.py + other.py,
            self.pz + other.pz
        )

    def __sub__(self, other):
        return FourVector(
            self.E - other.E,
            self.px - other.px,
            self.py - other.py,
            self.pz - other.pz
        )

    def dot(self, other) -> float:
        """Minkowski dot product."""
        return self.E * other.E - self.px * other.px - self.py * other.py - self.pz * other.pz

    def invariant_mass(self) -> float:
        """Calculate invariant mass."""
        m2 = self.dot(self)
        return np.sqrt(max(0, m2))

    def boost(self, beta_x: float, beta_y: float, beta_z: float) -> 'FourVector':
        """Lorentz boost transformation."""
        beta = np.sqrt(beta_x**2 + beta_y**2 + beta_z**2)
        if beta >= 1:
            raise ValueError("Beta must be < 1")
        if beta < 1e-10:
            return FourVector(self.E, self.px, self.py, self.pz)

        gamma = 1 / np.sqrt(1 - beta**2)
        n = np.array([beta_x, beta_y, beta_z]) / beta
        p = np.array([self.px, self.py, self.pz])

        p_parallel = np.dot(p, n) * n
        p_perp = p - p_parallel

        E_new = gamma * (self.E - beta * np.dot(p, n))
        p_parallel_new = gamma * (p_parallel - beta * self.E * n)
        p_new = p_parallel_new + p_perp

        return FourVector(E_new, p_new[0], p_new[1], p_new[2])


class ParticlePhysicsLab:
    """Advanced particle physics simulation laboratory."""

    def __init__(self):
        """Initialize the particle physics lab with PDG data."""
        self.name = "Particle Physics Laboratory"
        self.version = "2.0.0"
        self.particles = self._initialize_pdg_particles()

    def _initialize_pdg_particles(self) -> Dict[str, Particle]:
        """Initialize dictionary of common particles with PDG values."""
        particles = {
            # Leptons
            'electron': Particle('electron', 11, 0.000511, -1, 0.5, 0.5, lifetime=np.inf),
            'muon': Particle('muon', 13, 0.105658, -1, 0.5, 0.5, lifetime=2.197e-6),
            'tau': Particle('tau', 15, 1.77686, -1, 0.5, 0.5, lifetime=2.903e-13),
            'electron_neutrino': Particle('νe', 12, 0, 0, 0.5, 0.5),
            'muon_neutrino': Particle('νμ', 14, 0, 0, 0.5, 0.5),
            'tau_neutrino': Particle('ντ', 16, 0, 0, 0.5, 0.5),

            # Quarks
            'up': Particle('up', 2, 0.00216, 2/3, 0.5, 0.5, color_charge=3),
            'down': Particle('down', 1, 0.00467, -1/3, 0.5, 0.5, color_charge=3),
            'charm': Particle('charm', 4, 1.27, 2/3, 0.5, 0, color_charge=3),
            'strange': Particle('strange', 3, 0.093, -1/3, 0.5, 0, color_charge=3),
            'top': Particle('top', 6, 172.76, 2/3, 0.5, 0, color_charge=3, width=1.42),
            'bottom': Particle('bottom', 5, 4.18, -1/3, 0.5, 0, color_charge=3),

            # Gauge bosons
            'photon': Particle('photon', 22, 0, 0, 1, 0),
            'W+': Particle('W+', 24, MW, 1, 1, 1, width=2.085),
            'W-': Particle('W-', -24, MW, -1, 1, 1, width=2.085),
            'Z': Particle('Z', 23, MZ, 0, 1, 0, width=2.4952),
            'gluon': Particle('gluon', 21, 0, 0, 1, 0, color_charge=8),
            'higgs': Particle('Higgs', 25, MH, 0, 0, 0, width=0.00407),

            # Hadrons
            'proton': Particle('proton', 2212, 0.938272, 1, 0.5, 0.5, lifetime=np.inf),
            'neutron': Particle('neutron', 2112, 0.939565, 0, 0.5, 0.5, lifetime=879.4),
            'pion+': Particle('π+', 211, 0.13957, 1, 0, 1, lifetime=2.603e-8),
            'pion-': Particle('π-', -211, 0.13957, -1, 0, 1, lifetime=2.603e-8),
            'pion0': Particle('π0', 111, 0.13498, 0, 0, 1, lifetime=8.52e-17),
            'kaon+': Particle('K+', 321, 0.49368, 1, 0, 0.5, lifetime=1.238e-8),
            'kaon-': Particle('K-', -321, 0.49368, -1, 0, 0.5, lifetime=1.238e-8),
            'kaon0': Particle('K0', 311, 0.49761, 0, 0, 0.5),
        }
        return particles

    def calculate_cross_section(self, process: str, s: float,
                               cos_theta: Optional[float] = None) -> float:
        """
        Calculate cross-section for various processes.

        Args:
            process: Process name (e.g., 'ee_to_mumu', 'pp_elastic')
            s: Center-of-mass energy squared (GeV²)
            cos_theta: Cosine of scattering angle (for differential cross-section)

        Returns:
            Cross-section in millibarns (mb)
        """
        sqrt_s = np.sqrt(s)

        if process == 'ee_to_mumu':
            # e⁺e⁻ → μ⁺μ⁻ (QED)
            if sqrt_s < 2 * self.particles['muon'].mass:
                return 0.0

            # Total cross-section
            sigma = (4 * np.pi * ALPHA**2) / (3 * s) * HBAR**2 * C**2 / MBARN_TO_M2 * 1e3

            # Differential cross-section
            if cos_theta is not None:
                dsigma_dcos = sigma * (3/8) * (1 + cos_theta**2)
                return dsigma_dcos

            return sigma

        elif process == 'ee_to_hadrons':
            # e⁺e⁻ → hadrons (via quark-antiquark)
            # R-ratio approximation
            Nc = 3  # Number of colors

            # Sum over accessible quark flavors
            R = 0
            for q_name in ['up', 'down', 'charm', 'strange', 'bottom']:
                q = self.particles[q_name]
                if sqrt_s > 2 * q.mass:
                    R += Nc * q.charge**2

            sigma_point = (4 * np.pi * ALPHA**2) / (3 * s) * HBAR**2 * C**2 / MBARN_TO_M2 * 1e3
            return R * sigma_point

        elif process == 'pp_elastic':
            # Proton-proton elastic scattering (Regge theory approximation)
            t = -2 * s * (1 - cos_theta) if cos_theta else -0.1  # GeV²

            # Pomeron trajectory
            alpha_P = lambda t: 1.08 + 0.25 * t

            # Elastic cross-section (simplified)
            A = 100  # mb·GeV²
            sigma = A * s**(2 * alpha_P(t) - 2)

            return sigma

        elif process == 'deep_inelastic':
            # Deep inelastic scattering (ep → eX)
            Q2 = s / 4  # Approximate Q²
            x = 0.1  # Bjorken x (example value)

            # Structure function F2 (simplified)
            F2 = x * (1 - x) * (3 + 2 * np.log(Q2))

            # Cross-section
            sigma = (4 * np.pi * ALPHA**2 / Q2**2) * F2 * HBAR**2 * C**2 / MBARN_TO_M2 * 1e3

            return sigma

        else:
            raise ValueError(f"Unknown process: {process}")

    def calculate_decay_rate(self, parent: str, products: List[str]) -> float:
        """
        Calculate decay rate for particle decays.

        Args:
            parent: Parent particle name
            products: List of decay product names

        Returns:
            Decay rate in GeV
        """
        p = self.particles[parent]

        if parent == 'muon' and set(products) == {'electron', 'electron_neutrino', 'muon_neutrino'}:
            # Muon decay: μ⁻ → e⁻ν̄ₑνμ
            gamma = (GF**2 * p.mass**5) / (192 * np.pi**3) * (HBAR * C)**3

        elif parent == 'neutron' and set(products) == {'proton', 'electron', 'electron_neutrino'}:
            # Neutron beta decay
            Q = (self.particles['neutron'].mass - self.particles['proton'].mass -
                 self.particles['electron'].mass)  # GeV

            # Phase space factor (simplified)
            f = Q**5 / 30

            gamma = (GF**2 * V_UD**2 * f) / (2 * np.pi**3) * (HBAR * C)**3

        elif parent == 'pion+' and set(products) == {'muon', 'muon_neutrino'}:
            # Charged pion decay: π⁺ → μ⁺νμ
            m_pi = self.particles['pion+'].mass
            m_mu = self.particles['muon'].mass

            gamma = (GF**2 * V_UD**2 * m_pi * m_mu**2) / (8 * np.pi) * \
                   (1 - m_mu**2 / m_pi**2)**2 * (HBAR * C)**3

        elif parent == 'higgs':
            # Simplified Higgs decay width
            if 'bottom' in products:
                # H → bb̄
                m_h = self.particles['higgs'].mass
                m_b = self.particles['bottom'].mass
                Nc = 3

                gamma = (3 * m_b**2 * m_h) / (8 * np.pi * (246)**2) * \
                       np.sqrt(1 - 4 * m_b**2 / m_h**2) * Nc

            elif 'W+' in products and 'W-' in products:
                # H → WW
                m_h = self.particles['higgs'].mass
                x = 4 * MW**2 / m_h**2

                if x < 1:
                    gamma = (m_h**3 / (64 * np.pi * MW**2)) * \
                           np.sqrt(1 - x) * (1 - x + 0.75 * x**2)
                else:
                    gamma = 0

            else:
                gamma = 0.001  # Default small width

        else:
            # Generic two-body decay (simplified)
            if len(products) == 2:
                m_parent = p.mass
                m1 = self.particles[products[0]].mass
                m2 = self.particles[products[1]].mass

                if m_parent > m1 + m2:
                    # Momentum in CM frame
                    p_cm = np.sqrt((m_parent**2 - (m1 + m2)**2) *
                                  (m_parent**2 - (m1 - m2)**2)) / (2 * m_parent)

                    # Phase space factor
                    gamma = p_cm / (8 * np.pi * m_parent**2) * ALPHA

                else:
                    gamma = 0
            else:
                gamma = 0.001  # Default

        return gamma

    def calculate_mandelstam_variables(self, p1: FourVector, p2: FourVector,
                                      p3: FourVector, p4: FourVector) -> Tuple[float, float, float]:
        """
        Calculate Mandelstam variables for 2→2 scattering.

        Args:
            p1, p2: Initial state four-momenta
            p3, p4: Final state four-momenta

        Returns:
            Tuple of (s, t, u) Mandelstam variables in GeV²
        """
        s = (p1 + p2).dot(p1 + p2)
        t = (p1 - p3).dot(p1 - p3)
        u = (p1 - p4).dot(p1 - p4)

        return s, t, u

    def feynman_amplitude(self, process: str, s: float, t: float) -> complex:
        """
        Calculate Feynman amplitude for simple processes.

        Args:
            process: Process name
            s: Mandelstam s (GeV²)
            t: Mandelstam t (GeV²)

        Returns:
            Complex amplitude
        """
        if process == 'ee_to_mumu':
            # QED amplitude for e⁺e⁻ → μ⁺μ⁻
            # s-channel photon exchange
            propagator = 1 / s
            vertex = -E_CHARGE**2  # Simplified vertex factor

            M = vertex * propagator

        elif process == 'electron_muon_scattering':
            # e⁻μ⁻ → e⁻μ⁻ via photon exchange
            # t-channel
            propagator = 1 / t
            vertex = -E_CHARGE**2

            M = vertex * propagator

        elif process == 'compton':
            # Compton scattering γe⁻ → γe⁻
            # s and u channel contributions
            u = -s - t  # Using s + t + u = 0 for massless photons

            M_s = -E_CHARGE**2 / s
            M_u = -E_CHARGE**2 / u

            M = M_s + M_u

        else:
            M = complex(1, 0)

        return M

    def calculate_branching_ratio(self, parent: str,
                                 decay_channel: List[str]) -> float:
        """
        Calculate branching ratio for a decay channel.

        Args:
            parent: Parent particle name
            decay_channel: Specific decay products

        Returns:
            Branching ratio (0 to 1)
        """
        # Get partial width for this channel
        gamma_partial = self.calculate_decay_rate(parent, decay_channel)

        # Get total width
        p = self.particles[parent]
        if p.width:
            gamma_total = p.width
        else:
            # Estimate total width (simplified)
            gamma_total = gamma_partial / 0.5  # Assume 50% branching

        return min(1.0, gamma_partial / gamma_total)

    def parton_distribution_function(self, x: float, Q2: float,
                                    parton: str = 'u') -> float:
        """
        Simple parametrization of parton distribution functions.

        Args:
            x: Bjorken x (momentum fraction)
            Q2: Momentum transfer squared (GeV²)
            parton: Parton type ('u', 'd', 's', 'g')

        Returns:
            PDF value
        """
        if x <= 0 or x >= 1:
            return 0.0

        # CTEQ-like parametrization (simplified)
        if parton == 'u':  # Up quark
            pdf = 1.7 * x**0.5 * (1 - x)**3 * (1 + 4 * x)
        elif parton == 'd':  # Down quark
            pdf = 0.8 * x**0.5 * (1 - x)**4 * (1 + 6 * x)
        elif parton == 's':  # Strange quark
            pdf = 0.3 * x**0.5 * (1 - x)**6
        elif parton == 'g':  # Gluon
            pdf = 3.0 * x**(-0.1) * (1 - x)**5
        else:
            pdf = 0.0

        # Q² evolution (simplified DGLAP)
        Q0 = 1.0  # GeV
        alpha_s = 0.118  # Strong coupling at Z mass

        if Q2 > Q0**2:
            pdf *= (np.log(Q2 / Q0**2) / np.log(MZ**2 / Q0**2))**(alpha_s / np.pi)

        return pdf

    def invariant_mass_spectrum(self, particles: List[FourVector]) -> float:
        """
        Calculate invariant mass of a system of particles.

        Args:
            particles: List of four-vectors

        Returns:
            Invariant mass in GeV
        """
        total = particles[0]
        for p in particles[1:]:
            total = total + p

        return total.invariant_mass()

    def rapidity(self, p: FourVector) -> float:
        """
        Calculate rapidity of a particle.

        Args:
            p: Four-vector

        Returns:
            Rapidity
        """
        return 0.5 * np.log((p.E + p.pz) / (p.E - p.pz))

    def pseudorapidity(self, px: float, py: float, pz: float) -> float:
        """
        Calculate pseudorapidity from momentum components.

        Args:
            px, py, pz: Momentum components

        Returns:
            Pseudorapidity η
        """
        p = np.sqrt(px**2 + py**2 + pz**2)
        if p == 0:
            return 0

        cos_theta = pz / p

        if abs(cos_theta) >= 1:
            return np.sign(cos_theta) * np.inf

        theta = np.arccos(cos_theta)
        return -np.log(np.tan(theta / 2))

    def transverse_momentum(self, px: float, py: float) -> float:
        """Calculate transverse momentum."""
        return np.sqrt(px**2 + py**2)

    def missing_transverse_energy(self, visible_particles: List[FourVector]) -> Tuple[float, float]:
        """
        Calculate missing transverse energy (for neutrinos, etc.).

        Args:
            visible_particles: List of visible particle four-vectors

        Returns:
            Tuple of (missing ET, missing phi)
        """
        sum_px = sum(p.px for p in visible_particles)
        sum_py = sum(p.py for p in visible_particles)

        met = np.sqrt(sum_px**2 + sum_py**2)
        phi = np.arctan2(-sum_py, -sum_px)

        return met, phi

    def jet_clustering(self, particles: List[FourVector],
                      R: float = 0.4, algorithm: str = 'antikt') -> List[List[FourVector]]:
        """
        Simple jet clustering algorithm.

        Args:
            particles: List of particle four-vectors
            R: Jet radius parameter
            algorithm: 'kt', 'antikt', or 'cambridge'

        Returns:
            List of jets (each jet is a list of particles)
        """
        jets = []
        remaining = particles.copy()

        while remaining:
            if len(remaining) == 1:
                jets.append([remaining[0]])
                break

            # Find minimum distance
            min_dist = np.inf
            min_pair = (0, 0)
            is_beam = False

            for i in range(len(remaining)):
                # Distance to beam
                pt_i = self.transverse_momentum(remaining[i].px, remaining[i].py)

                if algorithm == 'kt':
                    d_iB = pt_i**2
                elif algorithm == 'antikt':
                    d_iB = 1 / pt_i**2 if pt_i > 0 else np.inf
                else:  # Cambridge/Aachen
                    d_iB = 1

                if d_iB < min_dist:
                    min_dist = d_iB
                    min_pair = (i, -1)
                    is_beam = True

                # Distance between particles
                for j in range(i + 1, len(remaining)):
                    pt_j = self.transverse_momentum(remaining[j].px, remaining[j].py)

                    # Calculate ΔR
                    eta_i = self.pseudorapidity(remaining[i].px, remaining[i].py, remaining[i].pz)
                    eta_j = self.pseudorapidity(remaining[j].px, remaining[j].py, remaining[j].pz)
                    phi_i = np.arctan2(remaining[i].py, remaining[i].px)
                    phi_j = np.arctan2(remaining[j].py, remaining[j].px)

                    delta_eta = eta_i - eta_j
                    delta_phi = phi_i - phi_j
                    if delta_phi > np.pi:
                        delta_phi -= 2 * np.pi
                    if delta_phi < -np.pi:
                        delta_phi += 2 * np.pi

                    delta_R2 = delta_eta**2 + delta_phi**2

                    if algorithm == 'kt':
                        d_ij = min(pt_i**2, pt_j**2) * delta_R2 / R**2
                    elif algorithm == 'antikt':
                        d_ij = min(1/pt_i**2, 1/pt_j**2) * delta_R2 / R**2
                    else:  # Cambridge/Aachen
                        d_ij = delta_R2 / R**2

                    if d_ij < min_dist:
                        min_dist = d_ij
                        min_pair = (i, j)
                        is_beam = False

            # Perform clustering
            if is_beam or min_pair[1] == -1:
                # Promote to jet
                jets.append([remaining[min_pair[0]]])
                remaining.pop(min_pair[0])
            else:
                # Merge particles
                i, j = min_pair
                merged = remaining[i] + remaining[j]
                remaining[i] = merged
                remaining.pop(j)

        return jets

    def luminosity(self, n1: float, n2: float, f: float, A: float) -> float:
        """
        Calculate instantaneous luminosity for collider.

        Args:
            n1: Number of particles in bunch 1
            n2: Number of particles in bunch 2
            f: Collision frequency (Hz)
            A: Cross-sectional area (m²)

        Returns:
            Luminosity in cm⁻²s⁻¹
        """
        L = n1 * n2 * f / A
        return L * 1e-4  # Convert to cm⁻²s⁻¹


def run_demo():
    """Demonstrate particle physics calculations."""
    lab = ParticlePhysicsLab()
    print(f"Initializing {lab.name} v{lab.version}")
    print("=" * 60)

    # 1. Cross-section calculations
    print("\n1. Cross-Section Calculations:")

    # e⁺e⁻ → μ⁺μ⁻ at 10 GeV
    s = 100  # GeV²
    sigma = lab.calculate_cross_section('ee_to_mumu', s)
    print(f"   e⁺e⁻ → μ⁺μ⁻ at √s = 10 GeV: {sigma:.3f} mb")

    # e⁺e⁻ → hadrons at Z pole
    s_Z = MZ**2
    sigma_had = lab.calculate_cross_section('ee_to_hadrons', s_Z)
    print(f"   e⁺e⁻ → hadrons at Z pole: {sigma_had:.1f} mb")

    # 2. Decay rates and lifetimes
    print("\n2. Decay Rates and Lifetimes:")

    # Muon decay
    gamma_muon = lab.calculate_decay_rate('muon',
                                         ['electron', 'electron_neutrino', 'muon_neutrino'])
    tau_muon = HBAR / gamma_muon / GEV_TO_J
    print(f"   Muon lifetime: {tau_muon*1e6:.2f} μs")

    # Pion decay
    br_pion = lab.calculate_branching_ratio('pion+', ['muon', 'muon_neutrino'])
    print(f"   π⁺ → μ⁺νμ branching ratio: {br_pion:.3f}")

    # 3. Four-vectors and kinematics
    print("\n3. Kinematics and Mandelstam Variables:")

    # Two-body scattering
    p1 = FourVector(5, 0, 0, 4.899)  # Incoming particle 1
    p2 = FourVector(5, 0, 0, -4.899)  # Incoming particle 2
    p3 = FourVector(5, 3, 0, 3.999)  # Outgoing particle 1
    p4 = FourVector(5, -3, 0, -3.999)  # Outgoing particle 2

    s, t, u = lab.calculate_mandelstam_variables(p1, p2, p3, p4)
    print(f"   s = {s:.2f} GeV²")
    print(f"   t = {t:.2f} GeV²")
    print(f"   u = {u:.2f} GeV²")
    print(f"   s + t + u = {s + t + u:.2f} GeV² (should be ~0 for massless)")

    # 4. Parton distribution functions
    print("\n4. Parton Distribution Functions (x=0.1, Q²=100 GeV²):")

    x = 0.1
    Q2 = 100

    pdf_u = lab.parton_distribution_function(x, Q2, 'u')
    pdf_d = lab.parton_distribution_function(x, Q2, 'd')
    pdf_g = lab.parton_distribution_function(x, Q2, 'g')

    print(f"   u quark PDF: {pdf_u:.3f}")
    print(f"   d quark PDF: {pdf_d:.3f}")
    print(f"   gluon PDF: {pdf_g:.3f}")

    # 5. Jet clustering
    print("\n5. Jet Clustering (anti-kT algorithm):")

    # Create some particles
    particles = [
        FourVector(10, 8, 2, 5),
        FourVector(8, 7, 1, 4),
        FourVector(15, -10, -8, 5),
        FourVector(5, -3, -2, 3),
        FourVector(3, 0.5, 2, 2)
    ]

    jets = lab.jet_clustering(particles, R=0.4, algorithm='antikt')
    print(f"   Found {len(jets)} jets")

    for i, jet in enumerate(jets):
        pt_sum = sum(lab.transverse_momentum(p.px, p.py) for p in jet)
        print(f"   Jet {i+1}: {len(jet)} particles, pT = {pt_sum:.1f} GeV")

    # 6. Invariant mass
    print("\n6. Invariant Mass Calculations:")

    # Two particles forming a resonance
    p_resonance = [
        FourVector(50, 30, 0, 20),
        FourVector(45, -30, 0, -15)
    ]

    m_inv = lab.invariant_mass_spectrum(p_resonance)
    print(f"   Invariant mass of two-particle system: {m_inv:.1f} GeV")

    # 7. Missing transverse energy
    print("\n7. Missing Transverse Energy:")

    visible = [
        FourVector(40, 25, 10, 20),
        FourVector(30, -15, -5, 15),
        FourVector(20, -5, -8, 10)
    ]

    met, phi = lab.missing_transverse_energy(visible)
    print(f"   Missing ET: {met:.1f} GeV")
    print(f"   Missing φ: {phi:.2f} rad")

    # 8. Particle properties from PDG
    print("\n8. Particle Properties (PDG):")

    for name in ['electron', 'muon', 'top', 'W+', 'higgs']:
        p = lab.particles[name]
        print(f"   {p.name}: m = {p.mass:.3f} GeV, Q = {p.charge}e, spin = {p.spin}ℏ")

    # 9. Luminosity calculation
    print("\n9. Collider Luminosity:")

    # LHC-like parameters
    n_particles = 1.15e11  # Protons per bunch
    f_collision = 11245  # Hz (bunch crossing frequency)
    beam_area = np.pi * (16.7e-6)**2  # m² (beam size at IP)

    L = lab.luminosity(n_particles, n_particles, f_collision, beam_area)
    print(f"   Instantaneous luminosity: {L:.2e} cm⁻²s⁻¹")

    print("\n" + "=" * 60)
    print("Particle Physics Lab demonstration complete!")


if __name__ == "__main__":
    run_demo()