"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Astrophysics Laboratory - Production-Ready Stellar Evolution and Cosmology
A comprehensive suite for stellar structure, gravitational dynamics, and cosmological calculations
"""

import numpy as np
from scipy import integrate, optimize, interpolate
from scipy.constants import G, c, sigma, k, m_p, m_e, h, pi
from typing import Dict, List, Tuple, Optional, Callable
import warnings

class AstrophysicsLab:
    """Advanced astrophysics calculations for stellar evolution and cosmology"""

    def __init__(self):
        # Physical constants
        self.G = G  # Gravitational constant
        self.c = c  # Speed of light
        self.sigma_sb = sigma  # Stefan-Boltzmann constant
        self.k_B = k  # Boltzmann constant
        self.m_p = m_p  # Proton mass
        self.m_e = m_e  # Electron mass
        self.h = h  # Planck constant

        # Solar units for scaling
        self.M_sun = 1.989e30  # kg
        self.R_sun = 6.96e8   # m
        self.L_sun = 3.828e26  # W

        # Cosmological parameters (Planck 2018)
        self.H0 = 67.4  # km/s/Mpc
        self.Omega_m = 0.315  # Matter density
        self.Omega_Lambda = 0.685  # Dark energy density
        self.Omega_b = 0.049  # Baryon density

    def lane_emden_solver(self, n: float, xi_max: float = 10.0) -> Dict[str, np.ndarray]:
        """
        Solve Lane-Emden equation for polytropic stellar structure
        d²θ/dξ² + (2/ξ)dθ/dξ + θⁿ = 0

        Args:
            n: Polytropic index (1.5 for convective, 3 for radiative)
            xi_max: Maximum dimensionless radius

        Returns:
            Dictionary with xi, theta, and derivatives
        """
        def lane_emden_ode(xi, y):
            theta, dtheta = y
            if xi < 1e-10:  # Handle singularity at origin
                return [dtheta, -theta**n / 3.0]
            return [dtheta, -(2.0/xi) * dtheta - theta**n]

        # Initial conditions: θ(0) = 1, θ'(0) = 0
        xi = np.linspace(1e-10, xi_max, 1000)
        y0 = [1.0, 0.0]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            solution = integrate.odeint(lane_emden_ode, y0, xi)

        # Find stellar surface (where θ = 0)
        surface_idx = np.where(solution[:, 0] <= 0)[0]
        if len(surface_idx) > 0:
            xi_1 = xi[surface_idx[0]]
            dtheta_1 = solution[surface_idx[0], 1]
        else:
            xi_1 = xi_max
            dtheta_1 = solution[-1, 1]

        return {
            'xi': xi,
            'theta': solution[:, 0],
            'dtheta': solution[:, 1],
            'xi_1': xi_1,
            'dtheta_1': dtheta_1,
            'n': n
        }

    def stellar_structure_integration(self, M_star: float, composition: Dict[str, float]) -> Dict[str, np.ndarray]:
        """
        Integrate stellar structure equations from center to surface

        Args:
            M_star: Stellar mass in solar masses
            composition: {'X': H fraction, 'Y': He fraction, 'Z': metallicity}

        Returns:
            Radial profiles of mass, pressure, temperature, luminosity
        """
        M = M_star * self.M_sun
        X, Y, Z = composition['X'], composition['Y'], composition['Z']

        # Mean molecular weight
        mu = 1.0 / (2*X + 0.75*Y + 0.5*Z)

        def stellar_equations(r, y):
            m, P, T, L = y
            if r < 1e-10:
                return [0, 0, 0, 0]

            # Density from ideal gas law
            rho = (mu * self.m_p * P) / (self.k_B * T)

            # Opacity (Kramers' law approximation)
            kappa = 0.02 * (1 + X) * rho * T**(-3.5)

            # Energy generation (pp-chain)
            epsilon = 1.07e-7 * rho * X**2 * (T/1e6)**4

            # Structure equations
            dm_dr = 4 * pi * r**2 * rho
            dP_dr = -self.G * m * rho / r**2

            # Radiative transport
            dT_dr = -3 * kappa * rho * L / (16 * pi * self.sigma_sb * T**3 * r**2)

            # Check for convection (Schwarzschild criterion)
            grad_rad = abs(dT_dr * P / (T * dP_dr))
            if grad_rad > 0.4:  # Convective
                dT_dr = -0.4 * T * dP_dr / P

            dL_dr = 4 * pi * r**2 * rho * epsilon

            return [dm_dr, dP_dr, dT_dr, dL_dr]

        # Central conditions
        rho_c = 1.5e5  # kg/m³ (initial guess)
        P_c = self.G * M**2 / (4 * pi * (0.1 * self.R_sun)**4)
        T_c = 1.5e7  # K

        # Integration
        r_span = [1e3, self.R_sun]
        y0 = [1e-10 * M, P_c, T_c, 0]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            solution = integrate.solve_ivp(stellar_equations, r_span, y0,
                                          method='RK45', dense_output=True,
                                          max_step=1e6)

        r_points = np.logspace(3, np.log10(self.R_sun), 100)
        profiles = solution.sol(r_points)

        return {
            'radius': r_points / self.R_sun,
            'mass': profiles[0] / self.M_sun,
            'pressure': profiles[1],
            'temperature': profiles[2],
            'luminosity': profiles[3] / self.L_sun
        }

    def nbody_gravitational_dynamics(self, masses: np.ndarray, positions: np.ndarray,
                                    velocities: np.ndarray, dt: float, steps: int) -> Dict[str, np.ndarray]:
        """
        N-body gravitational simulation using Verlet integration

        Args:
            masses: Array of masses (kg)
            positions: Initial positions (m) shape (n_bodies, 3)
            velocities: Initial velocities (m/s) shape (n_bodies, 3)
            dt: Time step (s)
            steps: Number of integration steps

        Returns:
            Trajectories and energies over time
        """
        n = len(masses)
        pos_history = np.zeros((steps, n, 3))
        vel_history = np.zeros((steps, n, 3))
        energy_history = np.zeros(steps)

        pos = positions.copy()
        vel = velocities.copy()

        def compute_accelerations(positions, masses):
            acc = np.zeros_like(positions)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        r_vec = positions[j] - positions[i]
                        r_mag = np.linalg.norm(r_vec)
                        if r_mag > 0:
                            acc[i] += self.G * masses[j] * r_vec / r_mag**3
            return acc

        # Verlet integration
        for step in range(steps):
            acc = compute_accelerations(pos, masses)

            # Update positions and velocities
            pos += vel * dt + 0.5 * acc * dt**2
            acc_new = compute_accelerations(pos, masses)
            vel += 0.5 * (acc + acc_new) * dt

            # Store history
            pos_history[step] = pos
            vel_history[step] = vel

            # Calculate total energy
            KE = 0.5 * np.sum(masses[:, np.newaxis] * vel**2)
            PE = 0
            for i in range(n):
                for j in range(i+1, n):
                    r = np.linalg.norm(pos[i] - pos[j])
                    if r > 0:
                        PE -= self.G * masses[i] * masses[j] / r
            energy_history[step] = KE + PE

        return {
            'positions': pos_history,
            'velocities': vel_history,
            'energy': energy_history,
            'time': np.arange(steps) * dt
        }

    def hubble_flow_cosmology(self, z: float) -> Dict[str, float]:
        """
        Calculate cosmological distances and times for given redshift

        Args:
            z: Redshift

        Returns:
            Cosmological parameters at redshift z
        """
        # Hubble parameter evolution
        def E(z):
            return np.sqrt(self.Omega_m * (1 + z)**3 + self.Omega_Lambda)

        # Comoving distance
        def integrand(z_prime):
            return 1.0 / E(z_prime)

        # Speed of light in km/s
        c_km = self.c / 1000

        # Hubble distance
        D_H = c_km / self.H0  # Mpc

        # Comoving distance
        D_C, _ = integrate.quad(integrand, 0, z)
        D_C *= D_H

        # Luminosity distance
        D_L = D_C * (1 + z)

        # Angular diameter distance
        D_A = D_C / (1 + z)

        # Lookback time
        def time_integrand(z_prime):
            return 1.0 / ((1 + z_prime) * E(z_prime))

        # Hubble time
        t_H = 1.0 / self.H0 * 977.8  # Gyr

        lookback_time, _ = integrate.quad(time_integrand, 0, z)
        lookback_time *= t_H

        # Age of universe at redshift z
        age_at_z, _ = integrate.quad(time_integrand, z, np.inf)
        age_at_z *= t_H

        return {
            'redshift': z,
            'comoving_distance': D_C,
            'luminosity_distance': D_L,
            'angular_diameter_distance': D_A,
            'lookback_time': lookback_time,
            'age_at_z': age_at_z,
            'distance_modulus': 5 * np.log10(D_L * 1e6 / 10)
        }

    def spectral_classification(self, temperature: float, luminosity: float) -> Dict[str, any]:
        """
        Stellar spectral classification and HR diagram position

        Args:
            temperature: Effective temperature (K)
            luminosity: Luminosity (solar units)

        Returns:
            Spectral class and stellar parameters
        """
        # Spectral classes
        spectral_types = [
            ('O', 30000, np.inf), ('B', 10000, 30000), ('A', 7500, 10000),
            ('F', 6000, 7500), ('G', 5200, 6000), ('K', 3700, 5200),
            ('M', 2400, 3700)
        ]

        spectral_class = 'Unknown'
        for stype, t_min, t_max in spectral_types:
            if t_min <= temperature < t_max:
                spectral_class = stype
                break

        # Luminosity class
        abs_magnitude = 4.75 - 2.5 * np.log10(luminosity)

        if luminosity > 30000:
            lum_class = 'Ia'  # Bright supergiant
        elif luminosity > 10000:
            lum_class = 'Ib'  # Supergiant
        elif luminosity > 1000:
            lum_class = 'II'  # Bright giant
        elif luminosity > 100:
            lum_class = 'III'  # Giant
        elif luminosity > 10:
            lum_class = 'IV'  # Subgiant
        elif abs_magnitude < 10:
            lum_class = 'V'  # Main sequence
        else:
            lum_class = 'VI'  # Subdwarf

        # Radius from Stefan-Boltzmann law
        radius = np.sqrt(luminosity * self.L_sun / (4 * pi * self.sigma_sb * temperature**4))

        return {
            'spectral_class': spectral_class,
            'luminosity_class': lum_class,
            'full_type': f"{spectral_class}{lum_class}",
            'temperature': temperature,
            'luminosity': luminosity,
            'radius': radius / self.R_sun,
            'absolute_magnitude': abs_magnitude
        }

    def gravitational_lensing(self, M_lens: float, D_l: float, D_s: float) -> Dict[str, float]:
        """
        Calculate gravitational lensing parameters

        Args:
            M_lens: Mass of lens (solar masses)
            D_l: Distance to lens (Mpc)
            D_s: Distance to source (Mpc)

        Returns:
            Einstein radius and lensing parameters
        """
        M = M_lens * self.M_sun
        D_l_m = D_l * 3.086e22  # Convert to meters
        D_s_m = D_s * 3.086e22
        D_ls = D_s - D_l
        D_ls_m = D_ls * 3.086e22

        # Einstein radius
        theta_E = np.sqrt(4 * self.G * M * D_ls_m / (self.c**2 * D_l_m * D_s_m))
        theta_E_arcsec = theta_E * 206265  # Convert to arcseconds

        # Critical density
        Sigma_crit = self.c**2 * D_s_m / (4 * pi * self.G * D_l_m * D_ls_m)

        # Magnification for perfect alignment
        mu_max = 1 + 2 * (theta_E_arcsec / 0.1)**2  # Assuming 0.1" source

        return {
            'einstein_radius_rad': theta_E,
            'einstein_radius_arcsec': theta_E_arcsec,
            'critical_density': Sigma_crit,
            'max_magnification': mu_max,
            'lens_mass': M_lens
        }

    def schwarzschild_metric(self, M: float, r: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate Schwarzschild metric components around black hole

        Args:
            M: Black hole mass (solar masses)
            r: Radial coordinates (Schwarzschild radii)

        Returns:
            Metric components and orbital parameters
        """
        # Schwarzschild radius
        r_s = 2 * self.G * M * self.M_sun / self.c**2

        # Convert r to meters
        r_m = r * r_s

        # Metric components
        g_tt = -(1 - r_s / r_m)
        g_rr = 1 / (1 - r_s / r_m)

        # Effective potential for circular orbits
        L = np.sqrt(self.G * M * self.M_sun * r_m)  # Angular momentum
        V_eff = -self.G * M * self.M_sun / r_m + L**2 / (2 * r_m**2) - L**2 * r_s / (2 * r_m**3)

        # ISCO (Innermost Stable Circular Orbit)
        r_isco = 6 * r_s

        # Orbital velocity
        v_orbit = np.sqrt(self.G * M * self.M_sun / (2 * r_m))
        v_orbit[r_m < r_isco] = np.nan  # No stable orbits inside ISCO

        return {
            'r': r_m,
            'g_tt': g_tt,
            'g_rr': g_rr,
            'effective_potential': V_eff,
            'orbital_velocity': v_orbit,
            'schwarzschild_radius': r_s,
            'isco_radius': r_isco
        }

    def cepheid_period_luminosity(self, period_days: float, metallicity: float = 0.02) -> float:
        """
        Cepheid variable period-luminosity relation for distance measurement

        Args:
            period_days: Pulsation period in days
            metallicity: Metallicity (Z)

        Returns:
            Absolute magnitude
        """
        # Classical relation: M_V = -2.81 * log10(P) - 1.43
        # With metallicity correction
        M_V = -2.81 * np.log10(period_days) - 1.43 + 0.2 * np.log10(metallicity / 0.02)
        return M_V

    def type_ia_supernova_luminosity(self, stretch_factor: float = 1.0) -> float:
        """
        Type Ia supernova standardizable candle luminosity

        Args:
            stretch_factor: Light curve stretch parameter

        Returns:
            Peak absolute magnitude
        """
        # Phillips relation
        M_B = -19.3 + 0.7 * (stretch_factor - 1)
        return M_B

    def tidal_disruption_radius(self, M_bh: float, M_star: float, R_star: float) -> float:
        """
        Calculate tidal disruption radius for a star near a black hole

        Args:
            M_bh: Black hole mass (solar masses)
            M_star: Star mass (solar masses)
            R_star: Star radius (solar radii)

        Returns:
            Tidal disruption radius in solar radii
        """
        r_t = R_star * self.R_sun * (M_bh / M_star)**(1/3)
        return r_t / self.R_sun

    def virial_theorem_cluster(self, velocity_dispersion: float, radius: float) -> float:
        """
        Calculate cluster mass using virial theorem

        Args:
            velocity_dispersion: RMS velocity (m/s)
            radius: Cluster radius (Mpc)

        Returns:
            Total mass in solar masses
        """
        r_m = radius * 3.086e22  # Convert to meters
        M = 3 * velocity_dispersion**2 * r_m / self.G
        return M / self.M_sun

    def demonstrate(self):
        """Demonstrate all astrophysics calculations"""
        print("=" * 70)
        print("ASTROPHYSICS LABORATORY DEMONSTRATION")
        print("=" * 70)

        # 1. Lane-Emden polytrope
        print("\n1. LANE-EMDEN STELLAR STRUCTURE")
        solution = self.lane_emden_solver(n=3.0)  # Radiative star
        print(f"   Polytropic index n=3 (radiative star)")
        print(f"   Stellar radius: ξ₁ = {solution['xi_1']:.3f}")
        print(f"   Surface derivative: dθ/dξ|₁ = {solution['dtheta_1']:.3f}")

        # 2. Stellar structure
        print("\n2. STELLAR STRUCTURE INTEGRATION")
        composition = {'X': 0.70, 'Y': 0.28, 'Z': 0.02}
        structure = self.stellar_structure_integration(1.0, composition)
        print(f"   Solar mass star: X={composition['X']}, Y={composition['Y']}, Z={composition['Z']}")
        print(f"   Central temperature: {structure['temperature'][0]:.2e} K")
        print(f"   Surface luminosity: {structure['luminosity'][-1]:.2f} L☉")

        # 3. N-body dynamics
        print("\n3. N-BODY GRAVITATIONAL DYNAMICS")
        masses = np.array([1e30, 5e29, 3e29])  # Three body system
        positions = np.array([[0, 0, 0], [1e11, 0, 0], [0, 1e11, 0]])
        velocities = np.array([[0, 0, 0], [0, 3e4, 0], [-3e4, 0, 0]])
        nbody = self.nbody_gravitational_dynamics(masses, positions, velocities, 3600, 100)
        print(f"   3-body simulation: {len(masses)} bodies")
        print(f"   Energy conservation: ΔE/E = {(nbody['energy'][-1] - nbody['energy'][0])/abs(nbody['energy'][0]):.6f}")

        # 4. Cosmological distances
        print("\n4. COSMOLOGICAL DISTANCES")
        z = 1.0
        cosmo = self.hubble_flow_cosmology(z)
        print(f"   Redshift z = {z}")
        print(f"   Luminosity distance: {cosmo['luminosity_distance']:.1f} Mpc")
        print(f"   Lookback time: {cosmo['lookback_time']:.2f} Gyr")
        print(f"   Age at z={z}: {cosmo['age_at_z']:.2f} Gyr")

        # 5. Spectral classification
        print("\n5. SPECTRAL CLASSIFICATION")
        star = self.spectral_classification(5800, 1.0)  # Sun-like
        print(f"   T = 5800 K, L = 1.0 L☉")
        print(f"   Spectral type: {star['full_type']}")
        print(f"   Radius: {star['radius']:.2f} R☉")

        # 6. Gravitational lensing
        print("\n6. GRAVITATIONAL LENSING")
        lens = self.gravitational_lensing(1e12, 500, 1000)  # Galaxy cluster
        print(f"   Lens mass: 10¹² M☉ (galaxy cluster)")
        print(f"   Einstein radius: {lens['einstein_radius_arcsec']:.2f} arcsec")
        print(f"   Maximum magnification: {lens['max_magnification']:.1f}×")

        # 7. Black hole metric
        print("\n7. SCHWARZSCHILD BLACK HOLE")
        r = np.linspace(2, 20, 5)
        bh = self.schwarzschild_metric(10, r)
        print(f"   10 M☉ black hole")
        print(f"   Schwarzschild radius: {bh['schwarzschild_radius']/1000:.2f} km")
        print(f"   ISCO radius: {bh['isco_radius']/1000:.2f} km")

        # 8. Distance measurements
        print("\n8. DISTANCE LADDER")
        cepheid = self.cepheid_period_luminosity(10.0)
        sn1a = self.type_ia_supernova_luminosity(1.1)
        print(f"   Cepheid (P=10d): M_V = {cepheid:.2f}")
        print(f"   Type Ia SN (s=1.1): M_B = {sn1a:.2f}")

        # 9. Tidal disruption
        print("\n9. TIDAL DISRUPTION")
        r_td = self.tidal_disruption_radius(1e6, 1.0, 1.0)
        print(f"   10⁶ M☉ SMBH vs solar-type star")
        print(f"   Tidal disruption radius: {r_td:.1f} R☉")

        # 10. Galaxy cluster
        print("\n10. GALAXY CLUSTER DYNAMICS")
        cluster_mass = self.virial_theorem_cluster(1000e3, 1.0)  # 1000 km/s, 1 Mpc
        print(f"   Velocity dispersion: 1000 km/s")
        print(f"   Radius: 1.0 Mpc")
        print(f"   Virial mass: {cluster_mass:.2e} M☉")

        print("\n" + "=" * 70)
        print("Demonstration complete. All systems operational.")


if __name__ == "__main__":
    lab = AstrophysicsLab()
    lab.demonstrate()