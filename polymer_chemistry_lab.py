"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

POLYMER CHEMISTRY LAB - Advanced Production-Ready Implementation
Free gift to the scientific community from QuLabInfinite.

This module provides comprehensive polymer chemistry simulation capabilities including:
- Polymerization kinetics (chain growth, step growth, living)
- Molecular weight distributions (Schulz-Zimm, Poisson)
- Glass transition temperature prediction
- Mechanical properties (Young's modulus, tensile strength)
- Flory-Huggins solution theory
- Mark-Houwink viscosity relationships
- Polymer crystallization kinetics (Avrami)
- Rubber elasticity theory
- Degradation and aging models
- Copolymerization kinetics
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable
from scipy.constants import k as kb, R, Avogadro
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize, fsolve
from scipy.special import gamma
import warnings

# Physical constants
BOLTZMANN_CONSTANT = kb  # J/K
GAS_CONSTANT = R  # J/(mol·K)
AVOGADRO_NUMBER = Avogadro

@dataclass
class Monomer:
    """Represents a monomer unit."""
    name: str
    molecular_weight: float  # g/mol
    density: float  # g/cm³
    functionality: int  # Number of reactive groups
    reactivity_ratio: float = 1.0  # For copolymerization
    glass_transition: float = 273.15  # K

@dataclass
class Polymer:
    """Represents a polymer with structural and physical properties."""
    name: str
    monomers: List[Monomer]
    degree_of_polymerization: float
    polydispersity: float = 1.0
    tacticity: str = 'atactic'  # isotactic, syndiotactic, atactic
    branching_degree: float = 0.0
    crystallinity: float = 0.0  # Fraction crystalline
    molecular_weight_n: Optional[float] = None  # Number average MW
    molecular_weight_w: Optional[float] = None  # Weight average MW

    def __post_init__(self):
        """Calculate derived properties."""
        if self.molecular_weight_n is None:
            monomer_mw = np.mean([m.molecular_weight for m in self.monomers])
            self.molecular_weight_n = self.degree_of_polymerization * monomer_mw

        if self.molecular_weight_w is None:
            self.molecular_weight_w = self.molecular_weight_n * self.polydispersity

@dataclass
class PolymerizationConditions:
    """Conditions for polymerization reaction."""
    temperature: float  # K
    pressure: float  # Pa
    solvent: Optional[str] = None
    initiator_concentration: float = 1e-3  # mol/L
    monomer_concentration: float = 1.0  # mol/L
    time: float = 3600  # s

class PolymerChemistryLab:
    """Advanced polymer chemistry laboratory for comprehensive polymer simulations."""

    def __init__(self, temperature: float = 298.15):
        """Initialize polymer chemistry lab."""
        self.temperature = temperature
        self.R = GAS_CONSTANT

    # ==================== POLYMERIZATION KINETICS ====================

    def chain_growth_polymerization(self, monomer: Monomer, conditions: PolymerizationConditions,
                                   kp: float = 1e3, kt: float = 1e7,
                                   ki: float = 1e-4) -> Dict:
        """
        Model free radical chain growth polymerization.

        Includes initiation, propagation, and termination steps.
        """
        def polymerization_odes(y, t, ki, kp, kt):
            """ODEs for radical polymerization."""
            I, M, R, P = y  # Initiator, Monomer, Radical, Polymer

            # Rate equations
            r_init = ki * I  # Initiation
            r_prop = kp * R * M  # Propagation
            r_term = kt * R**2  # Termination (combination)

            # ODEs
            dI_dt = -r_init
            dM_dt = -r_prop
            dR_dt = 2 * r_init - 2 * r_term  # Factor of 2 from initiator efficiency
            dP_dt = r_term

            return [dI_dt, dM_dt, dR_dt, dP_dt]

        # Initial conditions
        y0 = [conditions.initiator_concentration,
              conditions.monomer_concentration,
              0.0, 0.0]  # No radicals or polymer initially

        # Time points
        t = np.linspace(0, conditions.time, 1000)

        # Solve ODEs
        solution = odeint(polymerization_odes, y0, t, args=(ki, kp, kt))

        # Extract final values
        I_final, M_final, R_final, P_final = solution[-1]

        # Calculate conversion
        conversion = 1 - M_final / conditions.monomer_concentration

        # Calculate average degree of polymerization (kinetic chain length)
        v = kp / (2 * kt)**0.5
        DP_n = v * (conditions.monomer_concentration / conditions.initiator_concentration)**0.5

        # Molecular weight distribution (most probable for radical polymerization)
        PDI = 2.0  # Theoretical for termination by combination

        return {
            'time': t,
            'conversion': conversion,
            'initiator': solution[:, 0],
            'monomer': solution[:, 1],
            'radical': solution[:, 2],
            'polymer': solution[:, 3],
            'degree_of_polymerization': DP_n,
            'polydispersity': PDI
        }

    def step_growth_polymerization(self, monomers: List[Monomer],
                                  conditions: PolymerizationConditions,
                                  k: float = 1e-3) -> Dict:
        """
        Model step-growth (condensation) polymerization.

        AA + BB → AA-BB + byproduct
        """
        def carothers_equation(p, functionality=2):
            """Carothers equation for degree of polymerization."""
            if functionality == 2:
                return 1 / (1 - p)
            else:
                # Modified for functionality > 2
                return (2 - p * functionality) / (2 - 2 * p)

        # Calculate extent of reaction over time (2nd order kinetics)
        t = np.linspace(0, conditions.time, 1000)
        M0 = conditions.monomer_concentration

        # For equal stoichiometry
        p = k * M0 * t / (1 + k * M0 * t)  # Conversion

        # Degree of polymerization
        avg_functionality = np.mean([m.functionality for m in monomers])
        DP_n = np.array([carothers_equation(pi, avg_functionality) for pi in p])

        # Molecular weight distribution (Flory distribution)
        PDI = 1 + p  # For linear step-growth

        return {
            'time': t,
            'conversion': p,
            'degree_of_polymerization': DP_n,
            'polydispersity': PDI,
            'number_average_mw': DP_n * monomers[0].molecular_weight
        }

    def living_polymerization(self, monomer: Monomer, conditions: PolymerizationConditions,
                            kp: float = 100) -> Dict:
        """
        Model living/controlled polymerization (e.g., anionic, RAFT, ATRP).

        No termination, narrow molecular weight distribution.
        """
        # First-order kinetics
        t = np.linspace(0, conditions.time, 1000)

        # Monomer consumption
        M = conditions.monomer_concentration * np.exp(-kp * conditions.initiator_concentration * t)

        # Conversion
        conversion = 1 - M / conditions.monomer_concentration

        # Degree of polymerization (all chains grow equally)
        DP_n = conversion * conditions.monomer_concentration / conditions.initiator_concentration

        # Narrow distribution (Poisson)
        PDI = 1 + 1 / DP_n  # Approaches 1 for high DP

        return {
            'time': t,
            'monomer': M,
            'conversion': conversion,
            'degree_of_polymerization': DP_n,
            'polydispersity': PDI
        }

    # ==================== MOLECULAR WEIGHT DISTRIBUTIONS ====================

    def schulz_zimm_distribution(self, MW_avg: float, PDI: float,
                                MW_range: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Schulz-Zimm molecular weight distribution.

        Common for radical and condensation polymerizations.
        """
        if MW_range is None:
            MW_range = np.linspace(MW_avg/10, MW_avg*5, 500)

        # Schulz-Zimm parameter
        a = 1 / (PDI - 1)

        # Distribution
        y = a**a / gamma(a) * (MW_range / MW_avg)**(a-1) * np.exp(-a * MW_range / MW_avg) / MW_avg

        return MW_range, y

    def poisson_distribution(self, DP_avg: float, DP_range: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Poisson distribution for living polymerization.

        Very narrow distribution.
        """
        if DP_range is None:
            DP_range = np.arange(max(1, int(DP_avg - 5*np.sqrt(DP_avg))),
                                int(DP_avg + 5*np.sqrt(DP_avg)))

        from scipy.stats import poisson
        y = poisson.pmf(DP_range, DP_avg)

        return DP_range, y

    def log_normal_distribution(self, MW_avg: float, sigma: float = 0.5,
                               MW_range: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate log-normal molecular weight distribution.

        Common for some polymerization systems.
        """
        if MW_range is None:
            MW_range = np.linspace(MW_avg/10, MW_avg*5, 500)

        mu = np.log(MW_avg) - sigma**2 / 2

        y = 1 / (MW_range * sigma * np.sqrt(2*np.pi)) * \
            np.exp(-(np.log(MW_range) - mu)**2 / (2*sigma**2))

        return MW_range, y

    # ==================== GLASS TRANSITION TEMPERATURE ====================

    def fox_equation(self, components: List[Tuple[float, float]]) -> float:
        """
        Calculate Tg of polymer blend/copolymer using Fox equation.

        1/Tg = Σ(wi/Tgi)
        """
        Tg_inv = sum(w / Tg for w, Tg in components)
        return 1 / Tg_inv

    def gordon_taylor_equation(self, Tg1: float, Tg2: float, w1: float, k: float = 1.0) -> float:
        """
        Calculate Tg using Gordon-Taylor equation.

        Tg = (w1*Tg1 + k*w2*Tg2) / (w1 + k*w2)
        """
        w2 = 1 - w1
        return (w1 * Tg1 + k * w2 * Tg2) / (w1 + k * w2)

    def free_volume_theory_tg(self, molecular_weight: float, K: float = 1e5) -> float:
        """
        Estimate Tg from molecular weight using free volume theory.

        Tg = Tg∞ - K/Mn (Flory-Fox equation)
        """
        Tg_inf = 380  # Asymptotic Tg for high MW (polymer dependent)
        return Tg_inf - K / molecular_weight

    # ==================== MECHANICAL PROPERTIES ====================

    def young_modulus(self, polymer: Polymer, temperature: Optional[float] = None) -> float:
        """
        Estimate Young's modulus based on polymer properties.
        """
        T = temperature if temperature else self.temperature

        # Base modulus (GPa)
        if polymer.crystallinity > 0.5:
            E_base = 2.0  # Semi-crystalline
        else:
            E_base = 0.01  # Rubbery/amorphous

        # Temperature dependence (WLF-like)
        Tg = self.estimate_tg(polymer)

        if T < Tg:
            # Glassy state
            E = E_base * 100
        elif T < Tg + 50:
            # Transition region
            E = E_base * 10 * np.exp(-(T - Tg) / 20)
        else:
            # Rubbery state
            E = E_base

        # Crystallinity effect
        E *= (1 + 2 * polymer.crystallinity)

        return E * 1e9  # Convert to Pa

    def tensile_strength(self, polymer: Polymer, strain_rate: float = 0.001) -> float:
        """
        Estimate tensile strength.

        Depends on MW, crystallinity, and strain rate.
        """
        # Base strength (MPa)
        sigma_base = 50

        # Molecular weight effect
        MW_factor = np.tanh(polymer.molecular_weight_n / 50000)

        # Crystallinity effect
        crystal_factor = 1 + polymer.crystallinity

        # Strain rate effect (power law)
        rate_factor = (strain_rate / 0.001)**0.1

        sigma = sigma_base * MW_factor * crystal_factor * rate_factor

        return sigma * 1e6  # Convert to Pa

    def rubber_elasticity(self, crosslink_density: float, temperature: Optional[float] = None) -> float:
        """
        Calculate shear modulus for rubber using statistical mechanics.

        G = ρRT/Mc where Mc is molecular weight between crosslinks
        """
        T = temperature if temperature else self.temperature

        # Neo-Hookean model
        G = crosslink_density * self.R * T

        return G

    def stress_relaxation(self, initial_stress: float, time: np.ndarray,
                         relaxation_times: List[float], weights: List[float]) -> np.ndarray:
        """
        Model stress relaxation using Maxwell model.

        σ(t) = σ0 * Σ wi * exp(-t/τi)
        """
        stress = np.zeros_like(time)

        for w, tau in zip(weights, relaxation_times):
            stress += w * np.exp(-time / tau)

        return initial_stress * stress

    # ==================== SOLUTION THERMODYNAMICS ====================

    def flory_huggins_free_energy(self, phi: float, chi: float, N: float) -> float:
        """
        Calculate Flory-Huggins free energy of mixing.

        ΔGmix/RT = φ*ln(φ) + (1-φ)*ln(1-φ)/N + χ*φ*(1-φ)
        """
        if phi <= 0 or phi >= 1:
            return np.inf

        dG = phi * np.log(phi) + (1 - phi) * np.log(1 - phi) / N + chi * phi * (1 - phi)

        return dG * self.R * self.temperature

    def solubility_parameter(self, cohesive_energy: float, molar_volume: float) -> float:
        """
        Calculate Hildebrand solubility parameter.

        δ = (E_coh/V_m)^0.5
        """
        return np.sqrt(cohesive_energy / molar_volume)

    def mark_houwink_viscosity(self, molecular_weight: float, K: float = 1e-4, a: float = 0.7) -> float:
        """
        Calculate intrinsic viscosity using Mark-Houwink equation.

        [η] = K*M^a
        """
        return K * molecular_weight**a

    def theta_temperature(self, enthalpy: float, entropy: float) -> float:
        """
        Calculate theta temperature (Flory temperature).

        At θ: Second virial coefficient = 0
        """
        return enthalpy / entropy

    # ==================== CRYSTALLIZATION KINETICS ====================

    def avrami_crystallization(self, time: np.ndarray, k: float = 1e-3, n: float = 3) -> np.ndarray:
        """
        Model crystallization kinetics using Avrami equation.

        X(t) = 1 - exp(-k*t^n)
        """
        crystallinity = 1 - np.exp(-k * time**n)
        return crystallinity

    def hoffman_weeks_melting(self, crystallization_temp: float, equilibrium_melting: float = 450) -> float:
        """
        Calculate melting temperature using Hoffman-Weeks equation.

        Tm = Tc/2 + Tm°/2
        """
        return crystallization_temp / 2 + equilibrium_melting / 2

    def lauritzen_hoffman_growth(self, temperature: float, Tg: float, Tm: float,
                                U: float = 1500, K_g: float = 1e5) -> float:
        """
        Calculate crystal growth rate using Lauritzen-Hoffman theory.
        """
        if temperature <= Tg or temperature >= Tm:
            return 0.0

        # WLF-type transport term
        transport = np.exp(-U / (self.R * (temperature - Tg + 30)))

        # Nucleation term
        undercooling = Tm - temperature
        nucleation = np.exp(-K_g / (temperature * undercooling))

        G = transport * nucleation

        return G

    # ==================== COPOLYMERIZATION ====================

    def copolymer_composition(self, f1: float, r1: float, r2: float) -> float:
        """
        Calculate instantaneous copolymer composition.

        Mayo-Lewis equation: F1 = (r1*f1² + f1*f2) / (r1*f1² + 2*f1*f2 + r2*f2²)
        """
        f2 = 1 - f1

        F1 = (r1 * f1**2 + f1 * f2) / (r1 * f1**2 + 2 * f1 * f2 + r2 * f2**2)

        return F1

    def sequence_distribution(self, F1: float, r1: float, r2: float) -> Dict:
        """
        Calculate sequence length distributions in copolymer.
        """
        F2 = 1 - F1

        # Average sequence lengths
        L1_avg = 1 + r1 * F1 / F2
        L2_avg = 1 + r2 * F2 / F1

        # Run number (transitions per 100 units)
        R = 200 * F1 * F2 / (r1 * F1**2 + 2 * F1 * F2 + r2 * F2**2)

        return {
            'avg_sequence_length_1': L1_avg,
            'avg_sequence_length_2': L2_avg,
            'run_number': R,
            'alternation_tendency': 1 / (r1 * r2)  # <1 alternating, >1 blocky
        }

    # ==================== DEGRADATION ====================

    def thermal_degradation(self, time: np.ndarray, activation_energy: float = 100000,
                          temperature: Optional[float] = None) -> np.ndarray:
        """
        Model thermal degradation kinetics.

        First-order random chain scission.
        """
        T = temperature if temperature else self.temperature

        # Arrhenius rate constant
        k = 1e13 * np.exp(-activation_energy / (self.R * T))

        # Molecular weight decrease
        MW_ratio = np.exp(-k * time)

        return MW_ratio

    def photo_oxidation(self, time: np.ndarray, uv_intensity: float = 1.0,
                       oxygen_concentration: float = 0.21) -> np.ndarray:
        """
        Model photo-oxidative degradation.
        """
        # Simplified model
        k_photo = 1e-6 * uv_intensity * oxygen_concentration

        # Carbonyl index growth
        carbonyl_index = 1 - np.exp(-k_photo * time)

        return carbonyl_index

    def hydrolytic_degradation(self, time: np.ndarray, pH: float = 7.0,
                             temperature: Optional[float] = None) -> np.ndarray:
        """
        Model hydrolytic degradation (e.g., polyesters).
        """
        T = temperature if temperature else self.temperature

        # pH-dependent rate
        if pH < 4 or pH > 10:
            k_base = 1e-5  # Acid/base catalyzed
        else:
            k_base = 1e-7  # Neutral

        # Temperature dependence
        k = k_base * np.exp(-60000 / (self.R * T))

        # Molecular weight decrease
        MW_ratio = 1 / (1 + k * time)

        return MW_ratio

    # ==================== HELPER METHODS ====================

    def estimate_tg(self, polymer: Polymer) -> float:
        """Estimate glass transition temperature."""
        # Simple estimation based on monomer Tg
        monomer_tgs = [m.glass_transition for m in polymer.monomers]
        weights = [1/len(monomer_tgs) for _ in monomer_tgs]

        components = list(zip(weights, monomer_tgs))
        base_tg = self.fox_equation(components)

        # Molecular weight correction
        tg = self.free_volume_theory_tg(polymer.molecular_weight_n)

        return min(base_tg, tg)

    # ==================== DEMONSTRATION ====================

    def run_comprehensive_demo(self):
        """Run comprehensive demonstration of polymer chemistry capabilities."""
        print("=" * 60)
        print("POLYMER CHEMISTRY LAB - Comprehensive Demonstration")
        print("=" * 60)

        # 1. Define monomers
        print("\n1. MONOMER PROPERTIES")
        print("-" * 40)

        styrene = Monomer(
            name="Styrene",
            molecular_weight=104.15,
            density=0.906,
            functionality=1,
            reactivity_ratio=0.52,
            glass_transition=373
        )

        mma = Monomer(
            name="MMA",
            molecular_weight=100.12,
            density=0.94,
            functionality=1,
            reactivity_ratio=0.46,
            glass_transition=378
        )

        print(f"Monomer 1: {styrene.name}, MW = {styrene.molecular_weight} g/mol")
        print(f"Monomer 2: {mma.name}, MW = {mma.molecular_weight} g/mol")

        # 2. Chain Growth Polymerization
        print("\n2. CHAIN GROWTH POLYMERIZATION")
        print("-" * 40)

        conditions = PolymerizationConditions(
            temperature=343.15,  # 70°C
            pressure=101325,
            monomer_concentration=5.0,
            initiator_concentration=0.01,
            time=7200
        )

        result = self.chain_growth_polymerization(styrene, conditions)

        print(f"Final conversion: {result['conversion']:.2%}")
        print(f"Degree of polymerization: {result['degree_of_polymerization']:.0f}")
        print(f"Polydispersity index: {result['polydispersity']:.2f}")

        # 3. Step Growth Polymerization
        print("\n3. STEP GROWTH POLYMERIZATION")
        print("-" * 40)

        diacid = Monomer("Adipic acid", 146.14, 1.36, 2)
        diamine = Monomer("Hexamethylenediamine", 116.20, 0.84, 2)

        result_step = self.step_growth_polymerization([diacid, diamine], conditions, k=0.01)

        final_idx = -1
        print(f"Final conversion: {result_step['conversion'][final_idx]:.2%}")
        print(f"Final DP: {result_step['degree_of_polymerization'][final_idx]:.0f}")
        print(f"Final PDI: {result_step['polydispersity'][final_idx]:.2f}")

        # 4. Living Polymerization
        print("\n4. LIVING POLYMERIZATION")
        print("-" * 40)

        result_living = self.living_polymerization(styrene, conditions, kp=0.001)

        print(f"Final conversion: {result_living['conversion'][-1]:.2%}")
        print(f"Final DP: {result_living['degree_of_polymerization'][-1]:.0f}")
        print(f"Final PDI: {result_living['polydispersity'][-1]:.3f}")

        # 5. Molecular Weight Distribution
        print("\n5. MOLECULAR WEIGHT DISTRIBUTION")
        print("-" * 40)

        MW_avg = 50000
        PDI = 2.0

        MW_range, distribution = self.schulz_zimm_distribution(MW_avg, PDI)
        peak_MW = MW_range[np.argmax(distribution)]

        print(f"Schulz-Zimm distribution:")
        print(f"  Average MW: {MW_avg} g/mol")
        print(f"  PDI: {PDI}")
        print(f"  Peak MW: {peak_MW:.0f} g/mol")

        # 6. Glass Transition
        print("\n6. GLASS TRANSITION TEMPERATURE")
        print("-" * 40)

        # Copolymer Tg
        components = [(0.5, 373), (0.5, 378)]  # 50:50 PS:PMMA
        Tg_fox = self.fox_equation(components)

        print(f"Copolymer Tg (Fox equation): {Tg_fox:.1f} K ({Tg_fox-273:.1f}°C)")

        # MW dependence
        Tg_low = self.free_volume_theory_tg(10000)
        Tg_high = self.free_volume_theory_tg(100000)

        print(f"Tg at MW=10k: {Tg_low:.1f} K")
        print(f"Tg at MW=100k: {Tg_high:.1f} K")

        # 7. Mechanical Properties
        print("\n7. MECHANICAL PROPERTIES")
        print("-" * 40)

        polymer = Polymer(
            name="Polystyrene",
            monomers=[styrene],
            degree_of_polymerization=500,
            polydispersity=2.0,
            crystallinity=0.0
        )

        E = self.young_modulus(polymer, 298)
        sigma = self.tensile_strength(polymer)

        print(f"Young's modulus: {E/1e9:.2f} GPa")
        print(f"Tensile strength: {sigma/1e6:.1f} MPa")

        # Rubber elasticity
        G = self.rubber_elasticity(crosslink_density=100)  # mol/m³
        print(f"Rubber shear modulus: {G/1e3:.1f} kPa")

        # 8. Solution Properties
        print("\n8. SOLUTION THERMODYNAMICS")
        print("-" * 40)

        # Flory-Huggins
        phi = 0.1  # Volume fraction
        chi = 0.45  # Interaction parameter
        N = 1000  # Degree of polymerization

        dG = self.flory_huggins_free_energy(phi, chi, N)
        print(f"Flory-Huggins ΔGmix: {dG:.1f} J/mol")

        # Mark-Houwink viscosity
        eta = self.mark_houwink_viscosity(50000)
        print(f"Intrinsic viscosity: {eta:.3e} dL/g")

        # 9. Crystallization
        print("\n9. CRYSTALLIZATION KINETICS")
        print("-" * 40)

        time = np.array([0, 10, 30, 60, 120, 300, 600])  # seconds
        crystallinity = self.avrami_crystallization(time, k=1e-4, n=3)

        print("Avrami crystallization:")
        for t, x in zip(time, crystallinity):
            print(f"  t={t}s: X={x:.3f}")

        # 10. Degradation
        print("\n10. POLYMER DEGRADATION")
        print("-" * 40)

        time_days = np.array([0, 30, 90, 180, 365])
        time_seconds = time_days * 24 * 3600

        MW_ratio = self.thermal_degradation(time_seconds, temperature=373)

        print("Thermal degradation at 100°C:")
        for t, ratio in zip(time_days, MW_ratio):
            print(f"  {t} days: MW/MW0 = {ratio:.3f}")

        print("\n" + "=" * 60)
        print("Demonstration complete!")
        print("=" * 60)

def run_demo():
    """Run the comprehensive polymer chemistry demonstration."""
    lab = PolymerChemistryLab()
    lab.run_comprehensive_demo()

if __name__ == '__main__':
    run_demo()