"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

CATALYSIS LAB - Advanced Production-Ready Implementation
Free gift to the scientific community from QuLabInfinite.

This module provides comprehensive catalysis simulation capabilities including:
- Turnover frequency (TOF) and turnover number (TON) calculations
- Sabatier principle and volcano plots
- Microkinetic modeling with reaction networks
- Catalyst deactivation and regeneration
- Langmuir-Hinshelwood kinetics
- Eley-Rideal mechanisms
- Temperature-programmed desorption (TPD)
- Surface coverage dynamics
- Diffusion limitations (Thiele modulus)
- Catalyst design optimization
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable
from scipy.constants import k as kb, R, Avogadro
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize, differential_evolution, fsolve
import warnings

# Physical constants
BOLTZMANN_CONSTANT = kb  # J/K
GAS_CONSTANT = R  # J/(mol·K)
AVOGADRO_NUMBER = Avogadro

@dataclass
class Catalyst:
    """Represents a heterogeneous catalyst with surface properties."""
    name: str
    metal: str
    support: str
    surface_area: float  # m²/g
    active_sites: float  # mol sites/g catalyst
    pore_diameter: float  # nm
    particle_size: float  # nm
    binding_energies: Dict[str, float] = field(default_factory=dict)  # eV
    activation_energies: Dict[str, float] = field(default_factory=dict)  # kJ/mol

@dataclass
class Adsorbate:
    """Represents a molecule adsorbed on catalyst surface."""
    name: str
    molecular_weight: float  # g/mol
    sigma: float  # Molecular cross-section (Ų)
    desorption_energy: float  # kJ/mol
    sticking_coefficient: float = 0.1
    diffusion_coefficient: float = 1e-5  # m²/s

@dataclass
class Reaction:
    """Represents a surface reaction."""
    name: str
    reactants: List[str]
    products: List[str]
    stoichiometry: Dict[str, int]  # Negative for reactants, positive for products
    activation_energy: float  # kJ/mol
    pre_exponential: float  # s⁻¹
    heat_of_reaction: float  # kJ/mol
    is_reversible: bool = True

@dataclass
class ReactorConditions:
    """Operating conditions for catalytic reactor."""
    temperature: float  # K
    pressure: float  # Pa
    flow_rate: float  # m³/s
    catalyst_mass: float  # kg
    reactor_volume: float  # m³

class CatalysisLab:
    """Advanced catalysis laboratory for comprehensive reaction simulations."""

    def __init__(self, temperature: float = 523.15, pressure: float = 101325):
        """Initialize catalysis lab."""
        self.temperature = temperature
        self.pressure = pressure
        self.R = GAS_CONSTANT

    # ==================== TURNOVER FREQUENCY ====================

    def turnover_frequency(self, rate: float, active_sites: float) -> float:
        """
        Calculate turnover frequency (TOF).

        TOF = reaction rate / number of active sites
        Units: s⁻¹
        """
        return rate / active_sites

    def turnover_number(self, moles_product: float, active_sites: float) -> float:
        """
        Calculate turnover number (TON).

        TON = moles of product / moles of active sites
        """
        return moles_product / active_sites

    def site_time_yield(self, production_rate: float, active_sites: float,
                        catalyst_mass: float) -> float:
        """
        Calculate site-time yield (STY).

        STY = production rate / (active sites × catalyst mass)
        Units: mol/(mol_sites·kg·s)
        """
        return production_rate / (active_sites * catalyst_mass)

    # ==================== SABATIER PRINCIPLE ====================

    def sabatier_volcano_plot(self, binding_energies: np.ndarray,
                             reaction_energy: float = -50.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Sabatier volcano plot data.

        The optimal catalyst has intermediate binding energy.
        Too weak: low coverage
        Too strong: poisoning
        """
        # Brønsted-Evans-Polanyi relation
        alpha = 0.5  # Transfer coefficient
        activation_energies = alpha * binding_energies + reaction_energy / 2

        # Calculate rates using transition state theory
        rates = np.zeros_like(binding_energies)

        for i, (BE, Ea) in enumerate(zip(binding_energies, activation_energies)):
            # Surface coverage (Langmuir isotherm)
            K_ads = np.exp(-BE * 1000 / (self.R * self.temperature))
            theta = K_ads / (1 + K_ads)

            # Rate constant
            k = 1e13 * np.exp(-Ea * 1000 / (self.R * self.temperature))

            # Overall rate
            rates[i] = k * theta * (1 - theta)  # Need free sites for products

        return binding_energies, np.log10(rates)

    def optimal_binding_energy(self, reaction_energy: float,
                              temperature: Optional[float] = None) -> float:
        """
        Calculate optimal binding energy using Sabatier principle.

        Optimal when d(rate)/d(BE) = 0
        """
        T = temperature if temperature else self.temperature

        # For simple model: optimal BE ≈ -ΔH_rxn/2
        optimal_BE = -reaction_energy / 2

        # Temperature correction
        entropy_factor = T * 0.001  # Entropy contribution (simplified)
        optimal_BE += entropy_factor

        return optimal_BE

    # ==================== MICROKINETIC MODELING ====================

    def microkinetic_model(self, reactions: List[Reaction],
                          initial_coverages: Dict[str, float],
                          conditions: ReactorConditions,
                          time_span: Tuple[float, float]) -> Dict:
        """
        Solve microkinetic model for reaction network.

        Uses mean-field approximation for surface coverages.
        """
        species = set()
        for rxn in reactions:
            species.update(rxn.reactants)
            species.update(rxn.products)
        species = sorted(list(species))

        # Map species to indices
        species_idx = {sp: i for i, sp in enumerate(species)}

        # Initial conditions
        theta0 = np.zeros(len(species))
        for sp, cov in initial_coverages.items():
            if sp in species_idx:
                theta0[species_idx[sp]] = cov

        def coverage_odes(t, theta):
            """ODEs for surface coverage evolution."""
            dtheta_dt = np.zeros_like(theta)

            # Ensure coverages stay in [0, 1]
            theta = np.clip(theta, 0, 1)

            # Free sites
            theta_free = max(0, 1 - np.sum(theta))

            for rxn in reactions:
                # Forward rate
                k_f = rxn.pre_exponential * np.exp(-rxn.activation_energy * 1000 /
                                                  (self.R * conditions.temperature))

                rate_f = k_f
                for reactant in rxn.reactants:
                    if reactant == '*':  # Free site
                        rate_f *= theta_free
                    else:
                        idx = species_idx.get(reactant)
                        if idx is not None:
                            rate_f *= theta[idx]

                # Reverse rate (if reversible)
                rate_r = 0
                if rxn.is_reversible:
                    K_eq = np.exp(-rxn.heat_of_reaction * 1000 /
                                 (self.R * conditions.temperature))
                    k_r = k_f / K_eq

                    rate_r = k_r
                    for product in rxn.products:
                        if product == '*':
                            rate_r *= theta_free
                        else:
                            idx = species_idx.get(product)
                            if idx is not None:
                                rate_r *= theta[idx]

                # Net rate
                net_rate = rate_f - rate_r

                # Update coverage changes
                for sp, stoich in rxn.stoichiometry.items():
                    if sp in species_idx:
                        dtheta_dt[species_idx[sp]] += stoich * net_rate

            return dtheta_dt

        # Solve ODEs
        solution = solve_ivp(coverage_odes, time_span, theta0,
                           method='BDF', dense_output=True)

        # Calculate production rates
        production_rates = {}
        final_coverages = solution.y[:, -1]

        for sp in species:
            idx = species_idx[sp]
            # Rate of change at steady state
            rate = coverage_odes(time_span[1], final_coverages)[idx]
            production_rates[sp] = rate * conditions.catalyst_mass * 1000  # mol/s

        return {
            'time': solution.t,
            'coverages': solution.y,
            'species': species,
            'production_rates': production_rates,
            'final_coverages': {sp: final_coverages[species_idx[sp]] for sp in species}
        }

    # ==================== LANGMUIR-HINSHELWOOD KINETICS ====================

    def langmuir_hinshelwood_rate(self, partial_pressures: Dict[str, float],
                                 adsorption_constants: Dict[str, float],
                                 rate_constant: float,
                                 mechanism: str = 'bimolecular') -> float:
        """
        Calculate reaction rate using Langmuir-Hinshelwood mechanism.

        Both reactants adsorb on surface before reaction.
        """
        if mechanism == 'unimolecular':
            # A* → Products
            species = list(partial_pressures.keys())[0]
            K_A = adsorption_constants[species]
            P_A = partial_pressures[species]

            denominator = 1 + K_A * P_A
            rate = rate_constant * K_A * P_A / denominator

        elif mechanism == 'bimolecular':
            # A* + B* → Products
            species = list(partial_pressures.keys())[:2]
            K_A = adsorption_constants[species[0]]
            K_B = adsorption_constants[species[1]]
            P_A = partial_pressures[species[0]]
            P_B = partial_pressures[species[1]]

            denominator = (1 + K_A * P_A + K_B * P_B)**2
            rate = rate_constant * K_A * K_B * P_A * P_B / denominator

        else:  # Competitive adsorption
            # A* + B* → Products (with competitive adsorption)
            denominator = 1
            for sp, P in partial_pressures.items():
                if sp in adsorption_constants:
                    denominator += adsorption_constants[sp] * P

            # Product of adsorbed species
            numerator = rate_constant
            for sp, P in partial_pressures.items():
                if sp in adsorption_constants:
                    numerator *= adsorption_constants[sp] * P

            rate = numerator / denominator**2

        return rate

    # ==================== ELEY-RIDEAL MECHANISM ====================

    def eley_rideal_rate(self, partial_pressure_gas: float,
                        coverage_adsorbed: float,
                        rate_constant: float) -> float:
        """
        Calculate rate for Eley-Rideal mechanism.

        Gas phase molecule reacts directly with adsorbed species.
        A(g) + B* → Products
        """
        rate = rate_constant * partial_pressure_gas * coverage_adsorbed
        return rate

    # ==================== CATALYST DEACTIVATION ====================

    def deactivation_model(self, time: np.ndarray, deactivation_order: int = 1,
                          deactivation_constant: float = 0.001,
                          regeneration_constant: float = 0.0) -> np.ndarray:
        """
        Model catalyst deactivation over time.

        da/dt = -k_d * a^n + k_r * (1-a)
        where a is catalyst activity (0 to 1)
        """
        def activity_ode(a, t):
            dadt = -deactivation_constant * a**deactivation_order
            if regeneration_constant > 0:
                dadt += regeneration_constant * (1 - a)
            return dadt

        # Initial activity = 1
        activity = odeint(activity_ode, 1.0, time)
        return activity.flatten()

    def fouling_resistance(self, time: float, fouling_rate: float = 0.01,
                          initial_resistance: float = 0.0) -> float:
        """
        Calculate fouling resistance over time.

        R_f = R_f0 + k_f * t
        """
        return initial_resistance + fouling_rate * time

    def sintering_model(self, temperature: float, time: float,
                       initial_size: float = 5.0) -> float:
        """
        Model catalyst particle sintering (Ostwald ripening).

        d^n - d0^n = k*t where n ≈ 3 for Ostwald ripening
        """
        n = 3  # Growth exponent
        E_sintering = 150000  # J/mol

        # Temperature-dependent rate constant
        k_sinter = 1e-3 * np.exp(-E_sintering / (self.R * temperature))

        # Final particle size
        d_final = (initial_size**n + k_sinter * time)**(1/n)

        return d_final

    # ==================== TEMPERATURE-PROGRAMMED DESORPTION ====================

    def tpd_simulation(self, adsorbate: Adsorbate, heating_rate: float = 10.0,
                      temp_range: Tuple[float, float] = (300, 800),
                      initial_coverage: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Temperature-Programmed Desorption (TPD).

        Redhead equation for first-order desorption.
        """
        temperatures = np.linspace(temp_range[0], temp_range[1], 500)
        beta = heating_rate / 60  # K/s

        # Frequency factor (typical value)
        nu = 1e13  # s⁻¹

        # Calculate desorption rate at each temperature
        rates = np.zeros_like(temperatures)

        for i, T in enumerate(temperatures):
            # Polanyi-Wigner equation
            k_des = nu * np.exp(-adsorbate.desorption_energy * 1000 / (self.R * T))

            # Coverage evolution (simplified - assumes fast desorption)
            if i == 0:
                theta = initial_coverage
            else:
                dT = temperatures[i] - temperatures[i-1]
                dt = dT / beta
                theta = theta * np.exp(-k_des * dt)

            # Desorption rate
            rates[i] = k_des * theta

        # Find peak temperature (Redhead equation)
        T_peak_idx = np.argmax(rates)
        T_peak = temperatures[T_peak_idx]

        return temperatures, rates

    def tpd_peak_temperature(self, desorption_energy: float,
                            heating_rate: float = 10.0) -> float:
        """
        Calculate TPD peak temperature using Redhead equation.

        For first-order desorption: E_d/(R*T_p²) = (ν/β)*exp(-E_d/(R*T_p))
        """
        beta = heating_rate / 60  # K/s
        nu = 1e13  # s⁻¹

        # Solve Redhead equation iteratively
        def redhead_eq(T):
            return desorption_energy * 1000 / (self.R * T**2) - \
                   (nu / beta) * np.exp(-desorption_energy * 1000 / (self.R * T))

        # Initial guess
        T_guess = desorption_energy * 1000 / (self.R * 30)

        T_peak = fsolve(redhead_eq, T_guess)[0]

        return T_peak

    # ==================== DIFFUSION LIMITATIONS ====================

    def thiele_modulus(self, rate_constant: float, catalyst_particle_size: float,
                       effective_diffusivity: float, concentration: float,
                       reaction_order: int = 1) -> float:
        """
        Calculate Thiele modulus for internal diffusion limitations.

        φ = L√(k*C^(n-1)/D_eff) for nth order reaction
        """
        L = catalyst_particle_size / 2  # Characteristic length (radius for sphere)

        if reaction_order == 1:
            phi = L * np.sqrt(rate_constant / effective_diffusivity)
        else:
            phi = L * np.sqrt(rate_constant * concentration**(reaction_order-1) /
                            effective_diffusivity)

        return phi

    def effectiveness_factor(self, thiele_modulus: float, geometry: str = 'sphere') -> float:
        """
        Calculate effectiveness factor for catalyst pellet.

        η = (actual rate) / (rate without diffusion limitations)
        """
        phi = thiele_modulus

        if geometry == 'slab':
            # Infinite slab
            if phi < 0.1:
                eta = 1.0
            else:
                eta = np.tanh(phi) / phi

        elif geometry == 'cylinder':
            # Infinite cylinder
            from scipy.special import i0, i1
            if phi < 0.1:
                eta = 1.0
            else:
                eta = 2 * i1(phi) / (phi * i0(phi))

        else:  # sphere
            # Sphere
            if phi < 0.1:
                eta = 1.0
            else:
                eta = 3 * (phi * (1/np.tanh(phi)) - 1) / phi**2

        return eta

    def weisz_prater_criterion(self, observed_rate: float, concentration_surface: float,
                              particle_radius: float, effective_diffusivity: float) -> float:
        """
        Calculate Weisz-Prater criterion for diffusion limitations.

        If Cwp < 1: no internal diffusion limitations
        If Cwp > 1: significant internal diffusion limitations
        """
        Cwp = observed_rate * particle_radius**2 / (effective_diffusivity * concentration_surface)
        return Cwp

    # ==================== CATALYST DESIGN OPTIMIZATION ====================

    def optimize_catalyst_design(self, target_conversion: float,
                                feed_conditions: Dict, constraints: Dict) -> Dict:
        """
        Optimize catalyst design for target conversion.

        Variables: metal loading, particle size, support properties
        """
        def objective(x):
            """Minimize cost while achieving target conversion."""
            metal_loading, particle_size, pore_size = x

            # Create catalyst
            catalyst = Catalyst(
                name="Optimized",
                metal="Pt",
                support="Al2O3",
                surface_area=200 / particle_size,  # Inversely proportional
                active_sites=metal_loading * 0.01,  # mol/g
                pore_diameter=pore_size,
                particle_size=particle_size,
                binding_energies={'CO': -1.5, 'O2': -0.8},
                activation_energies={'oxidation': 80}
            )

            # Calculate conversion (simplified model)
            k = 1e5 * np.exp(-80000 / (self.R * self.temperature))
            tau = constraints.get('residence_time', 1.0)
            conversion = 1 - np.exp(-k * catalyst.active_sites * tau)

            # Cost function
            metal_cost = metal_loading * 50  # $/g
            support_cost = (100 / particle_size) * 0.1  # $/g

            # Penalty for not meeting target
            conversion_penalty = 1000 * max(0, target_conversion - conversion)**2

            return metal_cost + support_cost + conversion_penalty

        # Constraints
        bounds = [
            (0.1, 10),   # Metal loading (wt%)
            (1, 100),    # Particle size (nm)
            (2, 50)      # Pore size (nm)
        ]

        # Optimize
        result = differential_evolution(objective, bounds, maxiter=100)

        optimal_design = {
            'metal_loading': result.x[0],
            'particle_size': result.x[1],
            'pore_size': result.x[2],
            'cost': result.fun
        }

        return optimal_design

    # ==================== REACTION NETWORK ANALYSIS ====================

    def reaction_network_analysis(self, reactions: List[Reaction],
                                 species: List[str]) -> Dict:
        """
        Analyze reaction network connectivity and pathways.
        """
        n_reactions = len(reactions)
        n_species = len(species)

        # Stoichiometry matrix
        S = np.zeros((n_species, n_reactions))

        species_idx = {sp: i for i, sp in enumerate(species)}

        for j, rxn in enumerate(reactions):
            for sp, stoich in rxn.stoichiometry.items():
                if sp in species_idx:
                    S[species_idx[sp], j] = stoich

        # Calculate rank (number of independent reactions)
        rank = np.linalg.matrix_rank(S)

        # Find conservation relations
        from scipy.linalg import null_space
        conservation_matrix = null_space(S.T)

        # Identify rate-determining steps (highest activation energy)
        rds_idx = np.argmax([rxn.activation_energy for rxn in reactions])
        rds = reactions[rds_idx]

        return {
            'stoichiometry_matrix': S,
            'rank': rank,
            'degrees_of_freedom': n_reactions - rank,
            'conservation_relations': conservation_matrix,
            'rate_determining_step': rds.name
        }

    # ==================== DEMONSTRATION ====================

    def run_comprehensive_demo(self):
        """Run comprehensive demonstration of catalysis capabilities."""
        print("=" * 60)
        print("CATALYSIS LAB - Comprehensive Demonstration")
        print("=" * 60)

        # 1. Catalyst and Reaction Setup
        print("\n1. CATALYST PROPERTIES")
        print("-" * 40)

        catalyst = Catalyst(
            name="Pt/Al2O3",
            metal="Pt",
            support="Al2O3",
            surface_area=200.0,  # m²/g
            active_sites=2e-3,  # mol/g
            pore_diameter=10.0,  # nm
            particle_size=5.0,  # nm
            binding_energies={'CO': -1.8, 'O2': -0.9, 'CO2': -0.3},
            activation_energies={'CO_oxidation': 70}  # kJ/mol
        )

        print(f"Catalyst: {catalyst.name}")
        print(f"Surface area: {catalyst.surface_area} m²/g")
        print(f"Active sites: {catalyst.active_sites*1e3} mmol/g")

        # 2. Turnover Frequency
        print("\n2. TURNOVER FREQUENCY ANALYSIS")
        print("-" * 40)

        reaction_rate = 1e-6  # mol/s
        tof = self.turnover_frequency(reaction_rate, catalyst.active_sites)
        print(f"Reaction rate: {reaction_rate*1e6} µmol/s")
        print(f"TOF: {tof:.2f} s⁻¹")

        ton = self.turnover_number(0.01, catalyst.active_sites)
        print(f"TON after producing 0.01 mol: {ton:.0f}")

        # 3. Sabatier Principle
        print("\n3. SABATIER VOLCANO PLOT")
        print("-" * 40)

        binding_energies = np.linspace(-3, 0, 20)
        BE, log_rates = self.sabatier_volcano_plot(binding_energies, -50)

        optimal_BE = self.optimal_binding_energy(-50)
        print(f"Optimal binding energy: {optimal_BE:.2f} eV")

        max_rate_idx = np.argmax(log_rates)
        print(f"Maximum rate at BE = {BE[max_rate_idx]:.2f} eV")

        # 4. Microkinetic Modeling
        print("\n4. MICROKINETIC MODELING")
        print("-" * 40)

        reactions = [
            Reaction(
                name="CO adsorption",
                reactants=["CO", "*"],
                products=["CO*"],
                stoichiometry={"CO": -1, "*": -1, "CO*": 1},
                activation_energy=0,
                pre_exponential=1e13,
                heat_of_reaction=-180,
                is_reversible=True
            ),
            Reaction(
                name="O2 dissociative adsorption",
                reactants=["O2", "*", "*"],
                products=["O*", "O*"],
                stoichiometry={"O2": -1, "*": -2, "O*": 2},
                activation_energy=0,
                pre_exponential=1e13,
                heat_of_reaction=-220,
                is_reversible=True
            ),
            Reaction(
                name="CO oxidation",
                reactants=["CO*", "O*"],
                products=["CO2", "*", "*"],
                stoichiometry={"CO*": -1, "O*": -1, "CO2": 1, "*": 2},
                activation_energy=70,
                pre_exponential=1e13,
                heat_of_reaction=-280,
                is_reversible=False
            )
        ]

        conditions = ReactorConditions(
            temperature=523.15,
            pressure=101325,
            flow_rate=1e-3,
            catalyst_mass=0.001,
            reactor_volume=1e-3
        )

        initial_coverages = {"*": 1.0}  # All sites initially free

        result = self.microkinetic_model(reactions, initial_coverages,
                                        conditions, (0, 10))

        print(f"Final surface coverages:")
        for sp, cov in result['final_coverages'].items():
            if cov > 0.01:
                print(f"  {sp}: {cov:.3f}")

        # 5. Langmuir-Hinshelwood Kinetics
        print("\n5. LANGMUIR-HINSHELWOOD KINETICS")
        print("-" * 40)

        partial_pressures = {'CO': 1000, 'O2': 2000}  # Pa
        adsorption_constants = {'CO': 0.01, 'O2': 0.005}  # Pa⁻¹
        k_rate = 1e-3

        rate_LH = self.langmuir_hinshelwood_rate(partial_pressures,
                                                adsorption_constants,
                                                k_rate, 'bimolecular')
        print(f"L-H rate: {rate_LH*1e6:.2f} µmol/s")

        # 6. Catalyst Deactivation
        print("\n6. CATALYST DEACTIVATION")
        print("-" * 40)

        time = np.linspace(0, 1000, 100)  # hours
        activity = self.deactivation_model(time, deactivation_order=1,
                                          deactivation_constant=0.001)

        print(f"Initial activity: {activity[0]:.3f}")
        print(f"Activity after 100 h: {activity[10]:.3f}")
        print(f"Activity after 1000 h: {activity[-1]:.3f}")

        # 7. TPD Simulation
        print("\n7. TEMPERATURE-PROGRAMMED DESORPTION")
        print("-" * 40)

        adsorbate = Adsorbate(
            name="CO",
            molecular_weight=28,
            sigma=15,
            desorption_energy=180,  # kJ/mol
            sticking_coefficient=0.8
        )

        T_range, rates = self.tpd_simulation(adsorbate, heating_rate=10)
        T_peak_idx = np.argmax(rates)
        T_peak = T_range[T_peak_idx]

        print(f"TPD peak temperature: {T_peak:.1f} K")
        print(f"Desorption energy: {adsorbate.desorption_energy} kJ/mol")

        # 8. Diffusion Limitations
        print("\n8. DIFFUSION LIMITATIONS")
        print("-" * 40)

        k_app = 0.1  # s⁻¹
        D_eff = 1e-6  # m²/s
        particle_size = 1e-3  # m

        phi = self.thiele_modulus(k_app, particle_size, D_eff, 0.01)
        eta = self.effectiveness_factor(phi, 'sphere')

        print(f"Thiele modulus: {phi:.3f}")
        print(f"Effectiveness factor: {eta:.3f}")

        if phi < 0.3:
            print("Reaction limited (no diffusion limitations)")
        elif phi > 3:
            print("Diffusion limited")
        else:
            print("Mixed control")

        print("\n" + "=" * 60)
        print("Demonstration complete!")
        print("=" * 60)

def run_demo():
    """Run the comprehensive catalysis demonstration."""
    lab = CatalysisLab()
    lab.run_comprehensive_demo()

if __name__ == '__main__':
    run_demo()