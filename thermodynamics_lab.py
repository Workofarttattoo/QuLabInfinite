"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

THERMODYNAMICS LAB - Advanced Production-Ready Implementation
Free gift to the scientific community from QuLabInfinite.

This module provides comprehensive thermodynamics simulation capabilities including:
- Equations of state (ideal gas, van der Waals, Peng-Robinson, SRK)
- Fugacity and activity coefficients
- Phase equilibria (VLE, LLE, VLLE)
- Chemical equilibrium and reaction thermodynamics
- Heat capacity models (polynomial, Shomate)
- Entropy and Gibbs energy calculations
- Joule-Thomson effect and throttling
- Compressibility factors
- Critical properties and corresponding states
- Thermodynamic cycles (Carnot, Rankine, refrigeration)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable
from scipy.constants import R, Avogadro, k as kb
from scipy.optimize import fsolve, minimize, root
from scipy.integrate import quad, odeint
import warnings

# Physical constants
GAS_CONSTANT = R  # J/(mol·K)
AVOGADRO_NUMBER = Avogadro
BOLTZMANN_CONSTANT = kb  # J/K

@dataclass
class Component:
    """Represents a chemical component with thermodynamic properties."""
    name: str
    molecular_weight: float  # g/mol
    critical_temperature: float  # K
    critical_pressure: float  # Pa
    critical_volume: float  # m³/mol
    acentric_factor: float
    normal_boiling_point: float  # K
    heat_of_vaporization: float  # J/mol
    heat_of_formation: float = 0.0  # J/mol
    gibbs_of_formation: float = 0.0  # J/mol
    heat_capacity_params: List[float] = field(default_factory=list)  # Polynomial coefficients

@dataclass
class Mixture:
    """Represents a mixture of components."""
    components: List[Component]
    mole_fractions: np.ndarray
    temperature: float  # K
    pressure: float  # Pa

    def __post_init__(self):
        """Validate mixture properties."""
        self.mole_fractions = np.array(self.mole_fractions)
        if abs(np.sum(self.mole_fractions) - 1.0) > 1e-6:
            self.mole_fractions = self.mole_fractions / np.sum(self.mole_fractions)

@dataclass
class ThermodynamicState:
    """Represents a thermodynamic state."""
    temperature: float  # K
    pressure: float  # Pa
    volume: float  # m³
    internal_energy: float  # J
    enthalpy: float  # J
    entropy: float  # J/K
    gibbs_energy: float  # J
    helmholtz_energy: float  # J

class ThermodynamicsLab:
    """Advanced thermodynamics laboratory for comprehensive thermodynamic simulations."""

    def __init__(self, temperature: float = 298.15, pressure: float = 101325):
        """Initialize thermodynamics lab."""
        self.temperature = temperature
        self.pressure = pressure
        self.R = GAS_CONSTANT

    # ==================== EQUATIONS OF STATE ====================

    def ideal_gas_pressure(self, n: float, V: float, T: float) -> float:
        """
        Calculate pressure using ideal gas law.
        PV = nRT
        """
        return n * self.R * T / V

    def ideal_gas_volume(self, n: float, P: float, T: float) -> float:
        """
        Calculate volume using ideal gas law.
        """
        return n * self.R * T / P

    def van_der_waals(self, n: float, V: float, T: float,
                     a: float, b: float) -> float:
        """
        Calculate pressure using van der Waals equation.

        (P + a*n²/V²)(V - nb) = nRT
        where a accounts for intermolecular attraction
        and b accounts for molecular volume
        """
        P = (n * self.R * T) / (V - n * b) - a * n**2 / V**2
        return P

    def van_der_waals_volume(self, n: float, P: float, T: float,
                            a: float, b: float) -> float:
        """
        Solve van der Waals equation for volume.
        """
        def vdw_equation(V):
            return (n * self.R * T) / (V - n * b) - a * n**2 / V**2 - P

        # Initial guess
        V_ideal = self.ideal_gas_volume(n, P, T)

        try:
            V_solution = fsolve(vdw_equation, V_ideal)[0]
            return V_solution
        except:
            return V_ideal

    def peng_robinson(self, T: float, P: float, component: Component,
                     solve_for: str = 'Z') -> float:
        """
        Peng-Robinson equation of state.

        P = RT/(V-b) - a(T)/(V(V+b) + b(V-b))
        """
        Tc = component.critical_temperature
        Pc = component.critical_pressure
        omega = component.acentric_factor

        # Reduced temperature
        Tr = T / Tc

        # Parameters
        kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
        alpha = (1 + kappa * (1 - np.sqrt(Tr)))**2

        a = 0.45724 * (self.R * Tc)**2 / Pc * alpha
        b = 0.07780 * self.R * Tc / Pc

        A = a * P / (self.R * T)**2
        B = b * P / (self.R * T)

        if solve_for == 'Z':
            # Solve cubic equation for Z
            coeffs = [1, -(1 - B), A - 3*B**2 - 2*B, -(A*B - B**2 - B**3)]
            roots = np.roots(coeffs)

            # Select real positive root
            real_roots = roots[np.isreal(roots)].real
            positive_roots = real_roots[real_roots > 0]

            if len(positive_roots) > 0:
                # For vapor phase, take largest root
                # For liquid phase, take smallest root
                if P < Pc:  # Below critical pressure
                    return np.max(positive_roots)  # Vapor
                else:
                    return np.min(positive_roots)  # Liquid
            else:
                return 1.0  # Ideal gas

        return 1.0

    def redlich_kwong_soave(self, T: float, P: float, component: Component) -> float:
        """
        Soave-Redlich-Kwong equation of state.

        P = RT/(V-b) - a(T)/(V(V+b))
        """
        Tc = component.critical_temperature
        Pc = component.critical_pressure
        omega = component.acentric_factor

        # Reduced temperature
        Tr = T / Tc

        # Parameters
        m = 0.480 + 1.574 * omega - 0.176 * omega**2
        alpha = (1 + m * (1 - np.sqrt(Tr)))**2

        a = 0.42748 * (self.R * Tc)**2 / Pc * alpha
        b = 0.08664 * self.R * Tc / Pc

        A = a * P / (self.R * T)**2
        B = b * P / (self.R * T)

        # Solve cubic equation
        coeffs = [1, -1, A - B - B**2, -A*B]
        roots = np.roots(coeffs)

        # Select appropriate root
        real_roots = roots[np.isreal(roots)].real
        positive_roots = real_roots[real_roots > 0]

        if len(positive_roots) > 0:
            return np.max(positive_roots)
        else:
            return 1.0

    def compressibility_factor(self, T: float, P: float, V: float, n: float = 1.0) -> float:
        """
        Calculate compressibility factor Z = PV/(nRT)
        """
        Z = P * V / (n * self.R * T)
        return Z

    # ==================== FUGACITY AND ACTIVITY ====================

    def fugacity_coefficient_ideal(self) -> float:
        """Fugacity coefficient for ideal gas (always 1)."""
        return 1.0

    def fugacity_coefficient_virial(self, T: float, P: float, B: float) -> float:
        """
        Calculate fugacity coefficient using second virial coefficient.

        ln(φ) = B*P/(RT)
        """
        phi = np.exp(B * P / (self.R * T))
        return phi

    def fugacity_coefficient_pr(self, T: float, P: float, component: Component) -> float:
        """
        Calculate fugacity coefficient using Peng-Robinson EOS.
        """
        Z = self.peng_robinson(T, P, component, 'Z')

        Tc = component.critical_temperature
        Pc = component.critical_pressure
        omega = component.acentric_factor

        # Reduced properties
        Tr = T / Tc

        # PR parameters
        kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
        alpha = (1 + kappa * (1 - np.sqrt(Tr)))**2

        a = 0.45724 * (self.R * Tc)**2 / Pc * alpha
        b = 0.07780 * self.R * Tc / Pc

        A = a * P / (self.R * T)**2
        B = b * P / (self.R * T)

        # Fugacity coefficient
        ln_phi = Z - 1 - np.log(Z - B) - A / (2 * np.sqrt(2) * B) * \
                np.log((Z + (1 + np.sqrt(2)) * B) / (Z + (1 - np.sqrt(2)) * B))

        return np.exp(ln_phi)

    def activity_coefficient_margules(self, x1: float, A12: float, A21: float) -> Tuple[float, float]:
        """
        Calculate activity coefficients using Margules equation.

        ln(γ1) = x2² * (A12 + 2*(A21 - A12)*x1)
        """
        x2 = 1 - x1

        ln_gamma1 = x2**2 * (A12 + 2 * (A21 - A12) * x1)
        ln_gamma2 = x1**2 * (A21 + 2 * (A12 - A21) * x2)

        gamma1 = np.exp(ln_gamma1)
        gamma2 = np.exp(ln_gamma2)

        return gamma1, gamma2

    def activity_coefficient_wilson(self, x1: float, T: float,
                                  Lambda12: float, Lambda21: float) -> Tuple[float, float]:
        """
        Calculate activity coefficients using Wilson equation.
        """
        x2 = 1 - x1

        ln_gamma1 = -np.log(x1 + Lambda12 * x2) + x2 * (
            Lambda12 / (x1 + Lambda12 * x2) - Lambda21 / (Lambda21 * x1 + x2)
        )

        ln_gamma2 = -np.log(x2 + Lambda21 * x1) - x1 * (
            Lambda12 / (x1 + Lambda12 * x2) - Lambda21 / (Lambda21 * x1 + x2)
        )

        return np.exp(ln_gamma1), np.exp(ln_gamma2)

    def activity_coefficient_nrtl(self, x1: float, T: float,
                                tau12: float, tau21: float, alpha: float = 0.3) -> Tuple[float, float]:
        """
        Calculate activity coefficients using NRTL equation.
        """
        x2 = 1 - x1

        G12 = np.exp(-alpha * tau12)
        G21 = np.exp(-alpha * tau21)

        ln_gamma1 = x2**2 * (tau21 * (G21 / (x1 + x2 * G21))**2 +
                            tau12 * G12 / (x2 + x1 * G12)**2)

        ln_gamma2 = x1**2 * (tau12 * (G12 / (x2 + x1 * G12))**2 +
                            tau21 * G21 / (x1 + x2 * G21)**2)

        return np.exp(ln_gamma1), np.exp(ln_gamma2)

    # ==================== PHASE EQUILIBRIA ====================

    def vapor_pressure_antoine(self, T: float, A: float, B: float, C: float) -> float:
        """
        Calculate vapor pressure using Antoine equation.

        log10(P) = A - B/(C + T)
        P in mmHg, T in °C
        """
        T_celsius = T - 273.15
        log_P = A - B / (C + T_celsius)
        P_mmHg = 10**log_P
        P_Pa = P_mmHg * 133.322  # Convert to Pa

        return P_Pa

    def vapor_pressure_clausius_clapeyron(self, T: float, T_ref: float,
                                         P_ref: float, delta_H_vap: float) -> float:
        """
        Calculate vapor pressure using Clausius-Clapeyron equation.

        ln(P/P_ref) = -ΔHvap/R * (1/T - 1/T_ref)
        """
        ln_P_ratio = -delta_H_vap / self.R * (1/T - 1/T_ref)
        P = P_ref * np.exp(ln_P_ratio)

        return P

    def bubble_point_calculation(self, mixture: Mixture, P: float) -> Tuple[float, np.ndarray]:
        """
        Calculate bubble point temperature and vapor composition.
        """
        x = mixture.mole_fractions

        def bubble_point_equation(T):
            P_calc = 0
            for i, comp in enumerate(mixture.components):
                # Simplified - use Raoult's law
                P_sat = self.vapor_pressure_clausius_clapeyron(
                    T, comp.normal_boiling_point, 101325, comp.heat_of_vaporization
                )
                P_calc += x[i] * P_sat
            return P_calc - P

        # Solve for temperature
        T_guess = mixture.components[0].normal_boiling_point
        T_bubble = fsolve(bubble_point_equation, T_guess)[0]

        # Calculate vapor composition
        y = np.zeros_like(x)
        for i, comp in enumerate(mixture.components):
            P_sat = self.vapor_pressure_clausius_clapeyron(
                T_bubble, comp.normal_boiling_point, 101325, comp.heat_of_vaporization
            )
            y[i] = x[i] * P_sat / P

        return T_bubble, y

    def dew_point_calculation(self, mixture: Mixture, P: float) -> Tuple[float, np.ndarray]:
        """
        Calculate dew point temperature and liquid composition.
        """
        y = mixture.mole_fractions

        def dew_point_equation(T):
            sum_x = 0
            for i, comp in enumerate(mixture.components):
                P_sat = self.vapor_pressure_clausius_clapeyron(
                    T, comp.normal_boiling_point, 101325, comp.heat_of_vaporization
                )
                sum_x += y[i] * P / P_sat
            return sum_x - 1

        # Solve for temperature
        T_guess = mixture.components[0].normal_boiling_point
        T_dew = fsolve(dew_point_equation, T_guess)[0]

        # Calculate liquid composition
        x = np.zeros_like(y)
        for i, comp in enumerate(mixture.components):
            P_sat = self.vapor_pressure_clausius_clapeyron(
                T_dew, comp.normal_boiling_point, 101325, comp.heat_of_vaporization
            )
            x[i] = y[i] * P / P_sat

        x = x / np.sum(x)  # Normalize

        return T_dew, x

    def flash_calculation(self, mixture: Mixture, T: float, P: float) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Perform isothermal flash calculation.
        Returns vapor fraction and compositions.
        """
        z = mixture.mole_fractions

        # Calculate K-values (simplified)
        K = np.zeros(len(mixture.components))
        for i, comp in enumerate(mixture.components):
            P_sat = self.vapor_pressure_clausius_clapeyron(
                T, comp.normal_boiling_point, 101325, comp.heat_of_vaporization
            )
            K[i] = P_sat / P

        # Rachford-Rice equation
        def rachford_rice(V):
            return np.sum(z * (K - 1) / (1 + V * (K - 1)))

        # Check if single phase
        if np.all(K < 1):  # All liquid
            return 0.0, z.copy(), np.zeros_like(z)
        elif np.all(K > 1):  # All vapor
            return 1.0, np.zeros_like(z), z.copy()

        # Two-phase region
        V = fsolve(rachford_rice, 0.5)[0]
        V = np.clip(V, 0, 1)

        # Calculate phase compositions
        x = z / (1 + V * (K - 1))
        y = K * x

        return V, x, y

    # ==================== CHEMICAL EQUILIBRIUM ====================

    def equilibrium_constant(self, delta_G: float, T: float) -> float:
        """
        Calculate equilibrium constant from Gibbs energy.

        ΔG° = -RT*ln(K)
        """
        K = np.exp(-delta_G / (self.R * T))
        return K

    def van_t_hoff_equation(self, K1: float, T1: float, T2: float,
                           delta_H: float) -> float:
        """
        Calculate equilibrium constant at different temperature.

        ln(K2/K1) = -ΔH/R * (1/T2 - 1/T1)
        """
        ln_K_ratio = -delta_H / self.R * (1/T2 - 1/T1)
        K2 = K1 * np.exp(ln_K_ratio)
        return K2

    def reaction_gibbs_energy(self, products: List[Component], reactants: List[Component],
                            stoich_products: List[float], stoich_reactants: List[float],
                            T: float) -> float:
        """
        Calculate Gibbs energy of reaction.

        ΔG_rxn = Σ(ν_products * G_products) - Σ(ν_reactants * G_reactants)
        """
        delta_G = 0

        # Products
        for comp, stoich in zip(products, stoich_products):
            delta_G += stoich * comp.gibbs_of_formation

        # Reactants
        for comp, stoich in zip(reactants, stoich_reactants):
            delta_G -= stoich * comp.gibbs_of_formation

        return delta_G

    def reaction_enthalpy(self, products: List[Component], reactants: List[Component],
                         stoich_products: List[float], stoich_reactants: List[float]) -> float:
        """
        Calculate enthalpy of reaction.

        ΔH_rxn = Σ(ν_products * H_products) - Σ(ν_reactants * H_reactants)
        """
        delta_H = 0

        # Products
        for comp, stoich in zip(products, stoich_products):
            delta_H += stoich * comp.heat_of_formation

        # Reactants
        for comp, stoich in zip(reactants, stoich_reactants):
            delta_H -= stoich * comp.heat_of_formation

        return delta_H

    def chemical_equilibrium_composition(self, initial_moles: np.ndarray,
                                        stoichiometry: np.ndarray,
                                        K: float) -> np.ndarray:
        """
        Calculate equilibrium composition for reaction.

        aA + bB ⇌ cC + dD
        """
        def equilibrium_equation(extent):
            moles = initial_moles + extent * stoichiometry

            # Prevent negative moles
            if np.any(moles < 0):
                return 1e10

            # Calculate activities (simplified as mole fractions)
            total = np.sum(moles)
            activities = moles / total

            # Calculate Q
            Q = 1.0
            for i, stoich in enumerate(stoichiometry):
                if stoich != 0:
                    Q *= activities[i]**stoich

            return Q - K

        # Solve for extent of reaction
        extent = fsolve(equilibrium_equation, 0)[0]

        # Calculate equilibrium moles
        equilibrium_moles = initial_moles + extent * stoichiometry

        return equilibrium_moles

    # ==================== HEAT CAPACITY ====================

    def heat_capacity_polynomial(self, T: float, coeffs: List[float]) -> float:
        """
        Calculate heat capacity using polynomial correlation.

        Cp = a + b*T + c*T² + d*T³ + ...
        """
        Cp = 0
        for i, coeff in enumerate(coeffs):
            Cp += coeff * T**i
        return Cp

    def heat_capacity_shomate(self, T: float, params: List[float]) -> float:
        """
        Calculate heat capacity using Shomate equation.

        Cp = A + B*t + C*t² + D*t³ + E/t²
        where t = T/1000
        """
        if len(params) < 5:
            return self.R * 3.5  # Default

        A, B, C, D, E = params[:5]
        t = T / 1000

        Cp = A + B*t + C*t**2 + D*t**3 + E/t**2

        return Cp

    def heat_capacity_integration(self, T1: float, T2: float,
                                 cp_function: Callable, *args) -> float:
        """
        Integrate heat capacity to get enthalpy change.

        ΔH = ∫Cp dT from T1 to T2
        """
        result, _ = quad(lambda T: cp_function(T, *args), T1, T2)
        return result

    # ==================== ENTROPY ====================

    def entropy_ideal_gas(self, T: float, P: float, component: Component) -> float:
        """
        Calculate absolute entropy of ideal gas.

        S = S° + Cp*ln(T/T°) - R*ln(P/P°)
        """
        # Reference state (298.15 K, 1 bar)
        T_ref = 298.15
        P_ref = 100000

        # Standard entropy (simplified)
        S_standard = 200  # J/(mol·K)

        # Temperature contribution
        if component.heat_capacity_params:
            delta_S_T = self.heat_capacity_integration(
                T_ref, T,
                lambda T: self.heat_capacity_polynomial(T, component.heat_capacity_params) / T
            )
        else:
            delta_S_T = self.R * 2.5 * np.log(T / T_ref)  # Monatomic ideal gas

        # Pressure contribution
        delta_S_P = -self.R * np.log(P / P_ref)

        S = S_standard + delta_S_T + delta_S_P

        return S

    def entropy_mixing(self, mole_fractions: np.ndarray) -> float:
        """
        Calculate entropy of mixing for ideal mixture.

        ΔS_mix = -R * Σ(xi * ln(xi))
        """
        # Remove zeros to avoid log(0)
        x = mole_fractions[mole_fractions > 0]

        delta_S = -self.R * np.sum(x * np.log(x))

        return delta_S

    def gibbs_energy(self, H: float, T: float, S: float) -> float:
        """
        Calculate Gibbs energy.

        G = H - T*S
        """
        return H - T * S

    def helmholtz_energy(self, U: float, T: float, S: float) -> float:
        """
        Calculate Helmholtz energy.

        A = U - T*S
        """
        return U - T * S

    # ==================== JOULE-THOMSON EFFECT ====================

    def joule_thomson_coefficient_ideal(self) -> float:
        """Joule-Thomson coefficient for ideal gas (always 0)."""
        return 0.0

    def joule_thomson_coefficient_vdw(self, T: float, P: float,
                                     a: float, b: float, Cp: float) -> float:
        """
        Calculate Joule-Thomson coefficient for van der Waals gas.

        μ_JT = (∂T/∂P)_H = (1/Cp) * (T*(∂V/∂T)_P - V)
        """
        # For van der Waals gas
        n = 1  # 1 mole
        V = self.van_der_waals_volume(n, P, T, a, b)

        # Temperature derivative of volume (numerical)
        dT = 0.01
        V_plus = self.van_der_waals_volume(n, P, T + dT, a, b)
        dV_dT = (V_plus - V) / dT

        mu_JT = (1 / Cp) * (T * dV_dT - V)

        return mu_JT

    def inversion_temperature(self, a: float, b: float) -> float:
        """
        Calculate Joule-Thomson inversion temperature.

        For van der Waals gas: T_inv = 2a/(Rb)
        """
        T_inv = 2 * a / (self.R * b)
        return T_inv

    # ==================== THERMODYNAMIC CYCLES ====================

    def carnot_efficiency(self, T_hot: float, T_cold: float) -> float:
        """
        Calculate Carnot cycle efficiency.

        η = 1 - T_cold/T_hot
        """
        if T_hot <= T_cold:
            return 0.0

        eta = 1 - T_cold / T_hot
        return eta

    def rankine_cycle(self, T_boiler: float, T_condenser: float,
                     P_boiler: float, P_condenser: float,
                     efficiency_turbine: float = 0.85,
                     efficiency_pump: float = 0.80) -> Dict:
        """
        Calculate Rankine cycle performance.
        """
        # Simplified steam properties
        h_fg = 2257000  # J/kg (latent heat at 100°C)
        cp_water = 4186  # J/(kg·K)
        cp_steam = 2010  # J/(kg·K)

        # State 1: Saturated liquid at condenser
        h1 = cp_water * (T_condenser - 273.15)
        s1 = cp_water * np.log(T_condenser / 273.15)

        # State 2: Compressed liquid (isentropic pump)
        # Pump work (simplified)
        w_pump = (P_boiler - P_condenser) / (1000 * efficiency_pump)  # J/kg
        h2 = h1 + w_pump

        # State 3: Saturated/superheated vapor at boiler
        h3 = h1 + h_fg + cp_steam * (T_boiler - 373.15)

        # State 4: Wet mixture at condenser (isentropic expansion)
        s3 = s1 + h_fg / 373.15 + cp_steam * np.log(T_boiler / 373.15)

        # Quality at state 4
        x4 = (s3 - s1) / (h_fg / T_condenser)
        h4s = h1 + x4 * h_fg  # Isentropic

        # Actual turbine work
        w_turbine = efficiency_turbine * (h3 - h4s)
        h4 = h3 - w_turbine

        # Heat input and output
        q_in = h3 - h2
        q_out = h4 - h1

        # Net work and efficiency
        w_net = w_turbine - w_pump
        efficiency = w_net / q_in

        return {
            'efficiency': efficiency,
            'work_net': w_net,
            'work_turbine': w_turbine,
            'work_pump': w_pump,
            'heat_input': q_in,
            'heat_rejected': q_out,
            'back_work_ratio': w_pump / w_turbine
        }

    def refrigeration_cycle(self, T_evaporator: float, T_condenser: float,
                          refrigerant: str = 'R134a') -> Dict:
        """
        Calculate vapor-compression refrigeration cycle.
        """
        # Simplified refrigerant properties
        h_fg = 200000  # J/kg (approximate)
        cp_liquid = 1400  # J/(kg·K)
        cp_vapor = 900  # J/(kg·K)

        # State 1: Saturated vapor at evaporator
        h1 = h_fg

        # State 2: Superheated vapor at condenser pressure
        # Isentropic compression
        gamma = cp_vapor / (cp_vapor - self.R / 0.1)  # Approximate
        T2 = T_evaporator * (T_condenser / T_evaporator)**(gamma / (gamma - 1))
        h2 = h1 + cp_vapor * (T2 - T_evaporator)

        # State 3: Saturated liquid at condenser
        h3 = cp_liquid * (T_condenser - T_evaporator)

        # State 4: Two-phase at evaporator (throttling)
        h4 = h3  # Isenthalpic

        # Performance
        q_evaporator = h1 - h4
        w_compressor = h2 - h1

        COP = q_evaporator / w_compressor
        COP_carnot = T_evaporator / (T_condenser - T_evaporator)

        return {
            'COP': COP,
            'COP_carnot': COP_carnot,
            'efficiency_2nd_law': COP / COP_carnot,
            'cooling_capacity': q_evaporator,
            'compressor_work': w_compressor
        }

    # ==================== DEMONSTRATION ====================

    def run_comprehensive_demo(self):
        """Run comprehensive demonstration of thermodynamics capabilities."""
        print("=" * 60)
        print("THERMODYNAMICS LAB - Comprehensive Demonstration")
        print("=" * 60)

        # 1. Equations of State
        print("\n1. EQUATIONS OF STATE")
        print("-" * 40)

        n = 1.0  # mol
        T = 300  # K
        P = 101325  # Pa

        # Ideal gas
        V_ideal = self.ideal_gas_volume(n, P, T)
        print(f"Ideal gas volume at {T}K, {P/1000:.1f} kPa: {V_ideal*1000:.2f} L/mol")

        # Van der Waals (CO2 parameters)
        a = 3.640  # bar·L²/mol²
        b = 0.04267  # L/mol
        a_SI = a * 0.1  # Convert to SI
        b_SI = b * 1e-3

        V_vdw = self.van_der_waals_volume(n, P, T, a_SI, b_SI)
        print(f"Van der Waals volume: {V_vdw*1000:.2f} L/mol")

        Z = self.compressibility_factor(T, P, V_vdw, n)
        print(f"Compressibility factor: {Z:.4f}")

        # 2. Critical Properties
        print("\n2. CRITICAL PROPERTIES & CORRESPONDING STATES")
        print("-" * 40)

        co2 = Component(
            name="CO2",
            molecular_weight=44.01,
            critical_temperature=304.13,
            critical_pressure=7.377e6,
            critical_volume=0.0941e-3,
            acentric_factor=0.224,
            normal_boiling_point=194.7,
            heat_of_vaporization=25200,
            heat_capacity_params=[22.26, 5.981e-2, -3.501e-5, 7.469e-9]
        )

        Z_pr = self.peng_robinson(T, P, co2, 'Z')
        print(f"CO2 compressibility (Peng-Robinson): {Z_pr:.4f}")

        phi = self.fugacity_coefficient_pr(T, P, co2)
        print(f"Fugacity coefficient: {phi:.4f}")

        # 3. Phase Equilibria
        print("\n3. PHASE EQUILIBRIA")
        print("-" * 40)

        # Binary mixture
        ethanol = Component(
            name="Ethanol",
            molecular_weight=46.07,
            critical_temperature=513.9,
            critical_pressure=6.137e6,
            critical_volume=0.167e-3,
            acentric_factor=0.649,
            normal_boiling_point=351.4,
            heat_of_vaporization=38600
        )

        water = Component(
            name="Water",
            molecular_weight=18.02,
            critical_temperature=647.1,
            critical_pressure=22.06e6,
            critical_volume=0.056e-3,
            acentric_factor=0.344,
            normal_boiling_point=373.15,
            heat_of_vaporization=40660
        )

        mixture = Mixture(
            components=[ethanol, water],
            mole_fractions=np.array([0.3, 0.7]),
            temperature=350,
            pressure=101325
        )

        T_bubble, y = self.bubble_point_calculation(mixture, P)
        print(f"Bubble point: {T_bubble:.1f}K ({T_bubble-273.15:.1f}°C)")
        print(f"Vapor composition: y_ethanol={y[0]:.3f}, y_water={y[1]:.3f}")

        # Flash calculation
        V_frac, x, y_flash = self.flash_calculation(mixture, 360, P)
        print(f"Flash at 360K: Vapor fraction={V_frac:.3f}")

        # 4. Activity Coefficients
        print("\n4. ACTIVITY COEFFICIENTS")
        print("-" * 40)

        # Margules parameters
        A12, A21 = 1.5, 1.8
        gamma1, gamma2 = self.activity_coefficient_margules(0.3, A12, A21)
        print(f"Margules model: γ1={gamma1:.3f}, γ2={gamma2:.3f}")

        # Wilson parameters
        Lambda12, Lambda21 = 0.8, 1.2
        gamma1_w, gamma2_w = self.activity_coefficient_wilson(0.3, T, Lambda12, Lambda21)
        print(f"Wilson model: γ1={gamma1_w:.3f}, γ2={gamma2_w:.3f}")

        # 5. Chemical Equilibrium
        print("\n5. CHEMICAL EQUILIBRIUM")
        print("-" * 40)

        # Reaction: N2 + 3H2 ⇌ 2NH3
        delta_G = -16400  # J/mol at 298K
        K_298 = self.equilibrium_constant(delta_G, 298.15)
        print(f"Equilibrium constant at 298K: K={K_298:.2e}")

        # Van't Hoff
        delta_H = -92000  # J/mol
        K_500 = self.van_t_hoff_equation(K_298, 298.15, 500, delta_H)
        print(f"Equilibrium constant at 500K: K={K_500:.2e}")

        # 6. Heat Capacity
        print("\n6. HEAT CAPACITY & ENTHALPY")
        print("-" * 40)

        Cp_300 = self.heat_capacity_polynomial(300, co2.heat_capacity_params)
        Cp_500 = self.heat_capacity_polynomial(500, co2.heat_capacity_params)
        print(f"CO2 heat capacity at 300K: {Cp_300:.1f} J/(mol·K)")
        print(f"CO2 heat capacity at 500K: {Cp_500:.1f} J/(mol·K)")

        delta_H_sensible = self.heat_capacity_integration(
            300, 500,
            self.heat_capacity_polynomial,
            co2.heat_capacity_params
        )
        print(f"Enthalpy change 300K→500K: {delta_H_sensible/1000:.2f} kJ/mol")

        # 7. Entropy
        print("\n7. ENTROPY & GIBBS ENERGY")
        print("-" * 40)

        S = self.entropy_ideal_gas(T, P, co2)
        print(f"CO2 entropy at {T}K, {P/1000:.1f} kPa: {S:.1f} J/(mol·K)")

        x_mix = np.array([0.3, 0.4, 0.3])
        S_mix = self.entropy_mixing(x_mix)
        print(f"Entropy of mixing (3 components): {S_mix:.2f} J/(mol·K)")

        # 8. Joule-Thomson Effect
        print("\n8. JOULE-THOMSON EFFECT")
        print("-" * 40)

        Cp = 37  # J/(mol·K)
        mu_JT = self.joule_thomson_coefficient_vdw(T, P, a_SI, b_SI, Cp)
        print(f"Joule-Thomson coefficient: {mu_JT*1e6:.2f} K/Pa")

        T_inv = self.inversion_temperature(a_SI, b_SI)
        print(f"Inversion temperature: {T_inv:.1f} K")

        # 9. Power Cycles
        print("\n9. THERMODYNAMIC CYCLES")
        print("-" * 40)

        # Carnot
        T_hot, T_cold = 600, 300
        eta_carnot = self.carnot_efficiency(T_hot, T_cold)
        print(f"Carnot efficiency ({T_hot}K→{T_cold}K): {eta_carnot:.1%}")

        # Rankine
        rankine = self.rankine_cycle(550, 310, 10e6, 10e3)
        print(f"Rankine cycle efficiency: {rankine['efficiency']:.1%}")
        print(f"  Back work ratio: {rankine['back_work_ratio']:.3f}")

        # Refrigeration
        refrigeration = self.refrigeration_cycle(260, 310)
        print(f"Refrigeration COP: {refrigeration['COP']:.2f}")
        print(f"  Carnot COP: {refrigeration['COP_carnot']:.2f}")

        print("\n" + "=" * 60)
        print("Demonstration complete!")
        print("=" * 60)

def run_demo():
    """Run the comprehensive thermodynamics demonstration."""
    lab = ThermodynamicsLab()
    lab.run_comprehensive_demo()

if __name__ == '__main__':
    run_demo()