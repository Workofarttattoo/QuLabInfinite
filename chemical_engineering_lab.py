"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

CHEMICAL ENGINEERING LAB
Production-ready chemical engineering calculations for reactor design, separations, heat transfer, and process optimization.
Free gift to the scientific community from QuLabInfinite.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy import constants, optimize, integrate
from typing import Dict, Tuple, List, Optional, Callable
import warnings

@dataclass
class ChemicalEngineeringLab:
    """
    Comprehensive chemical engineering calculations including:
    - Reactor design (CSTR, PFR, Batch)
    - Mass and energy balances
    - Separation processes (distillation, extraction, absorption)
    - Heat exchanger design
    - Reaction kinetics
    - Process optimization
    """

    # Standard conditions
    temperature: float = 298.15  # K (25°C)
    pressure: float = 101325  # Pa (1 atm)
    gas_constant: float = constants.R  # J/(mol·K)

    # Physical properties (water as default)
    density: float = 997  # kg/m³
    viscosity: float = 8.9e-4  # Pa·s
    heat_capacity: float = 4184  # J/(kg·K)
    thermal_conductivity: float = 0.606  # W/(m·K)

    def __post_init__(self):
        """Initialize derived properties"""
        self.avogadro = constants.Avogadro
        self.boltzmann = constants.k

    def ideal_gas_law(self, p: Optional[float] = None, v: Optional[float] = None,
                     n: Optional[float] = None, t: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate missing value using ideal gas law: PV = nRT

        Args:
            p: Pressure in Pa
            v: Volume in m³
            n: Moles
            t: Temperature in K

        Returns:
            Dictionary with all four values
        """
        r = self.gas_constant

        # Count known values
        known = sum(x is not None for x in [p, v, n, t])
        if known != 3:
            raise ValueError("Exactly three values must be provided")

        if p is None:
            p = n * r * t / v
        elif v is None:
            v = n * r * t / p
        elif n is None:
            n = p * v / (r * t)
        elif t is None:
            t = p * v / (n * r)

        return {'pressure': p, 'volume': v, 'moles': n, 'temperature': t}

    def reaction_rate(self, concentration: np.ndarray, rate_constant: float,
                     order: List[float], temperature: Optional[float] = None) -> float:
        """
        Calculate reaction rate using power law kinetics

        Rate = k * ∏(C_i^n_i)

        Args:
            concentration: Concentration array in mol/m³
            rate_constant: Rate constant (units depend on order)
            order: Reaction order for each component
            temperature: Temperature in K (for Arrhenius correction)

        Returns:
            Reaction rate in mol/(m³·s)
        """
        if len(concentration) != len(order):
            raise ValueError("Concentration and order arrays must have same length")

        rate = rate_constant
        for c, n in zip(concentration, order):
            rate *= c ** n

        return rate

    def arrhenius_equation(self, pre_exponential: float, activation_energy: float,
                          temperature: Optional[float] = None) -> float:
        """
        Calculate rate constant using Arrhenius equation

        k = A * exp(-Ea/(RT))

        Args:
            pre_exponential: Pre-exponential factor (same units as k)
            activation_energy: Activation energy in J/mol
            temperature: Temperature in K (default: self.temperature)

        Returns:
            Rate constant
        """
        if temperature is None:
            temperature = self.temperature

        return pre_exponential * np.exp(-activation_energy / (self.gas_constant * temperature))

    def cstr_design(self, flow_rate: float, concentration_in: float,
                   rate_constant: float, reaction_order: float,
                   conversion: float) -> Dict[str, float]:
        """
        Design Continuous Stirred Tank Reactor (CSTR)

        Args:
            flow_rate: Volumetric flow rate in m³/s
            concentration_in: Inlet concentration in mol/m³
            rate_constant: Rate constant (units depend on order)
            reaction_order: Reaction order
            conversion: Desired conversion (0-1)

        Returns:
            Dictionary with reactor volume, residence time, outlet concentration
        """
        if not 0 <= conversion <= 1:
            raise ValueError("Conversion must be between 0 and 1")

        concentration_out = concentration_in * (1 - conversion)

        if reaction_order == 0:  # Zero order
            residence_time = concentration_in * conversion / rate_constant
        elif reaction_order == 1:  # First order
            residence_time = conversion / (rate_constant * (1 - conversion))
        elif reaction_order == 2:  # Second order
            residence_time = conversion / (rate_constant * concentration_in * (1 - conversion)**2)
        else:  # General nth order
            n = reaction_order
            residence_time = (concentration_in**(1-n) *
                            ((1/(1-conversion)**(n-1)) - 1) /
                            ((n-1) * rate_constant))

        volume = flow_rate * residence_time

        return {
            'volume': volume,  # m³
            'residence_time': residence_time,  # s
            'concentration_out': concentration_out,  # mol/m³
            'space_time': volume / flow_rate  # s
        }

    def pfr_design(self, flow_rate: float, concentration_in: float,
                  rate_constant: float, reaction_order: float,
                  conversion: float) -> Dict[str, float]:
        """
        Design Plug Flow Reactor (PFR)

        Args:
            flow_rate: Volumetric flow rate in m³/s
            concentration_in: Inlet concentration in mol/m³
            rate_constant: Rate constant (units depend on order)
            reaction_order: Reaction order
            conversion: Desired conversion (0-1)

        Returns:
            Dictionary with reactor volume, residence time, outlet concentration
        """
        if not 0 <= conversion <= 1:
            raise ValueError("Conversion must be between 0 and 1")

        concentration_out = concentration_in * (1 - conversion)

        if reaction_order == 0:  # Zero order
            residence_time = concentration_in * conversion / rate_constant
        elif reaction_order == 1:  # First order
            residence_time = -np.log(1 - conversion) / rate_constant
        elif reaction_order == 2:  # Second order
            residence_time = conversion / (rate_constant * concentration_in * (1 - conversion))
        else:  # General nth order (numerical integration)
            def rate_func(x):
                return 1 / (rate_constant * concentration_in**(reaction_order-1) *
                          (1 - x)**reaction_order)

            residence_time, _ = integrate.quad(rate_func, 0, conversion)
            residence_time *= concentration_in**(reaction_order-1)

        volume = flow_rate * residence_time

        return {
            'volume': volume,  # m³
            'residence_time': residence_time,  # s
            'concentration_out': concentration_out,  # mol/m³
            'space_velocity': flow_rate / volume  # 1/s
        }

    def batch_reactor(self, initial_concentration: float, rate_constant: float,
                     reaction_order: float, time: np.ndarray) -> np.ndarray:
        """
        Calculate concentration profile in batch reactor

        Args:
            initial_concentration: Initial concentration in mol/m³
            rate_constant: Rate constant (units depend on order)
            reaction_order: Reaction order
            time: Time array in seconds

        Returns:
            Concentration array in mol/m³
        """
        c0 = initial_concentration
        k = rate_constant
        n = reaction_order

        if n == 0:  # Zero order
            concentration = np.maximum(c0 - k * time, 0)
        elif n == 1:  # First order
            concentration = c0 * np.exp(-k * time)
        elif n == 2:  # Second order
            concentration = c0 / (1 + k * c0 * time)
        else:  # General nth order
            if n != 1:
                concentration = (c0**(1-n) + (n-1) * k * time)**(1/(1-n))
                concentration = np.where(concentration > 0, concentration, 0)
            else:
                concentration = c0 * np.exp(-k * time)

        return concentration

    def mass_balance(self, inlet_flow: Dict[str, float], outlet_flow: Dict[str, float],
                    generation: Dict[str, float] = None,
                    consumption: Dict[str, float] = None) -> Dict[str, float]:
        """
        Perform component mass balance

        Accumulation = In - Out + Generation - Consumption

        Args:
            inlet_flow: Component inlet flows in kg/s or mol/s
            outlet_flow: Component outlet flows in kg/s or mol/s
            generation: Component generation rates (default: 0)
            consumption: Component consumption rates (default: 0)

        Returns:
            Dictionary with accumulation rates for each component
        """
        components = set(inlet_flow.keys()) | set(outlet_flow.keys())

        if generation is None:
            generation = {}
        if consumption is None:
            consumption = {}

        accumulation = {}
        for comp in components:
            in_rate = inlet_flow.get(comp, 0)
            out_rate = outlet_flow.get(comp, 0)
            gen_rate = generation.get(comp, 0)
            cons_rate = consumption.get(comp, 0)

            accumulation[comp] = in_rate - out_rate + gen_rate - cons_rate

        return accumulation

    def energy_balance(self, heat_in: float, heat_out: float,
                      work: float = 0, heat_reaction: float = 0) -> float:
        """
        Perform energy balance

        Accumulation = Heat_in - Heat_out - Work + Heat_reaction

        Args:
            heat_in: Heat input rate in W
            heat_out: Heat output rate in W
            work: Work done by system in W
            heat_reaction: Heat of reaction in W (negative for exothermic)

        Returns:
            Energy accumulation rate in W
        """
        return heat_in - heat_out - work + heat_reaction

    def mccabe_thiele_stages(self, x_feed: float, x_distillate: float,
                           x_bottoms: float, reflux_ratio: float,
                           alpha: float) -> int:
        """
        Calculate number of theoretical stages for binary distillation using McCabe-Thiele

        Args:
            x_feed: Feed mole fraction of more volatile component
            x_distillate: Distillate mole fraction
            x_bottoms: Bottoms mole fraction
            reflux_ratio: Reflux ratio (L/D)
            alpha: Relative volatility

        Returns:
            Number of theoretical stages
        """
        # Rectifying section operating line slope
        slope_rect = reflux_ratio / (reflux_ratio + 1)

        # q-line (assuming saturated liquid feed)
        q = 1  # Saturated liquid

        # Step off stages
        x = x_distillate
        stages = 0

        while x > x_bottoms and stages < 100:  # Limit iterations
            # Equilibrium curve: y = alpha*x / (1 + (alpha-1)*x)
            y = alpha * x / (1 + (alpha - 1) * x)

            # Operating line
            if x > x_feed:  # Rectifying section
                x_new = (y - x_distillate/(reflux_ratio + 1)) / slope_rect
            else:  # Stripping section
                # Simplified stripping line
                x_new = y / alpha

            x = x_new
            stages += 1

        return stages

    def kremser_equation(self, y_in: float, x_in: float, flow_gas: float,
                        flow_liquid: float, henry_constant: float,
                        n_stages: int) -> Tuple[float, float]:
        """
        Calculate outlet compositions for gas absorption using Kremser equation

        Args:
            y_in: Inlet gas mole fraction
            x_in: Inlet liquid mole fraction
            flow_gas: Gas flow rate in mol/s
            flow_liquid: Liquid flow rate in mol/s
            henry_constant: Henry's law constant (y* = H*x)
            n_stages: Number of stages

        Returns:
            Tuple of (y_out, x_out) outlet mole fractions
        """
        # Absorption factor
        A = flow_liquid / (henry_constant * flow_gas)

        if abs(A - 1) < 1e-6:  # A ≈ 1
            fraction_absorbed = n_stages / (n_stages + 1)
        else:
            fraction_absorbed = (A**(n_stages + 1) - A) / (A**(n_stages + 1) - 1)

        # Calculate outlet compositions
        absorbed = flow_gas * y_in * fraction_absorbed
        y_out = y_in * (1 - fraction_absorbed)
        x_out = x_in + absorbed / flow_liquid

        return y_out, x_out

    def heat_exchanger_lmtd(self, t_hot_in: float, t_hot_out: float,
                          t_cold_in: float, t_cold_out: float,
                          flow_type: str = 'counter') -> float:
        """
        Calculate Log Mean Temperature Difference (LMTD) for heat exchanger

        Args:
            t_hot_in: Hot fluid inlet temperature in K
            t_hot_out: Hot fluid outlet temperature in K
            t_cold_in: Cold fluid inlet temperature in K
            t_cold_out: Cold fluid outlet temperature in K
            flow_type: 'counter' or 'parallel' flow

        Returns:
            LMTD in K
        """
        if flow_type == 'counter':
            dt1 = t_hot_in - t_cold_out
            dt2 = t_hot_out - t_cold_in
        elif flow_type == 'parallel':
            dt1 = t_hot_in - t_cold_in
            dt2 = t_hot_out - t_cold_out
        else:
            raise ValueError("flow_type must be 'counter' or 'parallel'")

        if abs(dt1 - dt2) < 1e-6:
            return dt1
        else:
            return (dt1 - dt2) / np.log(dt1 / dt2)

    def heat_exchanger_design(self, duty: float, overall_htc: float,
                            t_hot_in: float, t_hot_out: float,
                            t_cold_in: float, t_cold_out: float,
                            flow_type: str = 'counter') -> Dict[str, float]:
        """
        Design heat exchanger using LMTD method

        Args:
            duty: Heat duty in W
            overall_htc: Overall heat transfer coefficient in W/(m²·K)
            t_hot_in: Hot fluid inlet temperature in K
            t_hot_out: Hot fluid outlet temperature in K
            t_cold_in: Cold fluid inlet temperature in K
            t_cold_out: Cold fluid outlet temperature in K
            flow_type: 'counter' or 'parallel' flow

        Returns:
            Dictionary with area, LMTD, and NTU
        """
        lmtd = self.heat_exchanger_lmtd(t_hot_in, t_hot_out, t_cold_in,
                                       t_cold_out, flow_type)

        area = duty / (overall_htc * lmtd)

        # Calculate flow rates (assuming equal heat capacities for simplicity)
        cp = self.heat_capacity
        flow_hot = duty / (cp * abs(t_hot_in - t_hot_out))
        flow_cold = duty / (cp * abs(t_cold_out - t_cold_in))

        # Number of Transfer Units
        ntu = overall_htc * area / (min(flow_hot, flow_cold) * cp)

        return {
            'area': area,  # m²
            'lmtd': lmtd,  # K
            'ntu': ntu,
            'flow_hot': flow_hot,  # kg/s
            'flow_cold': flow_cold  # kg/s
        }

    def reynolds_number(self, velocity: float, diameter: float,
                       density: Optional[float] = None,
                       viscosity: Optional[float] = None) -> float:
        """
        Calculate Reynolds number for pipe flow

        Re = ρvD/μ

        Args:
            velocity: Flow velocity in m/s
            diameter: Pipe diameter in m
            density: Fluid density in kg/m³ (default: self.density)
            viscosity: Dynamic viscosity in Pa·s (default: self.viscosity)

        Returns:
            Reynolds number (dimensionless)
        """
        if density is None:
            density = self.density
        if viscosity is None:
            viscosity = self.viscosity

        return density * velocity * diameter / viscosity

    def pressure_drop_ergun(self, velocity: float, particle_diameter: float,
                           bed_length: float, void_fraction: float,
                           density: Optional[float] = None,
                           viscosity: Optional[float] = None) -> float:
        """
        Calculate pressure drop through packed bed using Ergun equation

        Args:
            velocity: Superficial velocity in m/s
            particle_diameter: Particle diameter in m
            bed_length: Bed length in m
            void_fraction: Bed void fraction (0-1)
            density: Fluid density in kg/m³
            viscosity: Dynamic viscosity in Pa·s

        Returns:
            Pressure drop in Pa
        """
        if density is None:
            density = self.density
        if viscosity is None:
            viscosity = self.viscosity

        # Ergun equation
        term1 = 150 * viscosity * (1 - void_fraction)**2 * velocity
        term1 /= (particle_diameter**2 * void_fraction**3)

        term2 = 1.75 * density * (1 - void_fraction) * velocity**2
        term2 /= (particle_diameter * void_fraction**3)

        return bed_length * (term1 + term2)

    def vle_raoults_law(self, x: np.ndarray, vapor_pressure: np.ndarray,
                       total_pressure: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """
        Calculate vapor-liquid equilibrium using Raoult's Law

        Args:
            x: Liquid mole fractions
            vapor_pressure: Pure component vapor pressures in Pa
            total_pressure: Total pressure in Pa (default: self.pressure)

        Returns:
            Tuple of (vapor mole fractions, calculated total pressure)
        """
        if total_pressure is None:
            total_pressure = self.pressure

        # Calculate partial pressures
        partial_pressures = x * vapor_pressure

        # Total pressure (for bubble point)
        p_total = np.sum(partial_pressures)

        # Vapor mole fractions
        y = partial_pressures / p_total

        return y, p_total

    def antoine_equation(self, temperature: float, a: float, b: float,
                        c: float) -> float:
        """
        Calculate vapor pressure using Antoine equation

        log10(P) = A - B/(C + T)

        Args:
            temperature: Temperature in K
            a, b, c: Antoine coefficients

        Returns:
            Vapor pressure in Pa
        """
        log_p = a - b / (c + temperature - 273.15)  # Convert to Celsius
        return 10**log_p * 133.322  # Convert from mmHg to Pa

    def fugacity_coefficient(self, pressure: float, temperature: float,
                           critical_pressure: float, critical_temperature: float,
                           acentric_factor: float) -> float:
        """
        Calculate fugacity coefficient using Peng-Robinson EOS

        Args:
            pressure: Pressure in Pa
            temperature: Temperature in K
            critical_pressure: Critical pressure in Pa
            critical_temperature: Critical temperature in K
            acentric_factor: Acentric factor

        Returns:
            Fugacity coefficient (dimensionless)
        """
        # Reduced properties
        Tr = temperature / critical_temperature
        Pr = pressure / critical_pressure

        # Peng-Robinson parameters
        kappa = 0.37464 + 1.54226 * acentric_factor - 0.26992 * acentric_factor**2
        alpha = (1 + kappa * (1 - np.sqrt(Tr)))**2

        a = 0.45724 * (self.gas_constant * critical_temperature)**2 / critical_pressure * alpha
        b = 0.07780 * self.gas_constant * critical_temperature / critical_pressure

        # Compressibility factor (simplified)
        A = a * pressure / (self.gas_constant * temperature)**2
        B = b * pressure / (self.gas_constant * temperature)

        # Cubic equation coefficients
        coeffs = [1, -(1 - B), A - 2*B - 3*B**2, -(A*B - B**2 - B**3)]

        # Solve for compressibility
        roots = np.roots(coeffs)
        Z = np.real(roots[np.isreal(roots)])[0]  # Take real root

        # Fugacity coefficient
        ln_phi = Z - 1 - np.log(Z - B) - A/(2*np.sqrt(2)*B) * \
                np.log((Z + (1 + np.sqrt(2))*B)/(Z + (1 - np.sqrt(2))*B))

        return np.exp(ln_phi)

    def process_optimization(self, objective_func: Callable, constraints: List[Dict],
                           bounds: List[Tuple], initial_guess: np.ndarray) -> Dict[str, any]:
        """
        Optimize process parameters

        Args:
            objective_func: Objective function to minimize
            constraints: List of constraint dictionaries
            bounds: Variable bounds as list of (min, max) tuples
            initial_guess: Initial guess for variables

        Returns:
            Dictionary with optimal values and objective
        """
        result = optimize.minimize(
            objective_func,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        return {
            'optimal_values': result.x,
            'objective': result.fun,
            'success': result.success,
            'message': result.message,
            'iterations': result.nit
        }


def run_demo():
    """Demonstrate chemical engineering calculations"""
    print("=" * 70)
    print("CHEMICAL ENGINEERING LAB - Production Demo")
    print("=" * 70)

    lab = ChemicalEngineeringLab()

    # Reactor Design
    print("\n1. REACTOR DESIGN")
    print("-" * 40)

    # CSTR Design
    cstr = lab.cstr_design(0.01, 100, 0.05, 1, 0.8)
    print(f"CSTR Design (80% conversion):")
    print(f"  Volume: {cstr['volume']:.3f} m³")
    print(f"  Residence time: {cstr['residence_time']:.1f} s")

    # PFR Design
    pfr = lab.pfr_design(0.01, 100, 0.05, 1, 0.8)
    print(f"PFR Design (80% conversion):")
    print(f"  Volume: {pfr['volume']:.3f} m³")
    print(f"  Residence time: {pfr['residence_time']:.1f} s")

    # Batch Reactor
    time = np.linspace(0, 100, 50)
    conc = lab.batch_reactor(100, 0.05, 1, time)
    print(f"Batch Reactor:")
    print(f"  Initial conc: 100 mol/m³")
    print(f"  Final conc (100s): {conc[-1]:.2f} mol/m³")

    # Mass and Energy Balance
    print("\n2. MASS & ENERGY BALANCES")
    print("-" * 40)
    inlet = {'A': 10, 'B': 5}
    outlet = {'A': 8, 'B': 4, 'C': 2}
    generation = {'C': 2}
    consumption = {'A': 2, 'B': 1}

    balance = lab.mass_balance(inlet, outlet, generation, consumption)
    print(f"Mass Balance:")
    for comp, acc in balance.items():
        print(f"  Component {comp}: {acc:.2f} kg/s accumulation")

    energy = lab.energy_balance(1000, 800, 50, -100)
    print(f"Energy Balance: {energy:.0f} W accumulation")

    # Separation Processes
    print("\n3. SEPARATION PROCESSES")
    print("-" * 40)
    stages = lab.mccabe_thiele_stages(0.5, 0.95, 0.05, 3, 2.5)
    print(f"Distillation (McCabe-Thiele):")
    print(f"  Theoretical stages: {stages}")

    y_out, x_out = lab.kremser_equation(0.1, 0, 100, 1000, 0.01, 5)
    print(f"Gas Absorption (5 stages):")
    print(f"  Gas outlet: {y_out:.4f}")
    print(f"  Liquid outlet: {x_out:.4f}")

    # Heat Transfer
    print("\n4. HEAT EXCHANGER DESIGN")
    print("-" * 40)
    hx = lab.heat_exchanger_design(50000, 500, 400, 320, 300, 380)
    print(f"Counter-flow Heat Exchanger:")
    print(f"  Area: {hx['area']:.2f} m²")
    print(f"  LMTD: {hx['lmtd']:.1f} K")
    print(f"  NTU: {hx['ntu']:.2f}")

    # Fluid Flow
    print("\n5. FLUID FLOW")
    print("-" * 40)
    Re = lab.reynolds_number(2.0, 0.05)
    print(f"Reynolds number (2 m/s, 50mm pipe): {Re:.0f}")

    dp = lab.pressure_drop_ergun(0.1, 0.005, 1.0, 0.4)
    print(f"Packed bed pressure drop: {dp/1000:.2f} kPa")

    # Thermodynamics
    print("\n6. THERMODYNAMIC PROPERTIES")
    print("-" * 40)
    ideal = lab.ideal_gas_law(p=101325, v=0.0224, t=273.15)
    print(f"Ideal gas (STP): n = {ideal['moles']:.3f} mol")

    # VLE calculation
    x = np.array([0.3, 0.7])
    p_sat = np.array([50000, 20000])
    y, p_total = lab.vle_raoults_law(x, p_sat)
    print(f"VLE (Raoult's Law):")
    print(f"  Vapor composition: {y}")
    print(f"  Total pressure: {p_total/1000:.1f} kPa")

    print("\n" + "=" * 70)
    print("Demo completed successfully!")


if __name__ == '__main__':
    run_demo()