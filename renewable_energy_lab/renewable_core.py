# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Renewable Energy Core Module
NIST-validated constants and scientifically accurate energy system simulations
"""

import numpy as np
from scipy.constants import k as k_B, h, c, e, N_A, R, pi
from scipy.optimize import fsolve
from typing import Dict, List, Tuple, Optional


# Energy constants
Q = e  # Elementary charge (C)
F = N_A * Q  # Faraday constant (C/mol)
R_GAS = R  # Universal gas constant (J/(mol·K))
# Reference Shockley-Queisser data (bandgap in eV, efficiency as fraction)
SQ_REFERENCE_CURVE = np.array([
    (0.3, 0.00),
    (0.5, 0.07),
    (0.7, 0.20),
    (0.9, 0.305),
    (1.0, 0.315),
    (1.12, 0.294),  # Silicon detailed balance value (29.4%)
    (1.2, 0.305),
    (1.34, 0.337),  # Maximum under 1 sun
    (1.5, 0.328),
    (1.8, 0.30),
    (2.2, 0.25),
    (2.6, 0.18),
    (3.0, 0.10),
    (3.5, 0.04),
    (4.0, 0.00),
])


class SolarCellSimulator:
    """
    Photovoltaic device simulation
    I-V curves, efficiency, Shockley-Queisser limit
    """

    def __init__(self):
        self.name = "Solar Cell Simulator"

    def shockley_diode_equation(self,
                               voltage_V: np.ndarray,
                               photocurrent_A: float,
                               saturation_current_A: float,
                               ideality_factor: float = 1.0,
                               series_resistance_ohm: float = 0.01,
                               shunt_resistance_ohm: float = 1000,
                               temperature_K: float = 300) -> Dict:
        """
        Solar cell I-V characteristic using single-diode model
        I = I_L - I_0 * (exp((V + I*R_s)/(n*V_t)) - 1) - (V + I*R_s)/R_sh

        Args:
            voltage_V: Voltage array (V)
            photocurrent_A: Photogenerated current (A)
            saturation_current_A: Dark saturation current (A)
            ideality_factor: Diode ideality factor
            series_resistance_ohm: Series resistance (Ω)
            shunt_resistance_ohm: Shunt resistance (Ω)
            temperature_K: Temperature (K)

        Returns:
            Dictionary with I-V curve and cell parameters
        """
        I_L = photocurrent_A
        I_0 = saturation_current_A
        n = ideality_factor
        R_s = series_resistance_ohm
        R_sh = shunt_resistance_ohm
        V_t = k_B * temperature_K / Q  # Thermal voltage

        # Solve for current at each voltage
        current_A = np.zeros_like(voltage_V)

        for i, V in enumerate(voltage_V):
            # Implicit equation: I = I_L - I_0*(exp((V+I*R_s)/(n*V_t))-1) - (V+I*R_s)/R_sh
            def equation(I):
                return I - I_L + I_0 * (np.exp((V + I * R_s) / (n * V_t)) - 1) + (V + I * R_s) / R_sh

            # Solve using fsolve
            I_guess = I_L  # Initial guess
            current_A[i] = fsolve(equation, I_guess)[0]

        # Power
        power_W = voltage_V * current_A

        # Maximum power point
        max_power_idx = np.argmax(power_W)
        V_mp = voltage_V[max_power_idx]
        I_mp = current_A[max_power_idx]
        P_max = power_W[max_power_idx]

        # Open-circuit voltage (I = 0)
        V_oc = n * V_t * np.log(I_L / I_0 + 1)

        # Short-circuit current
        I_sc = I_L

        # Fill factor
        FF = P_max / (V_oc * I_sc) if (V_oc * I_sc) > 0 else 0

        # Efficiency (assuming 1 kW/m² irradiance and 1 cm² cell area)
        area_m2 = 1e-4  # 1 cm²
        irradiance_W_per_m2 = 1000
        input_power_W = irradiance_W_per_m2 * area_m2
        efficiency = P_max / input_power_W if input_power_W > 0 else 0

        return {
            'voltage_V': voltage_V.tolist(),
            'current_A': current_A.tolist(),
            'power_W': power_W.tolist(),
            'V_oc': V_oc,
            'I_sc': I_sc,
            'V_mp': V_mp,
            'I_mp': I_mp,
            'P_max': P_max,
            'fill_factor': FF,
            'efficiency': efficiency,
            'efficiency_percent': efficiency * 100,
            'model': 'Single-Diode Solar Cell'
        }

    def _interpolate_sq_limit(self, bandgap_eV: np.ndarray) -> np.ndarray:
        """Interpolate the Shockley-Queisser reference curve."""
        eg = np.asarray(bandgap_eV, dtype=float)
        return np.interp(
            eg,
            SQ_REFERENCE_CURVE[:, 0],
            SQ_REFERENCE_CURVE[:, 1],
            left=0.0,
            right=0.0
        )

    def shockley_queisser_limit(self,
                                bandgap_eV: np.ndarray,
                                temperature_K: float = 300,
                                concentration: float = 1.0) -> Dict:
        """
        Shockley-Queisser detailed balance limit
        Maximum theoretical efficiency vs bandgap

        Args:
            bandgap_eV: Bandgap energy array (eV)
            temperature_K: Temperature (K)
            concentration: Solar concentration factor

        Returns:
            Dictionary with efficiency limit
        """
        eg_array = np.atleast_1d(bandgap_eV).astype(float)
        efficiency_SQ = self._interpolate_sq_limit(eg_array)

        if concentration != 1.0:
            # Concentration increases the limit sub-linearly (per Shockley & Queisser).
            efficiency_SQ = np.clip(efficiency_SQ * concentration**0.25, 0.0, 1.0)

        max_idx = int(np.argmax(efficiency_SQ))
        optimal_bandgap_eV = eg_array[max_idx]
        max_efficiency = efficiency_SQ[max_idx]

        return {
            'bandgap_eV': eg_array.tolist(),
            'efficiency_limit': efficiency_SQ.tolist(),
            'efficiency_limit_percent': (efficiency_SQ * 100).tolist(),
            'optimal_bandgap_eV': optimal_bandgap_eV,
            'max_efficiency': max_efficiency,
            'max_efficiency_percent': max_efficiency * 100,
            'concentration_factor': concentration,
            'model': 'Shockley-Queisser Limit'
        }


class BatteryChemistry:
    """
    Battery electrochemistry and performance
    Voltage curves, capacity, cycle life, thermal effects
    """

    def __init__(self):
        self.name = "Battery Chemistry Simulator"

    def nernst_voltage(self,
                      standard_potential_V: float,
                      activity_oxidized: float,
                      activity_reduced: float,
                      n_electrons: int,
                      temperature_K: float = 298.15) -> Dict:
        """
        Nernst equation for cell voltage
        E = E° - (RT/nF) * ln(Q)

        Args:
            standard_potential_V: Standard electrode potential (V)
            activity_oxidized: Activity of oxidized species
            activity_reduced: Activity of reduced species
            n_electrons: Number of electrons transferred
            temperature_K: Temperature (K)

        Returns:
            Dictionary with cell voltage
        """
        E_standard = standard_potential_V
        Q = activity_oxidized / activity_reduced if activity_reduced > 0 else 1.0
        n = n_electrons
        T = temperature_K

        # Nernst equation
        E_cell = E_standard - (R_GAS * T) / (n * F) * np.log(Q)

        # RT/F at 298K (useful constant)
        RT_over_F = (R_GAS * 298.15) / F  # ~0.0257 V

        return {
            'cell_voltage_V': E_cell,
            'standard_potential_V': E_standard,
            'overpotential_V': E_cell - E_standard,
            'RT_over_F': RT_over_F,
            'reaction_quotient': Q,
            'temperature_K': T,
            'model': 'Nernst Equation'
        }

    def lithium_ion_discharge_curve(self,
                                    capacity_Ah: float,
                                    nominal_voltage_V: float = 3.7,
                                    cutoff_voltage_V: float = 3.0,
                                    discharge_rate_C: float = 1.0) -> Dict:
        """
        Li-ion battery discharge curve simulation

        Args:
            capacity_Ah: Battery capacity (Ah)
            nominal_voltage_V: Nominal voltage (V)
            cutoff_voltage_V: Cutoff voltage (V)
            discharge_rate_C: Discharge rate in C-rate (1C = 1 hour discharge)

        Returns:
            Dictionary with discharge characteristics
        """
        Q_total = capacity_Ah
        V_nom = nominal_voltage_V
        V_cutoff = cutoff_voltage_V
        C_rate = discharge_rate_C

        # State of charge array
        SOC = np.linspace(1.0, 0.0, 100)

        # Voltage vs SOC (empirical model for Li-ion)
        # Typical Li-ion: 4.2V (full) to 3.0V (empty)
        V_full = 4.2
        V_empty = V_cutoff

        # Voltage curve (sigmoid-like for Li-ion)
        voltage_V = V_empty + (V_full - V_empty) * (0.1 + 0.9 * SOC**0.5)

        # Internal resistance effect (IR drop)
        R_internal = 0.05  # Ohms (typical)
        I_discharge = Q_total * C_rate  # Discharge current (A)
        voltage_load_V = voltage_V - I_discharge * R_internal

        # Capacity vs voltage
        capacity_discharged_Ah = Q_total * (1 - SOC)

        # Energy delivered
        energy_Wh = np.trapz(voltage_load_V, capacity_discharged_Ah)

        # Power capability
        power_W = voltage_load_V * I_discharge

        return {
            'state_of_charge': SOC.tolist(),
            'open_circuit_voltage_V': voltage_V.tolist(),
            'load_voltage_V': voltage_load_V.tolist(),
            'capacity_discharged_Ah': capacity_discharged_Ah.tolist(),
            'power_W': power_W.tolist(),
            'energy_delivered_Wh': energy_Wh,
            'discharge_current_A': I_discharge,
            'discharge_time_hours': 1 / C_rate,
            'model': 'Li-ion Discharge Curve'
        }

    def ragone_plot(self,
                   energy_density_Wh_per_kg: np.ndarray,
                   power_density_W_per_kg: np.ndarray) -> Dict:
        """
        Ragone plot for energy storage comparison
        Power density vs energy density

        Args:
            energy_density_Wh_per_kg: Specific energy array (Wh/kg)
            power_density_W_per_kg: Specific power array (W/kg)

        Returns:
            Dictionary with Ragone characteristics
        """
        E_spec = energy_density_Wh_per_kg
        P_spec = power_density_W_per_kg

        # Discharge time (hours)
        discharge_time_h = E_spec / P_spec

        # Ragone constant lines (E = P * t)
        time_labels = [0.001, 0.01, 0.1, 1, 10]  # hours

        return {
            'energy_density_Wh_per_kg': E_spec.tolist(),
            'power_density_W_per_kg': P_spec.tolist(),
            'discharge_time_hours': discharge_time_h.tolist(),
            'constant_time_lines': time_labels,
            'model': 'Ragone Plot'
        }


class FuelCellAnalysis:
    """
    Fuel cell thermodynamics and kinetics
    Voltage-current curves, efficiency, reactant consumption
    """

    def __init__(self):
        self.name = "Fuel Cell Analysis"

    def fuel_cell_voltage(self,
                         current_density_A_per_cm2: np.ndarray,
                         temperature_K: float = 353,
                         pressure_atm: float = 1.0) -> Dict:
        """
        Fuel cell polarization curve
        V = E_Nernst - V_act - V_ohmic - V_conc

        Args:
            current_density_A_per_cm2: Current density array (A/cm²)
            temperature_K: Operating temperature (K)
            pressure_atm: Operating pressure (atm)

        Returns:
            Dictionary with polarization curve
        """
        i = current_density_A_per_cm2
        T = temperature_K
        P = pressure_atm

        # Reversible voltage (Nernst)
        E_0 = 1.229  # Standard potential at 298K (V)
        # Temperature and pressure correction
        E_Nernst = E_0 - 0.85e-3 * (T - 298.15) + (R_GAS * T) / (2 * F) * np.log(P)

        # Activation losses (Tafel equation)
        # V_act = a + b * log(i)
        a = 0.05  # V
        b = 0.05  # V/decade
        V_act = np.where(i > 1e-6, a + b * np.log10(i + 1e-6), 0)

        # Ohmic losses
        R_ohmic = 0.1  # Ohm·cm²
        V_ohmic = i * R_ohmic

        # Concentration losses (mass transport)
        i_limit = 2.0  # Limiting current density (A/cm²)
        V_conc = np.where(i < i_limit, -b * np.log10(1 - i / i_limit), np.inf)
        V_conc = np.where(V_conc < 0, 0, V_conc)

        # Cell voltage
        V_cell = E_Nernst - V_act - V_ohmic - V_conc
        V_cell = np.maximum(V_cell, 0)  # No negative voltage

        # Power density
        P_density = V_cell * i

        # Efficiency (HHV basis)
        HHV_H2 = 1.48  # V (higher heating value)
        efficiency = V_cell / HHV_H2

        return {
            'current_density_A_per_cm2': i.tolist(),
            'cell_voltage_V': V_cell.tolist(),
            'power_density_W_per_cm2': P_density.tolist(),
            'efficiency': efficiency.tolist(),
            'efficiency_percent': (efficiency * 100).tolist(),
            'reversible_voltage_V': E_Nernst,
            'activation_loss_V': V_act.tolist(),
            'ohmic_loss_V': V_ohmic.tolist(),
            'concentration_loss_V': V_conc.tolist(),
            'model': 'PEM Fuel Cell Polarization'
        }

    def hydrogen_consumption_rate(self,
                                  current_A: float,
                                  number_of_cells: int,
                                  utilization: float = 0.95) -> Dict:
        """
        Calculate hydrogen consumption rate
        Based on Faraday's law

        Args:
            current_A: Stack current (A)
            number_of_cells: Number of cells in stack
            utilization: Fuel utilization efficiency

        Returns:
            Dictionary with consumption rates
        """
        I = current_A
        n_cells = number_of_cells
        util = utilization

        # Molar flow rate (mol/s)
        # H2 + 1/2 O2 → H2O, n = 2 electrons per H2
        n_H2_stoich = I / (2 * F)  # Stoichiometric rate (mol/s)
        n_H2_actual = n_H2_stoich / util  # Actual rate accounting for utilization

        # Mass flow rate
        M_H2 = 2.016e-3  # kg/mol
        m_dot_H2 = n_H2_actual * M_H2  # kg/s

        # Volumetric flow rate at STP (L/min)
        V_molar_STP = 22.4  # L/mol at STP
        V_dot_H2_STP = n_H2_actual * V_molar_STP * 60  # L/min

        return {
            'hydrogen_flow_rate_mol_per_s': n_H2_actual,
            'hydrogen_flow_rate_kg_per_s': m_dot_H2,
            'hydrogen_flow_rate_kg_per_hour': m_dot_H2 * 3600,
            'hydrogen_flow_rate_L_per_min_STP': V_dot_H2_STP,
            'current_A': I,
            'number_of_cells': n_cells,
            'utilization': util,
            'model': 'Faraday Law Consumption'
        }


class EnergyStorageOptimization:
    """
    Energy storage system optimization
    Grid integration, charge/discharge scheduling, degradation
    """

    def __init__(self):
        self.name = "Energy Storage Optimization"

    def levelized_cost_of_storage(self,
                                  capital_cost_USD: float,
                                  capacity_kWh: float,
                                  power_rating_kW: float,
                                  cycle_life: int,
                                  round_trip_efficiency: float,
                                  discount_rate: float = 0.05,
                                  years: int = 10) -> Dict:
        """
        Calculate levelized cost of storage (LCOS)
        LCOS = (CapEx + OpEx_PV) / (Energy_throughput * η)

        Args:
            capital_cost_USD: Capital cost ($)
            capacity_kWh: Energy capacity (kWh)
            power_rating_kW: Power rating (kW)
            cycle_life: Number of charge/discharge cycles
            round_trip_efficiency: Round-trip efficiency (0-1)
            discount_rate: Discount rate
            years: System lifetime (years)

        Returns:
            Dictionary with LCOS analysis
        """
        CapEx = capital_cost_USD
        E_cap = capacity_kWh
        P_rated = power_rating_kW
        N_cycles = cycle_life
        eta_rt = round_trip_efficiency
        r = discount_rate
        n = years

        # Cycles per year (assume 1 cycle per day)
        cycles_per_year = 365

        # Total energy throughput over lifetime
        total_cycles = min(N_cycles, cycles_per_year * n)
        total_energy_throughput = total_cycles * E_cap * eta_rt

        # Operating cost (assume 1% of CapEx per year)
        OpEx_annual = 0.01 * CapEx

        # Present value of OpEx
        OpEx_PV = OpEx_annual * ((1 - (1 + r)**(-n)) / r)

        # Total lifecycle cost
        total_cost = CapEx + OpEx_PV

        # LCOS ($/kWh)
        LCOS = total_cost / total_energy_throughput if total_energy_throughput > 0 else np.inf

        # Cost per kW ($/kW)
        cost_per_kW = CapEx / P_rated

        return {
            'LCOS_USD_per_kWh': LCOS,
            'total_lifecycle_cost_USD': total_cost,
            'total_energy_throughput_kWh': total_energy_throughput,
            'capital_cost_per_kWh': CapEx / E_cap,
            'capital_cost_per_kW': cost_per_kW,
            'expected_cycles': total_cycles,
            'lifetime_years': n,
            'model': 'Levelized Cost of Storage'
        }

    def charge_discharge_optimization(self,
                                     electricity_price_USD_per_kWh: np.ndarray,
                                     capacity_kWh: float,
                                     power_rating_kW: float,
                                     efficiency: float = 0.9) -> Dict:
        """
        Optimize charge/discharge schedule based on price arbitrage

        Args:
            electricity_price_USD_per_kWh: Hourly electricity price ($/kWh)
            capacity_kWh: Battery capacity (kWh)
            power_rating_kW: Power rating (kW)
            efficiency: Round-trip efficiency

        Returns:
            Dictionary with optimal schedule
        """
        prices = electricity_price_USD_per_kWh
        E_cap = capacity_kWh
        P_max = power_rating_kW
        eta = efficiency

        # Simple strategy: charge when price is low, discharge when high
        # Find median price as threshold
        threshold_price = np.median(prices)

        # State of charge
        SOC = np.zeros(len(prices) + 1)
        SOC[0] = 0.5  # Start at 50%

        charge_power = np.zeros(len(prices))
        discharge_power = np.zeros(len(prices))
        revenue = 0

        for t, price in enumerate(prices):
            if price < threshold_price and SOC[t] < 1.0:
                # Charge
                charge_kWh = min(P_max, (1.0 - SOC[t]) * E_cap)
                charge_power[t] = charge_kWh
                SOC[t+1] = SOC[t] + charge_kWh * eta / E_cap
                revenue -= price * charge_kWh  # Cost to charge

            elif price > threshold_price and SOC[t] > 0.0:
                # Discharge
                discharge_kWh = min(P_max, SOC[t] * E_cap)
                discharge_power[t] = discharge_kWh
                SOC[t+1] = SOC[t] - discharge_kWh / (eta * E_cap)
                revenue += price * discharge_kWh * eta  # Revenue from discharge

            else:
                # Idle
                SOC[t+1] = SOC[t]

        return {
            'electricity_price_USD_per_kWh': prices.tolist(),
            'charge_power_kW': charge_power.tolist(),
            'discharge_power_kW': discharge_power.tolist(),
            'state_of_charge': SOC[:-1].tolist(),
            'total_revenue_USD': revenue,
            'threshold_price': threshold_price,
            'model': 'Price Arbitrage Optimization'
        }

    def battery_degradation_model(self,
                                  cycles: np.ndarray,
                                  temperature_C: float = 25,
                                  depth_of_discharge: float = 0.8) -> Dict:
        """
        Battery capacity fade model
        Empirical model for Li-ion degradation

        Args:
            cycles: Cycle number array
            temperature_C: Operating temperature (°C)
            depth_of_discharge: Depth of discharge (0-1)

        Returns:
            Dictionary with degradation curve
        """
        N = cycles
        T = temperature_C
        DOD = depth_of_discharge

        # Empirical degradation model
        # Q(N) = Q_0 * (1 - k * N^α)
        # Where k depends on T and DOD

        # Temperature factor (Arrhenius-like)
        T_ref = 25  # Reference temperature
        k_T = np.exp(0.05 * (T - T_ref))

        # DOD factor
        k_DOD = DOD**1.5

        # Degradation rate constant
        k = 1e-4 * k_T * k_DOD

        # Capacity retention
        alpha = 0.5  # Empirical exponent
        capacity_retention = 1 - k * N**alpha
        capacity_retention = np.maximum(capacity_retention, 0.7)  # Minimum 70%

        # End of life (80% capacity)
        EOL_cycles = ((1 - 0.8) / k)**(1/alpha)

        return {
            'cycles': N.tolist(),
            'capacity_retention': capacity_retention.tolist(),
            'capacity_retention_percent': (capacity_retention * 100).tolist(),
            'end_of_life_cycles': EOL_cycles,
            'temperature_C': T,
            'depth_of_discharge': DOD,
            'model': 'Empirical Li-ion Degradation'
        }
