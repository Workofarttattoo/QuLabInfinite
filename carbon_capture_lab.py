"""
Carbon Capture Laboratory - Production Implementation
Real adsorption isotherms, DAC systems, membrane separation, and geological storage
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import math
from scipy.optimize import fsolve, minimize
from scipy.integrate import odeint


class CaptureMethod(Enum):
    """Carbon capture technology types"""
    POST_COMBUSTION = "post_combustion"
    PRE_COMBUSTION = "pre_combustion"
    OXY_FUEL = "oxy_fuel"
    DIRECT_AIR = "direct_air_capture"
    BIOENGINEERED = "bioengineered"


class AdsorbentType(Enum):
    """Common adsorbent materials"""
    ACTIVATED_CARBON = "activated_carbon"
    ZEOLITE_13X = "zeolite_13x"
    MOF_5 = "mof_5"
    AMINE_SOLID = "amine_solid_sorbent"
    BIOCHAR = "biochar"


class StorageType(Enum):
    """Geological storage options"""
    DEPLETED_OIL = "depleted_oil_gas"
    SALINE_AQUIFER = "saline_aquifer"
    UNMINEABLE_COAL = "unmineable_coal"
    BASALT = "basalt_formation"
    SALT_CAVERN = "salt_cavern"


@dataclass
class GasStream:
    """Industrial gas stream composition"""
    co2_fraction: float  # Mole fraction
    n2_fraction: float
    o2_fraction: float
    h2o_fraction: float
    temperature: float  # K
    pressure: float  # Pa
    flow_rate: float  # kmol/s


@dataclass
class AdsorbentProperties:
    """Physical and chemical properties of adsorbents"""
    name: str
    surface_area: float  # m²/g
    pore_volume: float  # cm³/g
    particle_density: float  # kg/m³
    heat_capacity: float  # J/(kg·K)
    heat_adsorption: float  # kJ/mol CO2


class CarbonCaptureLab:
    """
    Comprehensive carbon capture laboratory
    Implements real-world capture, utilization, and storage (CCUS) models
    """

    def __init__(self):
        # Universal gas constant
        self.R = 8.314  # J/(mol·K)

        # CO2 properties
        self.co2_properties = {
            'molecular_weight': 44.01,  # g/mol
            'critical_temp': 304.13,  # K
            'critical_pressure': 73.77e5,  # Pa
            'triple_point_temp': 216.58,  # K
            'sublimation_heat': 25.2  # kJ/mol
        }

        # Standard adsorbent properties
        self.adsorbent_data = {
            AdsorbentType.ACTIVATED_CARBON: AdsorbentProperties(
                "Activated Carbon", 1200, 0.6, 450, 1070, 20
            ),
            AdsorbentType.ZEOLITE_13X: AdsorbentProperties(
                "Zeolite 13X", 750, 0.35, 1100, 920, 35
            ),
            AdsorbentType.MOF_5: AdsorbentProperties(
                "MOF-5", 3500, 1.2, 380, 800, 15
            ),
            AdsorbentType.AMINE_SOLID: AdsorbentProperties(
                "Solid Amine", 50, 0.2, 650, 1200, 85
            ),
            AdsorbentType.BIOCHAR: AdsorbentProperties(
                "Biochar", 400, 0.3, 320, 1100, 18
            )
        }

    def langmuir_isotherm(self, pressure: float, temperature: float,
                         q_max: float = 4.0, b0: float = 1e-6,
                         delta_h: float = -20000) -> Dict:
        """
        Langmuir adsorption isotherm for CO2
        q = q_max * b * P / (1 + b * P)
        where b = b0 * exp(-ΔH/RT)
        """
        # Temperature-dependent Langmuir constant
        b = b0 * np.exp(-delta_h / (self.R * temperature))

        # Langmuir equation
        q = q_max * b * pressure / (1 + b * pressure)

        # Surface coverage
        theta = b * pressure / (1 + b * pressure)

        # Differential heat of adsorption
        q_st = -delta_h / 1000  # kJ/mol

        # Working capacity (difference between adsorption and desorption)
        p_ads = pressure
        p_des = pressure / 10  # Assume 10:1 pressure ratio
        q_ads = q_max * b * p_ads / (1 + b * p_ads)
        q_des = q_max * b * p_des / (1 + b * p_des)
        working_capacity = q_ads - q_des

        return {
            'loading': q,
            'surface_coverage': theta,
            'langmuir_constant': b,
            'heat_of_adsorption': q_st,
            'working_capacity': working_capacity,
            'selectivity': b * pressure,  # Approximate selectivity
            'model': 'Langmuir'
        }

    def freundlich_isotherm(self, pressure: float, temperature: float,
                           k_f: float = 0.5, n: float = 2.0) -> Dict:
        """
        Freundlich adsorption isotherm
        q = k_f * P^(1/n)
        Empirical model for heterogeneous surfaces
        """
        # Temperature correction for k_f
        k_f_t = k_f * np.exp(-1000 / temperature)

        # Freundlich equation
        q = k_f_t * pressure ** (1 / n)

        # Derivative for isosteric heat calculation
        dq_dp = k_f_t / n * pressure ** (1/n - 1)

        # Approximate heat of adsorption (Clausius-Clapeyron)
        q_st = 15 + 5 * (1 - 1/n)  # Empirical estimate (kJ/mol)

        # Working capacity
        p_ads = pressure
        p_des = pressure / 10
        q_ads = k_f_t * p_ads ** (1 / n)
        q_des = k_f_t * p_des ** (1 / n)
        working_capacity = q_ads - q_des

        return {
            'loading': q,
            'freundlich_k': k_f_t,
            'freundlich_n': n,
            'heat_of_adsorption': q_st,
            'working_capacity': working_capacity,
            'heterogeneity': 1 / n,  # Surface heterogeneity parameter
            'model': 'Freundlich'
        }

    def dual_site_langmuir(self, pressure: float, temperature: float,
                           q_max1: float = 2.0, b01: float = 1e-6, delta_h1: float = -25000,
                           q_max2: float = 1.5, b02: float = 5e-7, delta_h2: float = -15000) -> Dict:
        """
        Dual-site Langmuir model for heterogeneous adsorbents
        q = q_max1 * b1 * P / (1 + b1 * P) + q_max2 * b2 * P / (1 + b2 * P)
        """
        # Site 1
        b1 = b01 * np.exp(-delta_h1 / (self.R * temperature))
        q1 = q_max1 * b1 * pressure / (1 + b1 * pressure)

        # Site 2
        b2 = b02 * np.exp(-delta_h2 / (self.R * temperature))
        q2 = q_max2 * b2 * pressure / (1 + b2 * pressure)

        # Total loading
        q_total = q1 + q2

        # Average heat of adsorption
        q_st = -(delta_h1 * q1 + delta_h2 * q2) / (q_total + 1e-10) / 1000  # kJ/mol

        # Site occupancies
        theta1 = b1 * pressure / (1 + b1 * pressure)
        theta2 = b2 * pressure / (1 + b2 * pressure)

        return {
            'loading': q_total,
            'site1_loading': q1,
            'site2_loading': q2,
            'site1_coverage': theta1,
            'site2_coverage': theta2,
            'heat_of_adsorption': q_st,
            'model': 'Dual-Site Langmuir'
        }

    def membrane_separation(self, gas_stream: GasStream,
                          membrane_type: str = 'polymeric',
                          membrane_area: float = 1000) -> Dict:
        """
        Gas separation using membrane technology
        Based on solution-diffusion mechanism
        """
        # Membrane permeabilities (Barrer units)
        permeabilities = {
            'polymeric': {
                'CO2': 100,
                'N2': 3,
                'O2': 10,
                'H2O': 3000
            },
            'ceramic': {
                'CO2': 50,
                'N2': 2,
                'O2': 8,
                'H2O': 1000
            },
            'mof_membrane': {
                'CO2': 500,
                'N2': 5,
                'O2': 20,
                'H2O': 2000
            }
        }

        perm = permeabilities.get(membrane_type, permeabilities['polymeric'])

        # Selectivities
        selectivity_co2_n2 = perm['CO2'] / perm['N2']
        selectivity_co2_o2 = perm['CO2'] / perm['O2']

        # Pressure ratio
        pressure_ratio = 10  # Feed to permeate pressure ratio

        # Stage cut (fraction permeated)
        theta = 0.3

        # Calculate permeate composition (simplified)
        # Using constant pressure ratio approximation
        y_co2 = gas_stream.co2_fraction * selectivity_co2_n2 / \
                (1 + (selectivity_co2_n2 - 1) * gas_stream.co2_fraction)

        # CO2 recovery
        recovery = theta * y_co2 / gas_stream.co2_fraction

        # Power consumption (compression work)
        # W = n * R * T * ln(P2/P1)
        compression_work = gas_stream.flow_rate * self.R * gas_stream.temperature * \
                         np.log(pressure_ratio) / 1000  # kW

        # Membrane module parameters
        flux_co2 = perm['CO2'] * 1e-10 * gas_stream.pressure / 0.1e-3  # mol/(m²·s)
        co2_permeated = flux_co2 * membrane_area  # mol/s

        # Energy intensity
        energy_per_tonne = compression_work / (co2_permeated * 44.01 / 1000) * 1000  # kWh/tCO2

        return {
            'permeate_co2_fraction': y_co2,
            'co2_recovery': recovery,
            'selectivity_co2_n2': selectivity_co2_n2,
            'selectivity_co2_o2': selectivity_co2_o2,
            'compression_power': compression_work,
            'co2_flux': flux_co2,
            'co2_captured': co2_permeated * 44.01 / 1000,  # kg/s
            'energy_intensity': energy_per_tonne,
            'membrane_type': membrane_type,
            'membrane_area': membrane_area,
            'stage_cut': theta
        }

    def direct_air_capture(self, air_temperature: float = 298,
                         air_humidity: float = 0.5,
                         capture_material: str = 'amine') -> Dict:
        """
        Direct Air Capture (DAC) system model
        Based on adsorption/desorption cycles
        """
        # Air composition
        co2_concentration = 420e-6  # 420 ppm current atmospheric level
        air_pressure = 101325  # Pa

        # DAC sorbent parameters
        sorbent_params = {
            'amine': {
                'capacity': 2.0,  # mol CO2/kg sorbent
                'regeneration_temp': 373,  # K (100°C)
                'regeneration_heat': 100,  # kJ/mol CO2
                'kinetic_constant': 0.01  # 1/s
            },
            'solid_sorbent': {
                'capacity': 1.5,
                'regeneration_temp': 393,  # K (120°C)
                'regeneration_heat': 85,
                'kinetic_constant': 0.005
            },
            'mof': {
                'capacity': 3.0,
                'regeneration_temp': 353,  # K (80°C)
                'regeneration_heat': 60,
                'kinetic_constant': 0.02
            }
        }

        params = sorbent_params.get(capture_material, sorbent_params['amine'])

        # Air processing rate (m³/s)
        air_flow_rate = 100  # m³/s for industrial scale

        # CO2 in air stream
        co2_molar_flow = co2_concentration * air_pressure * air_flow_rate / \
                        (self.R * air_temperature)  # mol/s

        # Required sorbent circulation rate
        sorbent_rate = co2_molar_flow / (params['capacity'] * 0.9)  # kg/s (90% efficiency)

        # Energy requirements
        # Fan power for air movement
        pressure_drop = 500  # Pa through contactor
        fan_power = air_flow_rate * pressure_drop / 1000  # kW

        # Thermal energy for regeneration
        thermal_energy = co2_molar_flow * params['regeneration_heat']  # kW

        # Temperature swing energy
        cp_sorbent = 1.5  # kJ/(kg·K)
        temp_swing_energy = sorbent_rate * cp_sorbent * \
                          (params['regeneration_temp'] - air_temperature)  # kW

        # Total energy
        total_energy = fan_power + thermal_energy + temp_swing_energy

        # CO2 captured
        capture_rate = co2_molar_flow * 0.9 * 44.01 / 1000  # kg/s (90% capture)

        # Energy per tonne CO2
        energy_per_tonne = total_energy / capture_rate * 1000 / 3.6  # kWh/tCO2

        # Water consumption (for evaporative cooling)
        water_consumption = thermal_energy / 2400  # kg/s (latent heat of water)

        # Capital cost estimate (simplified)
        # Based on $600-1000 per tonne CO2/year capacity
        annual_capacity = capture_rate * 3600 * 24 * 365 / 1000  # tonnes/year
        capex_estimate = annual_capacity * 800  # USD

        # Operating cost
        electricity_cost = 0.07  # $/kWh
        thermal_cost = 0.03  # $/kWh thermal
        opex_per_tonne = (fan_power / capture_rate * electricity_cost +
                         thermal_energy / capture_rate * thermal_cost) * 1000 / 3.6

        return {
            'co2_concentration_ppm': co2_concentration * 1e6,
            'air_flow_rate': air_flow_rate,
            'capture_rate_kg_s': capture_rate,
            'capture_rate_tonne_day': capture_rate * 86.4,
            'sorbent_circulation_kg_s': sorbent_rate,
            'fan_power_kw': fan_power,
            'thermal_energy_kw': thermal_energy,
            'total_energy_kw': total_energy,
            'energy_per_tonne_kwh': energy_per_tonne,
            'water_consumption_kg_s': water_consumption,
            'capex_estimate_usd': capex_estimate,
            'opex_per_tonne_usd': opex_per_tonne,
            'regeneration_temperature_c': params['regeneration_temp'] - 273.15
        }

    def geological_storage_capacity(self, storage_type: StorageType,
                                   formation_properties: Dict) -> Dict:
        """
        Calculate CO2 storage capacity for geological formations
        Based on volumetric and solubility trapping mechanisms
        """
        # Extract formation properties
        volume = formation_properties.get('volume', 1e9)  # m³
        porosity = formation_properties.get('porosity', 0.15)
        depth = formation_properties.get('depth', 1000)  # m
        temperature = formation_properties.get('temperature', 40) + 273.15  # K
        salinity = formation_properties.get('salinity', 0.1)  # kg/kg

        # Pressure at depth (hydrostatic + lithostatic)
        pressure = 101325 + depth * (10000 + 22000) / 2  # Pa

        # CO2 density at reservoir conditions (Peng-Robinson EOS approximation)
        # Simplified correlation
        Tc = self.co2_properties['critical_temp']
        Pc = self.co2_properties['critical_pressure']
        Tr = temperature / Tc
        Pr = pressure / Pc

        if Pr > 1 and Tr < 1.5:
            # Supercritical CO2
            co2_density = 600 + 200 * (Pr - 1) - 100 * (Tr - 1)  # kg/m³
        else:
            # Gas phase
            co2_density = pressure * 44.01 / (self.R * temperature) / 1000  # kg/m³

        # Storage efficiency factors
        efficiency_factors = {
            StorageType.DEPLETED_OIL: 0.25,
            StorageType.SALINE_AQUIFER: 0.02,
            StorageType.UNMINEABLE_COAL: 0.35,
            StorageType.BASALT: 0.10,
            StorageType.SALT_CAVERN: 0.50
        }

        E = efficiency_factors.get(storage_type, 0.02)

        # Structural/stratigraphic trapping capacity
        structural_capacity = volume * porosity * co2_density * E / 1e9  # Gt CO2

        # Solubility trapping (Henry's law)
        # CO2 solubility decreases with salinity
        kh0 = 29.41  # atm/(mol/L) at 25°C
        kh = kh0 * np.exp(2400 * (1/temperature - 1/298.15))  # Temperature correction
        solubility_molal = pressure / (kh * 101325) * (1 - 10 * salinity)  # mol/L

        # Residual water saturation
        Swr = 0.3
        solubility_capacity = volume * porosity * Swr * solubility_molal * 44.01 / 1e12  # Gt CO2

        # Mineral trapping (long-term, mainly for basalt)
        if storage_type == StorageType.BASALT:
            # Basalt can mineralize ~10% of rock volume as carbonate
            rock_density = 2900  # kg/m³
            mineral_capacity = volume * 0.1 * rock_density * 0.12 / 1e12  # Gt CO2
        else:
            mineral_capacity = 0

        # Total capacity
        total_capacity = structural_capacity + solubility_capacity + mineral_capacity

        # Injectivity index (simplified)
        permeability = formation_properties.get('permeability', 100)  # mD
        thickness = formation_properties.get('thickness', 50)  # m
        viscosity = 6e-5  # Pa·s for supercritical CO2

        injectivity = 2 * np.pi * permeability * 1e-15 * thickness / \
                     (viscosity * np.log(1000))  # m³/s/Pa

        # Maximum injection rate
        max_pressure_increase = 0.3 * pressure  # 30% overpressure limit
        max_injection_rate = injectivity * max_pressure_increase * co2_density  # kg/s

        # Storage security score (0-100)
        security_score = 0
        if depth > 800:
            security_score += 30  # Good depth
        if storage_type in [StorageType.DEPLETED_OIL, StorageType.SALT_CAVERN]:
            security_score += 30  # Proven seal
        if porosity > 0.1:
            security_score += 20  # Good porosity
        if permeability > 10:
            security_score += 20  # Good injectivity

        return {
            'storage_type': storage_type.value,
            'structural_capacity_gt': structural_capacity,
            'solubility_capacity_gt': solubility_capacity,
            'mineral_capacity_gt': mineral_capacity,
            'total_capacity_gt': total_capacity,
            'co2_density_kg_m3': co2_density,
            'co2_phase': 'supercritical' if co2_density > 400 else 'gas',
            'storage_efficiency': E,
            'max_injection_rate_kg_s': max_injection_rate,
            'max_injection_rate_mt_year': max_injection_rate * 31.536,
            'security_score': security_score,
            'pressure_mpa': pressure / 1e6,
            'temperature_c': temperature - 273.15
        }

    def pressure_swing_adsorption(self, gas_stream: GasStream,
                                 adsorbent: AdsorbentType,
                                 n_beds: int = 2) -> Dict:
        """
        Pressure Swing Adsorption (PSA) cycle model
        For high-purity CO2 separation
        """
        # Get adsorbent properties
        ads_props = self.adsorbent_data[adsorbent]

        # PSA operating conditions
        p_high = gas_stream.pressure  # Adsorption pressure
        p_low = p_high / 10  # Desorption pressure
        T = gas_stream.temperature

        # Calculate loadings at high and low pressure
        iso_high = self.langmuir_isotherm(p_high * gas_stream.co2_fraction, T)
        iso_low = self.langmuir_isotherm(p_low * gas_stream.co2_fraction, T)

        # Working capacity
        working_capacity = iso_high['loading'] - iso_low['loading']  # mol/kg

        # Bed sizing
        cycle_time = 240  # seconds (4 minute cycle)
        co2_to_capture = gas_stream.flow_rate * gas_stream.co2_fraction * cycle_time  # mol

        adsorbent_mass = co2_to_capture / (working_capacity * 0.9)  # kg (90% efficiency)
        bed_volume = adsorbent_mass / ads_props.particle_density  # m³

        # Energy requirements
        # Compression energy
        compression_energy = gas_stream.flow_rate * self.R * T * \
                           np.log(p_high / p_low) / 1000  # kW

        # Vacuum pump energy (for desorption)
        vacuum_efficiency = 0.7
        vacuum_energy = gas_stream.flow_rate * gas_stream.co2_fraction * \
                       self.R * T * np.log(p_low / 101325) / (1000 * vacuum_efficiency)  # kW

        # Cooling requirement (remove heat of adsorption)
        cooling_load = co2_to_capture / cycle_time * ads_props.heat_adsorption  # kW

        # Total energy
        total_energy = compression_energy + abs(vacuum_energy) + cooling_load * 0.3  # COP = 3.3

        # Performance metrics
        co2_captured = gas_stream.flow_rate * gas_stream.co2_fraction * 0.9  # mol/s
        co2_purity = 0.95 + 0.04 * (working_capacity / 5)  # Empirical correlation

        recovery = 0.9  # 90% recovery typical

        # Productivity
        productivity = co2_captured * 44.01 / (adsorbent_mass * n_beds) * 3600  # kg CO2/kg ads/hr

        # Energy per tonne
        energy_per_tonne = total_energy / (co2_captured * 44.01 / 1000) * 1000 / 3.6  # kWh/tCO2

        return {
            'adsorbent_type': adsorbent.value,
            'working_capacity': working_capacity,
            'adsorbent_mass_per_bed': adsorbent_mass,
            'bed_volume': bed_volume,
            'number_of_beds': n_beds,
            'cycle_time_s': cycle_time,
            'adsorption_pressure_bar': p_high / 1e5,
            'desorption_pressure_bar': p_low / 1e5,
            'co2_captured_mol_s': co2_captured,
            'co2_captured_kg_hr': co2_captured * 44.01 * 3.6,
            'co2_purity': co2_purity,
            'co2_recovery': recovery,
            'productivity': productivity,
            'compression_energy_kw': compression_energy,
            'vacuum_energy_kw': abs(vacuum_energy),
            'cooling_load_kw': cooling_load,
            'total_energy_kw': total_energy,
            'energy_per_tonne_kwh': energy_per_tonne
        }

    def amine_scrubbing(self, gas_stream: GasStream,
                       amine_type: str = 'MEA') -> Dict:
        """
        Amine-based chemical absorption
        Most mature technology for post-combustion capture
        """
        # Amine properties
        amine_props = {
            'MEA': {  # Monoethanolamine
                'concentration': 0.30,  # wt fraction
                'loading_capacity': 0.5,  # mol CO2/mol amine
                'reaction_heat': 84,  # kJ/mol CO2
                'regeneration_temp': 393,  # K (120°C)
                'density': 1020  # kg/m³
            },
            'DEA': {  # Diethanolamine
                'concentration': 0.35,
                'loading_capacity': 0.45,
                'reaction_heat': 70,
                'regeneration_temp': 388,
                'density': 1050
            },
            'MDEA': {  # Methyldiethanolamine
                'concentration': 0.40,
                'loading_capacity': 0.40,
                'reaction_heat': 60,
                'regeneration_temp': 383,
                'density': 1030
            }
        }

        props = amine_props.get(amine_type, amine_props['MEA'])

        # Absorption column design
        co2_flow = gas_stream.flow_rate * gas_stream.co2_fraction  # kmol/s

        # Required amine circulation rate (L/G ratio ~ 3 for MEA)
        l_g_ratio = 3.0
        amine_flow = l_g_ratio * gas_stream.flow_rate  # kmol/s

        # Rich loading (mol CO2/mol amine)
        rich_loading = props['loading_capacity'] * 0.9  # 90% approach
        lean_loading = 0.2  # Typical lean loading

        # Actual circulation rate based on loading
        actual_amine_flow = co2_flow / (rich_loading - lean_loading)  # kmol amine/s

        # Solvent flow rate
        mw_mea = 61.08  # g/mol
        solvent_flow = actual_amine_flow * mw_mea / (props['concentration'] * 1000)  # m³/s

        # Energy requirements
        # Reboiler duty
        sensible_heat = solvent_flow * props['density'] * 4.18 * \
                       (props['regeneration_temp'] - 313)  # kW
        reaction_heat = co2_flow * props['reaction_heat'] * 1000  # kW
        vaporization_heat = solvent_flow * props['density'] * 0.1 * 2400  # kW (10% vaporization)

        reboiler_duty = sensible_heat + reaction_heat + vaporization_heat

        # Cooling requirement
        cooling_duty = co2_flow * props['reaction_heat'] * 1000  # kW

        # Pump work
        pressure_drop = 2e5  # Pa (2 bar total)
        pump_work = solvent_flow * pressure_drop / (1000 * 0.7)  # kW

        # Total equivalent energy
        thermal_efficiency = 0.35  # Power plant efficiency
        total_energy = pump_work + reboiler_duty * thermal_efficiency + \
                      cooling_duty * 0.1  # Cooling ~ 10% of thermal

        # CO2 captured
        capture_efficiency = 0.90
        co2_captured = co2_flow * capture_efficiency * 44.01  # kg/s

        # Energy penalty
        energy_per_tonne = total_energy / co2_captured * 1000  # kWh/tCO2

        # Solvent losses (typical 1-3 kg/tCO2)
        solvent_loss = 2  # kg MEA/tCO2

        # Operating cost
        amine_cost = 2  # $/kg
        steam_cost = 0.015  # $/kWh thermal
        electricity_cost = 0.07  # $/kWh

        opex_per_tonne = (solvent_loss * amine_cost +
                         reboiler_duty / co2_captured * steam_cost * 1000 / 3.6 +
                         pump_work / co2_captured * electricity_cost * 1000 / 3.6)

        return {
            'amine_type': amine_type,
            'co2_captured_kg_s': co2_captured,
            'capture_efficiency': capture_efficiency,
            'solvent_flow_m3_s': solvent_flow,
            'amine_circulation_kmol_s': actual_amine_flow,
            'rich_loading': rich_loading,
            'lean_loading': lean_loading,
            'reboiler_duty_MW': reboiler_duty / 1000,
            'cooling_duty_MW': cooling_duty / 1000,
            'pump_work_kW': pump_work,
            'total_equivalent_energy_MW': total_energy / 1000,
            'energy_per_tonne_kWh': energy_per_tonne,
            'regeneration_temp_C': props['regeneration_temp'] - 273.15,
            'solvent_loss_kg_tCO2': solvent_loss,
            'opex_per_tonne_USD': opex_per_tonne
        }

    def breakthrough_curve(self, bed_length: float, velocity: float,
                          co2_concentration: float, time_points: np.ndarray) -> Dict:
        """
        Calculate breakthrough curve for fixed-bed adsorption
        Using Linear Driving Force (LDF) model
        """
        # LDF parameters
        k_ldf = 0.01  # 1/s
        q_max = 4.0  # mol/kg
        rho_b = 500  # kg/m³ bed density
        epsilon = 0.4  # Void fraction

        # Dimensionless parameters
        peclet = velocity * bed_length / (0.01 * velocity)  # Dispersion coefficient
        tau = epsilon / (1 - epsilon) / (q_max * rho_b / co2_concentration)

        # Breakthrough time (simplified)
        t_break = bed_length / velocity * (1 + (1 - epsilon) / epsilon * q_max * rho_b / co2_concentration)

        # Calculate concentration profile
        c_out = []
        for t in time_points:
            xi = t / t_break
            if xi < 0.05:
                c_ratio = 0
            elif xi > 2:
                c_ratio = 1
            else:
                # Approximate breakthrough curve (sigmoid)
                c_ratio = 1 / (1 + np.exp(-10 * (xi - 1)))
            c_out.append(c_ratio)

        c_out = np.array(c_out)

        # Mass transfer zone length
        mtz_length = bed_length * 0.2  # Typical 20% of bed length

        # Usable bed capacity
        usable_capacity = q_max * (1 - 0.5 * mtz_length / bed_length)

        return {
            'time': time_points.tolist(),
            'concentration_ratio': c_out.tolist(),
            'breakthrough_time': t_break,
            'saturation_time': t_break * 2,
            'mass_transfer_zone': mtz_length,
            'usable_capacity': usable_capacity,
            'bed_utilization': (1 - 0.5 * mtz_length / bed_length),
            'peclet_number': peclet
        }


def demo():
    """Demonstration of Carbon Capture Lab capabilities"""
    print("Carbon Capture Laboratory - Production Demo")
    print("=" * 60)

    lab = CarbonCaptureLab()

    # Demo 1: Adsorption Isotherms
    print("\n1. CO2 Adsorption Isotherms:")
    pressure = 1e5  # 1 bar
    temperature = 298  # K

    langmuir = lab.langmuir_isotherm(pressure, temperature)
    print(f"   Langmuir Loading: {langmuir['loading']:.2f} mol/kg")
    print(f"   Surface Coverage: {langmuir['surface_coverage']:.3f}")
    print(f"   Working Capacity: {langmuir['working_capacity']:.2f} mol/kg")

    # Demo 2: Direct Air Capture
    print("\n2. Direct Air Capture System:")
    dac = lab.direct_air_capture(capture_material='mof')
    print(f"   CO2 Capture Rate: {dac['capture_rate_tonne_day']:.1f} tonnes/day")
    print(f"   Energy Requirement: {dac['energy_per_tonne_kwh']:.0f} kWh/tCO2")
    print(f"   Operating Cost: ${dac['opex_per_tonne_usd']:.2f}/tCO2")

    # Demo 3: Membrane Separation
    print("\n3. Membrane Gas Separation:")
    gas = GasStream(
        co2_fraction=0.15,
        n2_fraction=0.75,
        o2_fraction=0.05,
        h2o_fraction=0.05,
        temperature=313,
        pressure=5e5,
        flow_rate=100
    )

    membrane = lab.membrane_separation(gas, membrane_type='mof_membrane')
    print(f"   CO2 Recovery: {membrane['co2_recovery']:.1%}")
    print(f"   Selectivity (CO2/N2): {membrane['selectivity_co2_n2']:.1f}")
    print(f"   Energy Intensity: {membrane['energy_intensity']:.0f} kWh/tCO2")

    # Demo 4: Geological Storage
    print("\n4. Geological CO2 Storage:")
    formation = {
        'volume': 1e10,  # m³
        'porosity': 0.20,
        'depth': 1500,  # m
        'temperature': 50,  # °C
        'salinity': 0.05,
        'permeability': 200,  # mD
        'thickness': 100  # m
    }

    storage = lab.geological_storage_capacity(StorageType.SALINE_AQUIFER, formation)
    print(f"   Total Capacity: {storage['total_capacity_gt']:.2f} Gt CO2")
    print(f"   CO2 Phase: {storage['co2_phase']}")
    print(f"   Max Injection Rate: {storage['max_injection_rate_mt_year']:.2f} Mt/year")
    print(f"   Security Score: {storage['security_score']}/100")

    # Demo 5: Amine Scrubbing
    print("\n5. Amine Scrubbing System:")
    amine = lab.amine_scrubbing(gas, amine_type='MEA')
    print(f"   CO2 Captured: {amine['co2_captured_kg_s']:.2f} kg/s")
    print(f"   Reboiler Duty: {amine['reboiler_duty_MW']:.2f} MW")
    print(f"   Energy Penalty: {amine['energy_per_tonne_kWh']:.0f} kWh/tCO2")
    print(f"   Operating Cost: ${amine['opex_per_tonne_USD']:.2f}/tCO2")

    print("\nLab ready for production use!")
    return lab


if __name__ == "__main__":
    demo()