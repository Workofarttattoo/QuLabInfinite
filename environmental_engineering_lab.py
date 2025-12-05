"""
Environmental Engineering Laboratory - Production Implementation
Real wastewater treatment, air quality, soil remediation, and life cycle assessment
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import math


class TreatmentProcess(Enum):
    """Wastewater treatment process types"""
    ACTIVATED_SLUDGE = "activated_sludge"
    TRICKLING_FILTER = "trickling_filter"
    MEMBRANE_BIOREACTOR = "membrane_bioreactor"
    ANAEROBIC_DIGESTION = "anaerobic_digestion"


class PollutantType(Enum):
    """Air pollutant types for AQI calculation"""
    PM25 = "pm2.5"
    PM10 = "pm10"
    OZONE = "ozone"
    NO2 = "nitrogen_dioxide"
    SO2 = "sulfur_dioxide"
    CO = "carbon_monoxide"


@dataclass
class WastewaterCharacteristics:
    """Wastewater quality parameters"""
    bod5: float  # 5-day Biological Oxygen Demand (mg/L)
    cod: float  # Chemical Oxygen Demand (mg/L)
    tss: float  # Total Suspended Solids (mg/L)
    tn: float  # Total Nitrogen (mg/L)
    tp: float  # Total Phosphorus (mg/L)
    temperature: float  # Temperature (°C)
    ph: float  # pH
    flow_rate: float  # Flow rate (m³/day)


@dataclass
class SoilContamination:
    """Soil contamination parameters"""
    contaminant: str
    concentration: float  # mg/kg
    depth: float  # meters
    soil_type: str  # clay, sand, silt, loam
    porosity: float  # fraction
    bulk_density: float  # kg/m³


class EnvironmentalEngineeringLab:
    """
    Comprehensive environmental engineering laboratory
    Implements real-world treatment and assessment algorithms
    """

    def __init__(self):
        # Monod kinetics parameters for activated sludge
        self.monod_params = {
            'mu_max': 6.0,  # Maximum specific growth rate (day⁻¹)
            'Ks': 20.0,  # Half-saturation constant (mg/L)
            'Y': 0.4,  # Yield coefficient (mg VSS/mg BOD)
            'kd': 0.06,  # Decay coefficient (day⁻¹)
            'theta_c': 10.0  # Solids retention time (days)
        }

        # AQI breakpoints for different pollutants (EPA standards)
        self.aqi_breakpoints = {
            PollutantType.PM25: [
                (0, 12.0, 0, 50),
                (12.1, 35.4, 51, 100),
                (35.5, 55.4, 101, 150),
                (55.5, 150.4, 151, 200),
                (150.5, 250.4, 201, 300),
                (250.5, 500.4, 301, 500)
            ],
            PollutantType.PM10: [
                (0, 54, 0, 50),
                (55, 154, 51, 100),
                (155, 254, 101, 150),
                (255, 354, 151, 200),
                (355, 424, 201, 300),
                (425, 604, 301, 500)
            ],
            PollutantType.OZONE: [  # 8-hour average (ppm)
                (0.000, 0.054, 0, 50),
                (0.055, 0.070, 51, 100),
                (0.071, 0.085, 101, 150),
                (0.086, 0.105, 151, 200),
                (0.106, 0.200, 201, 300)
            ]
        }

    def activated_sludge_design(self, wastewater: WastewaterCharacteristics,
                                target_effluent_bod: float = 20.0) -> Dict:
        """
        Design activated sludge treatment system using Monod kinetics
        Based on Lawrence-McCarty model
        """
        # Temperature correction for kinetic parameters
        temp_factor = 1.072 ** (wastewater.temperature - 20)
        mu_max = self.monod_params['mu_max'] * temp_factor
        kd = self.monod_params['kd'] * temp_factor

        # Calculate effluent substrate concentration (Monod equation)
        Se = self.monod_params['Ks'] * (1 + kd * self.monod_params['theta_c']) / \
             (self.monod_params['theta_c'] * (mu_max - kd) - 1)

        # Biomass yield calculation
        Y_obs = self.monod_params['Y'] / (1 + kd * self.monod_params['theta_c'])

        # Reactor volume calculation
        So = wastewater.bod5  # Influent BOD
        Q = wastewater.flow_rate  # Flow rate
        X = 3000  # Mixed liquor suspended solids (mg/L) - typical value

        # Calculate hydraulic retention time
        theta = self.monod_params['theta_c'] * Y_obs * (So - Se) / X

        # Reactor volume
        V = Q * theta  # m³

        # Oxygen requirements (kg O2/day)
        # Based on BOD removal and endogenous respiration
        oxygen_required = Q * (So - Se) / 1000 * (1 - 1.42 * Y_obs)

        # Sludge production (kg/day)
        Px = Y_obs * Q * (So - Se) / 1000

        # Nutrient requirements (N and P)
        nitrogen_required = 0.12 * Px  # kg N/day
        phosphorus_required = 0.02 * Px  # kg P/day

        # F/M ratio (Food to Microorganism ratio)
        fm_ratio = (Q * So) / (V * X)

        return {
            'reactor_volume': V,
            'hydraulic_retention_time': theta,
            'effluent_bod': Se,
            'oxygen_required': oxygen_required,
            'sludge_production': Px,
            'nitrogen_required': nitrogen_required,
            'phosphorus_required': phosphorus_required,
            'fm_ratio': fm_ratio,
            'biomass_yield': Y_obs,
            'removal_efficiency': (So - Se) / So * 100
        }

    def calculate_aqi(self, pollutant: PollutantType, concentration: float,
                     averaging_period: str = '24h') -> Dict:
        """
        Calculate Air Quality Index (AQI) using EPA methodology
        """
        if pollutant not in self.aqi_breakpoints:
            return {'error': 'Pollutant type not supported'}

        breakpoints = self.aqi_breakpoints[pollutant]

        # Find appropriate breakpoint range
        for bp_low, bp_high, aqi_low, aqi_high in breakpoints:
            if bp_low <= concentration <= bp_high:
                # Linear interpolation formula for AQI
                aqi = ((aqi_high - aqi_low) / (bp_high - bp_low)) * \
                      (concentration - bp_low) + aqi_low

                # Determine category
                if aqi <= 50:
                    category = "Good"
                    color = "Green"
                elif aqi <= 100:
                    category = "Moderate"
                    color = "Yellow"
                elif aqi <= 150:
                    category = "Unhealthy for Sensitive Groups"
                    color = "Orange"
                elif aqi <= 200:
                    category = "Unhealthy"
                    color = "Red"
                elif aqi <= 300:
                    category = "Very Unhealthy"
                    color = "Purple"
                else:
                    category = "Hazardous"
                    color = "Maroon"

                return {
                    'aqi': round(aqi),
                    'category': category,
                    'color': color,
                    'concentration': concentration,
                    'pollutant': pollutant.value,
                    'health_message': self._get_health_message(aqi)
                }

        return {'aqi': 500, 'category': 'Beyond AQI', 'color': 'Maroon'}

    def soil_remediation_design(self, contamination: SoilContamination,
                                remediation_type: str = 'bioremediation') -> Dict:
        """
        Design soil remediation system based on contaminant characteristics
        Supports bioremediation, soil washing, and thermal desorption
        """
        volume = contamination.depth * 10000  # m³ per hectare
        mass = volume * contamination.bulk_density  # kg
        contaminant_mass = mass * contamination.concentration / 1e6  # kg

        if remediation_type == 'bioremediation':
            # Enhanced bioremediation design
            # First-order decay kinetics
            k_bio = self._get_biodegradation_rate(contamination.contaminant,
                                                  contamination.soil_type)

            # Time to reach target (95% reduction)
            target_reduction = 0.95
            treatment_time = -np.log(1 - target_reduction) / k_bio  # days

            # Nutrient requirements (C:N:P = 100:10:1)
            nitrogen_required = contaminant_mass * 0.1  # kg
            phosphorus_required = contaminant_mass * 0.01  # kg

            # Oxygen requirements for aerobic degradation
            oxygen_required = contaminant_mass * 3.0  # kg (stoichiometric)

            # Air injection rate
            air_injection_rate = oxygen_required / (treatment_time * 0.23)  # m³/day

            return {
                'treatment_type': 'Enhanced Bioremediation',
                'treatment_time': treatment_time,
                'biodegradation_rate': k_bio,
                'nitrogen_required': nitrogen_required,
                'phosphorus_required': phosphorus_required,
                'oxygen_required': oxygen_required,
                'air_injection_rate': air_injection_rate,
                'expected_removal': target_reduction * 100,
                'treated_volume': volume,
                'contaminant_mass_removed': contaminant_mass * target_reduction
            }

        elif remediation_type == 'soil_washing':
            # Soil washing design
            # Based on partition coefficient and wash cycles
            kd = self._get_partition_coefficient(contamination.contaminant,
                                                contamination.soil_type)

            # Washing efficiency per cycle
            wash_efficiency = 1 / (1 + kd * (contamination.bulk_density / 1000))

            # Number of wash cycles for 95% removal
            n_cycles = math.ceil(np.log(0.05) / np.log(1 - wash_efficiency))

            # Water requirements (L/kg soil ratio = 10)
            water_per_cycle = mass * 10  # L
            total_water = water_per_cycle * n_cycles / 1000  # m³

            # Surfactant requirements (0.1% by volume)
            surfactant_required = total_water * 0.001 * 1000  # kg

            return {
                'treatment_type': 'Soil Washing',
                'wash_cycles': n_cycles,
                'efficiency_per_cycle': wash_efficiency * 100,
                'water_required': total_water,
                'surfactant_required': surfactant_required,
                'partition_coefficient': kd,
                'expected_removal': 95,
                'treated_volume': volume,
                'treatment_time': n_cycles * 2  # days (2 days per cycle)
            }

        elif remediation_type == 'thermal_desorption':
            # Thermal desorption design
            # Based on vapor pressure and Henry's law
            vapor_pressure = self._get_vapor_pressure(contamination.contaminant)

            # Temperature required for efficient desorption
            target_temp = 200 + vapor_pressure * 50  # °C

            # Energy requirements
            specific_heat = 0.8  # kJ/kg·°C for soil
            latent_heat = 50  # kJ/kg for volatilization
            energy_required = mass * (specific_heat * (target_temp - 20) + latent_heat)

            # Treatment rate (typical rotary kiln)
            treatment_rate = 10  # tons/hour
            treatment_time = mass / (treatment_rate * 1000 * 24)  # days

            # Off-gas treatment requirements
            gas_flow_rate = treatment_rate * 100  # m³/hour

            return {
                'treatment_type': 'Thermal Desorption',
                'target_temperature': target_temp,
                'energy_required': energy_required / 1e6,  # GJ
                'treatment_time': treatment_time,
                'treatment_rate': treatment_rate,
                'gas_flow_rate': gas_flow_rate,
                'expected_removal': 99,
                'treated_volume': volume,
                'vapor_pressure': vapor_pressure
            }

        return {'error': 'Unknown remediation type'}

    def life_cycle_assessment(self, process_name: str,
                             inputs: Dict[str, float],
                             outputs: Dict[str, float]) -> Dict:
        """
        Perform simplified Life Cycle Assessment (LCA)
        Calculate environmental impacts using ReCiPe methodology
        """
        # Impact categories and characterization factors
        impact_factors = {
            'climate_change': {  # kg CO2-eq per unit
                'electricity': 0.5,  # kWh
                'natural_gas': 2.0,  # m³
                'diesel': 2.68,  # L
                'concrete': 0.1,  # kg
                'steel': 1.8,  # kg
                'chemicals': 2.5  # kg
            },
            'water_depletion': {  # m³ water-eq per unit
                'water': 1.0,  # m³
                'electricity': 0.002,  # kWh
                'chemicals': 0.5  # kg
            },
            'resource_depletion': {  # kg oil-eq per unit
                'electricity': 0.15,  # kWh
                'natural_gas': 0.8,  # m³
                'diesel': 1.0,  # L
                'steel': 0.5  # kg
            }
        }

        # Calculate impacts
        impacts = {}
        for category, factors in impact_factors.items():
            total_impact = 0
            for resource, amount in inputs.items():
                if resource in factors:
                    total_impact += amount * factors[resource]
            impacts[category] = total_impact

        # Calculate eco-efficiency
        # Assuming main output is treated water or cleaned soil
        main_output = outputs.get('treated_volume', 1.0)

        eco_efficiency = {
            'climate_efficiency': main_output / (impacts['climate_change'] + 1e-6),
            'water_efficiency': main_output / (impacts['water_depletion'] + 1e-6),
            'resource_efficiency': main_output / (impacts['resource_depletion'] + 1e-6)
        }

        # Environmental cost (simplified)
        environmental_cost = (
            impacts['climate_change'] * 0.05 +  # $/kg CO2
            impacts['water_depletion'] * 2.0 +  # $/m³
            impacts['resource_depletion'] * 1.0  # $/kg oil-eq
        )

        return {
            'process': process_name,
            'climate_change_impact': impacts['climate_change'],
            'water_depletion_impact': impacts['water_depletion'],
            'resource_depletion_impact': impacts['resource_depletion'],
            'eco_efficiency': eco_efficiency,
            'environmental_cost': environmental_cost,
            'sustainability_score': self._calculate_sustainability_score(impacts)
        }

    def trickling_filter_design(self, wastewater: WastewaterCharacteristics,
                                filter_media: str = 'plastic') -> Dict:
        """
        Design trickling filter for secondary treatment
        Using NRC equations and Eckenfelder model
        """
        # Filter media characteristics
        media_props = {
            'rock': {'specific_surface': 55, 'void_ratio': 0.45, 'n': 0.5},
            'plastic': {'specific_surface': 100, 'void_ratio': 0.95, 'n': 0.4}
        }

        props = media_props.get(filter_media, media_props['plastic'])

        # Design parameters
        So = wastewater.bod5
        Se = 20.0  # Target effluent BOD (mg/L)
        Q = wastewater.flow_rate

        # NRC equation for BOD removal
        # E = 1 / (1 + 0.44 * sqrt(W/VF))
        # where W = BOD load, V = volume, F = recirculation factor

        # Assume recirculation ratio R = 1
        R = 1.0
        F = (1 + R) / (1 + 0.1 * R) ** 2

        # Required efficiency
        E = (So - Se) / So

        # Solve for volume
        # Rearranging: V = W / (F * ((1/E - 1) / 0.44)^2)
        W = Q * So / 1000  # kg BOD/day
        V = W / (F * ((1/E - 1) / 0.44) ** 2)

        # Filter dimensions (assume depth = 2m for plastic, 1.5m for rock)
        depth = 2.0 if filter_media == 'plastic' else 1.5
        area = V / depth
        diameter = np.sqrt(4 * area / np.pi)

        # Hydraulic loading rate
        hydraulic_loading = Q / area  # m³/m²/day

        # Organic loading rate
        organic_loading = W / V  # kg BOD/m³/day

        # Ventilation requirements
        air_required = W * 30  # m³/day (30 m³ air/kg BOD)

        return {
            'filter_volume': V,
            'filter_area': area,
            'filter_depth': depth,
            'filter_diameter': diameter,
            'hydraulic_loading': hydraulic_loading,
            'organic_loading': organic_loading,
            'recirculation_ratio': R,
            'removal_efficiency': E * 100,
            'air_required': air_required,
            'media_type': filter_media,
            'specific_surface': props['specific_surface']
        }

    def membrane_bioreactor_design(self, wastewater: WastewaterCharacteristics) -> Dict:
        """
        Design Membrane Bioreactor (MBR) system
        Combines biological treatment with membrane filtration
        """
        # MBR operates at higher MLSS than conventional activated sludge
        X = 10000  # mg/L MLSS (typical for MBR)

        # Use similar kinetics but with better effluent quality
        So = wastewater.bod5
        Se = 5.0  # Very low effluent BOD achievable with MBR
        Q = wastewater.flow_rate

        # HRT calculation (shorter than conventional due to high MLSS)
        theta = 4.0  # hours
        V = Q * theta / 24  # m³

        # Membrane area calculation
        # Typical flux rate for MBR
        flux_rate = 20  # L/m²/h
        membrane_area = Q * 1000 / (flux_rate * 24)  # m²

        # Number of membrane modules (assume 40 m²/module)
        n_modules = math.ceil(membrane_area / 40)

        # Trans-membrane pressure
        tmp = 10 + 0.001 * X  # kPa

        # Energy requirements
        # Aeration energy
        aeration_energy = 1.2 * Q  # kWh/day
        # Membrane scouring
        scouring_energy = 0.3 * membrane_area  # kWh/day
        # Permeate pumping
        pumping_energy = Q * tmp / 3600  # kWh/day

        total_energy = aeration_energy + scouring_energy + pumping_energy

        # Chemical cleaning requirements
        cleaning_frequency = 30  # days
        chemical_per_clean = membrane_area * 0.5  # kg

        return {
            'reactor_volume': V,
            'membrane_area': membrane_area,
            'number_modules': n_modules,
            'mlss_concentration': X,
            'trans_membrane_pressure': tmp,
            'flux_rate': flux_rate,
            'total_energy': total_energy,
            'energy_per_m3': total_energy / Q,
            'cleaning_frequency': cleaning_frequency,
            'chemical_usage': chemical_per_clean * 365 / cleaning_frequency,
            'effluent_quality': {
                'bod': Se,
                'tss': 1.0,  # mg/L (excellent TSS removal)
                'turbidity': 0.1  # NTU
            }
        }

    def wetland_treatment_design(self, wastewater: WastewaterCharacteristics,
                                 wetland_type: str = 'subsurface') -> Dict:
        """
        Design constructed wetland for wastewater treatment
        Based on first-order kinetics and EPA design manual
        """
        # Temperature adjustment
        k20 = 1.104 if wetland_type == 'subsurface' else 0.678  # day⁻¹
        theta = 1.06
        k = k20 * theta ** (wastewater.temperature - 20)

        # Design for BOD removal
        Co = wastewater.bod5
        Ce = 20.0  # Target effluent BOD
        Q = wastewater.flow_rate

        # Aspect ratio and depth
        depth = 0.6 if wetland_type == 'subsurface' else 0.3  # m
        porosity = 0.4 if wetland_type == 'subsurface' else 1.0

        # Required residence time
        t = -np.log(Ce / Co) / k  # days

        # Volume and area
        V = Q * t  # m³
        As = V / (depth * porosity)  # m²

        # Length and width (assume L:W = 3:1)
        length = np.sqrt(3 * As)
        width = As / length

        # Hydraulic loading rate
        hlr = Q / As  # m/day

        # Plant density (plants/m²)
        plant_density = 4 if wetland_type == 'subsurface' else 6
        total_plants = As * plant_density

        # Expected nutrient removal
        n_removal = 40 if wetland_type == 'subsurface' else 30  # %
        p_removal = 30 if wetland_type == 'subsurface' else 20  # %

        return {
            'wetland_type': wetland_type,
            'surface_area': As,
            'length': length,
            'width': width,
            'depth': depth,
            'residence_time': t,
            'hydraulic_loading': hlr,
            'plant_density': plant_density,
            'total_plants': total_plants,
            'bod_removal': (Co - Ce) / Co * 100,
            'nitrogen_removal': n_removal,
            'phosphorus_removal': p_removal,
            'volume': V,
            'porosity': porosity
        }

    def anaerobic_digestion_design(self, organic_waste: Dict) -> Dict:
        """
        Design anaerobic digestion system for organic waste treatment
        Produces biogas and digestate
        """
        # Waste characteristics
        volatile_solids = organic_waste.get('volatile_solids', 100)  # kg/day
        cod = organic_waste.get('cod', 1500)  # mg/L

        # Design parameters
        temperature = 35  # °C (mesophilic)
        hrt = 20  # days (hydraulic retention time)
        organic_loading = 3.0  # kg VS/m³/day

        # Digester volume
        V = volatile_solids / organic_loading  # m³

        # Biogas production
        # Theoretical yield: 0.35 m³ CH4/kg COD removed
        cod_removed = cod * 0.9  # 90% COD removal
        methane_yield = 0.35  # m³ CH4/kg COD
        biogas_production = volatile_solids * cod_removed / 1000 * methane_yield

        # Biogas composition
        ch4_content = 65  # %
        co2_content = 30  # %

        # Energy production
        methane_energy = 35.8  # MJ/m³
        daily_energy = biogas_production * (ch4_content / 100) * methane_energy

        # Heat requirements for maintaining temperature
        heat_required = V * 4.18 * (temperature - 15) * 0.1  # MJ/day

        # Net energy
        net_energy = daily_energy - heat_required

        # Digestate production
        digestate = volatile_solids * 0.4  # kg/day (60% VS reduction)

        return {
            'digester_volume': V,
            'hydraulic_retention_time': hrt,
            'organic_loading_rate': organic_loading,
            'biogas_production': biogas_production,
            'methane_content': ch4_content,
            'co2_content': co2_content,
            'daily_energy_production': daily_energy,
            'heat_required': heat_required,
            'net_energy': net_energy,
            'digestate_production': digestate,
            'cod_removal': 90,
            'vs_reduction': 60
        }

    def _get_biodegradation_rate(self, contaminant: str, soil_type: str) -> float:
        """Get first-order biodegradation rate constant"""
        # Typical biodegradation rates (day⁻¹)
        rates = {
            'petroleum': {'sand': 0.05, 'clay': 0.02, 'silt': 0.03, 'loam': 0.04},
            'btex': {'sand': 0.1, 'clay': 0.04, 'silt': 0.06, 'loam': 0.08},
            'pah': {'sand': 0.02, 'clay': 0.008, 'silt': 0.012, 'loam': 0.015}
        }
        return rates.get(contaminant.lower(), {}).get(soil_type.lower(), 0.01)

    def _get_partition_coefficient(self, contaminant: str, soil_type: str) -> float:
        """Get soil-water partition coefficient"""
        # Typical Kd values (L/kg)
        kd_values = {
            'petroleum': {'sand': 10, 'clay': 100, 'silt': 50, 'loam': 30},
            'btex': {'sand': 2, 'clay': 20, 'silt': 10, 'loam': 6},
            'heavy_metals': {'sand': 50, 'clay': 500, 'silt': 250, 'loam': 150}
        }
        return kd_values.get(contaminant.lower(), {}).get(soil_type.lower(), 10)

    def _get_vapor_pressure(self, contaminant: str) -> float:
        """Get vapor pressure at 20°C (kPa)"""
        vapor_pressures = {
            'benzene': 10.0,
            'toluene': 2.9,
            'petroleum': 1.0,
            'tce': 7.8,
            'pce': 1.9
        }
        return vapor_pressures.get(contaminant.lower(), 1.0)

    def _get_health_message(self, aqi: float) -> str:
        """Get health message based on AQI value"""
        if aqi <= 50:
            return "Air quality is satisfactory"
        elif aqi <= 100:
            return "Unusually sensitive people should consider limiting prolonged outdoor exertion"
        elif aqi <= 150:
            return "Children, elderly, and people with heart or lung disease should limit prolonged outdoor exertion"
        elif aqi <= 200:
            return "Everyone should avoid prolonged outdoor exertion"
        elif aqi <= 300:
            return "Everyone should avoid all outdoor exertion"
        else:
            return "Emergency conditions: everyone should avoid any outdoor activity"

    def _calculate_sustainability_score(self, impacts: Dict) -> float:
        """Calculate overall sustainability score (0-100)"""
        # Normalize impacts and calculate weighted score
        max_impacts = {
            'climate_change': 1000,
            'water_depletion': 100,
            'resource_depletion': 500
        }

        score = 100
        for category, impact in impacts.items():
            if category in max_impacts:
                normalized = min(impact / max_impacts[category], 1.0)
                score -= normalized * 33.33

        return max(0, score)


def demo():
    """Demonstration of Environmental Engineering Lab capabilities"""
    print("Environmental Engineering Laboratory - Production Demo")
    print("=" * 60)

    lab = EnvironmentalEngineeringLab()

    # Demo 1: Activated Sludge Design
    print("\n1. Activated Sludge Treatment Design:")
    wastewater = WastewaterCharacteristics(
        bod5=250,  # mg/L
        cod=500,  # mg/L
        tss=200,  # mg/L
        tn=40,  # mg/L
        tp=8,  # mg/L
        temperature=20,  # °C
        ph=7.2,
        flow_rate=10000  # m³/day
    )

    design = lab.activated_sludge_design(wastewater)
    print(f"   Reactor Volume: {design['reactor_volume']:.1f} m³")
    print(f"   HRT: {design['hydraulic_retention_time']:.2f} days")
    print(f"   Oxygen Required: {design['oxygen_required']:.1f} kg/day")
    print(f"   BOD Removal: {design['removal_efficiency']:.1f}%")

    # Demo 2: Air Quality Index
    print("\n2. Air Quality Index Calculation:")
    aqi_result = lab.calculate_aqi(PollutantType.PM25, 45.5)
    print(f"   PM2.5 Concentration: {aqi_result['concentration']} µg/m³")
    print(f"   AQI: {aqi_result['aqi']}")
    print(f"   Category: {aqi_result['category']}")
    print(f"   Color Code: {aqi_result['color']}")

    # Demo 3: Soil Remediation
    print("\n3. Soil Remediation Design:")
    contamination = SoilContamination(
        contaminant='petroleum',
        concentration=5000,  # mg/kg
        depth=2.0,  # m
        soil_type='sand',
        porosity=0.35,
        bulk_density=1600  # kg/m³
    )

    remediation = lab.soil_remediation_design(contamination, 'bioremediation')
    print(f"   Treatment Type: {remediation['treatment_type']}")
    print(f"   Treatment Time: {remediation['treatment_time']:.1f} days")
    print(f"   Oxygen Required: {remediation['oxygen_required']:.1f} kg")
    print(f"   Expected Removal: {remediation['expected_removal']:.1f}%")

    # Demo 4: MBR Design
    print("\n4. Membrane Bioreactor Design:")
    mbr = lab.membrane_bioreactor_design(wastewater)
    print(f"   Membrane Area: {mbr['membrane_area']:.1f} m²")
    print(f"   Number of Modules: {mbr['number_modules']}")
    print(f"   Energy Requirement: {mbr['total_energy']:.1f} kWh/day")
    print(f"   Effluent BOD: {mbr['effluent_quality']['bod']} mg/L")

    # Demo 5: Constructed Wetland
    print("\n5. Constructed Wetland Design:")
    wetland = lab.wetland_treatment_design(wastewater, 'subsurface')
    print(f"   Surface Area: {wetland['surface_area']:.1f} m²")
    print(f"   Dimensions: {wetland['length']:.1f}m × {wetland['width']:.1f}m")
    print(f"   Residence Time: {wetland['residence_time']:.1f} days")
    print(f"   BOD Removal: {wetland['bod_removal']:.1f}%")

    print("\nLab ready for production use!")
    return lab


if __name__ == "__main__":
    demo()