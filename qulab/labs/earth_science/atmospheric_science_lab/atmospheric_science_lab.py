# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Atmospheric Science Laboratory - Climate and Weather Modeling
Implements scientific atmospheric models based on NIST/NOAA standards
"""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import json

# NIST/NOAA Physical Constants
R_SPECIFIC_DRY_AIR = 287.058  # J/(kg·K) - specific gas constant for dry air
CP_AIR = 1005.0  # J/(kg·K) - specific heat capacity at constant pressure
CV_AIR = 718.0  # J/(kg·K) - specific heat capacity at constant volume
GAMMA_AIR = CP_AIR / CV_AIR  # 1.4 - heat capacity ratio
G_GRAVITY = 9.80665  # m/s² - standard gravity
STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m²·K⁴)
SOLAR_CONSTANT = 1361.0  # W/m² - total solar irradiance
EARTH_RADIUS = 6371000.0  # m
AVOGADRO = 6.02214076e23  # mol⁻¹
BOLTZMANN = 1.380649e-23  # J/K

# Greenhouse Gas Properties (IPCC AR6 data)
GHG_PROPERTIES = {
    'CO2': {
        'molecular_weight': 44.01,  # g/mol
        'radiative_efficiency': 1.37e-5,  # W/(m²·ppb)
        'lifetime_years': 'variable',  # Complex carbon cycle
        'pre_industrial_ppm': 280.0,
        'current_ppm': 420.0  # 2024 average
    },
    'CH4': {
        'molecular_weight': 16.04,
        'radiative_efficiency': 3.63e-4,  # W/(m²·ppb)
        'lifetime_years': 12.4,
        'pre_industrial_ppb': 722.0,
        'current_ppb': 1923.0
    },
    'N2O': {
        'molecular_weight': 44.013,
        'radiative_efficiency': 3.00e-3,  # W/(m²·ppb)
        'lifetime_years': 109.0,
        'pre_industrial_ppb': 270.0,
        'current_ppb': 336.0
    },
    'O3': {
        'molecular_weight': 48.00,
        'radiative_efficiency': 0.042e-3,  # W/(m²·ppb)
        'lifetime_years': 0.05,  # ~days in troposphere
        'typical_ppb': 30.0
    }
}

# Air Quality Index (AQI) Breakpoints (EPA standard)
AQI_BREAKPOINTS = {
    'PM2.5': [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500)
    ],
    'PM10': [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 604, 301, 500)
    ],
    'O3': [  # 8-hour average, ppb
        (0, 54, 0, 50),
        (55, 70, 51, 100),
        (71, 85, 101, 150),
        (86, 105, 151, 200),
        (106, 200, 201, 300)
    ],
    'NO2': [  # ppb
        (0, 53, 0, 50),
        (54, 100, 51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
        (1250, 2049, 301, 500)
    ]
}


class ClimateModel:
    """
    Energy Balance Climate Model based on radiative forcing
    Implements simplified climate dynamics following IPCC methodologies
    """

    def __init__(self):
        self.climate_sensitivity = 3.0  # °C per doubling of CO2 (IPCC central estimate)
        self.ocean_heat_capacity = 4.0e8  # J/(m²·K) - effective mixed layer
        self.feedback_parameter = 1.2  # W/(m²·K) - climate feedback

    def calculate_radiative_forcing(self, co2_ppm: float, ch4_ppb: float,
                                    n2o_ppb: float) -> Dict[str, float]:
        """
        Calculate radiative forcing from greenhouse gas concentrations
        Uses IPCC AR6 simplified expressions

        Args:
            co2_ppm: CO2 concentration in ppm
            ch4_ppb: CH4 concentration in ppb
            n2o_ppb: N2O concentration in ppb

        Returns:
            Dictionary with forcing components in W/m²
        """
        # Pre-industrial baselines
        co2_0 = GHG_PROPERTIES['CO2']['pre_industrial_ppm']
        ch4_0 = GHG_PROPERTIES['CH4']['pre_industrial_ppb']
        n2o_0 = GHG_PROPERTIES['N2O']['pre_industrial_ppb']

        # CO2 forcing: simplified logarithmic form
        rf_co2 = 5.35 * np.log(co2_ppm / co2_0)

        # CH4 forcing: simplified expression with N2O interaction
        m = ch4_ppb
        m0 = ch4_0
        n = n2o_ppb
        n0 = n2o_0

        rf_ch4 = 0.036 * (np.sqrt(m) - np.sqrt(m0)) - (
            self._f(m, n0) - self._f(m0, n0)
        )

        # N2O forcing: simplified expression with CH4 interaction
        rf_n2o = 0.12 * (np.sqrt(n) - np.sqrt(n0)) - (
            self._f(m0, n) - self._f(m0, n0)
        )

        # Total anthropogenic forcing
        rf_total = rf_co2 + rf_ch4 + rf_n2o

        return {
            'co2_forcing_wm2': rf_co2,
            'ch4_forcing_wm2': rf_ch4,
            'n2o_forcing_wm2': rf_n2o,
            'total_forcing_wm2': rf_total
        }

    def _f(self, m: float, n: float) -> float:
        """Overlap function for CH4-N2O interaction"""
        return 0.47 * np.log(1 + 2.01e-5 * (m * n)**0.75 +
                            5.31e-15 * m * (m * n)**1.52)

    def project_temperature_change(self, forcing_wm2: float,
                                   years: int = 100) -> Dict[str, any]:
        """
        Project temperature change using energy balance model

        Args:
            forcing_wm2: Radiative forcing in W/m²
            years: Projection timespan

        Returns:
            Dictionary with temperature projection data
        """
        # Time array
        t = np.linspace(0, years, years * 12)  # Monthly resolution

        # Simple energy balance: C * dT/dt = F - λT
        # where C is heat capacity, F is forcing, λ is feedback parameter
        def temperature_ode(T, t):
            return (forcing_wm2 - self.feedback_parameter * T) / self.ocean_heat_capacity * (365.25 * 86400)

        # Solve ODE
        T0 = 0.0  # Start at pre-industrial
        temperature = odeint(temperature_ode, T0, t)

        # Equilibrium temperature (when dT/dt = 0)
        T_eq = forcing_wm2 / self.feedback_parameter

        return {
            'time_years': t.tolist(),
            'temperature_change_celsius': temperature.flatten().tolist(),
            'equilibrium_temperature_celsius': float(T_eq),
            'e_folding_time_years': float(self.ocean_heat_capacity /
                                         (self.feedback_parameter * 365.25 * 86400))
        }

    def calculate_climate_sensitivity(self, forcing_2x_co2: float = 3.71) -> float:
        """
        Calculate equilibrium climate sensitivity

        Args:
            forcing_2x_co2: Forcing from doubling CO2 (W/m², IPCC value: 3.71)

        Returns:
            Temperature change for 2×CO2 in °C
        """
        return forcing_2x_co2 / self.feedback_parameter


class AirQualityAnalyzer:
    """
    Air Quality Index (AQI) calculation following EPA standards
    Analyzes pollutant concentrations and health impacts
    """

    def calculate_aqi(self, pollutant: str, concentration: float) -> Dict[str, any]:
        """
        Calculate AQI for a given pollutant concentration

        Args:
            pollutant: Pollutant name (PM2.5, PM10, O3, NO2)
            concentration: Measured concentration in appropriate units

        Returns:
            Dictionary with AQI value and health category
        """
        if pollutant not in AQI_BREAKPOINTS:
            raise ValueError(f"Unknown pollutant: {pollutant}")

        breakpoints = AQI_BREAKPOINTS[pollutant]

        # Find appropriate breakpoint
        for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
            if bp_lo <= concentration <= bp_hi:
                # Linear interpolation
                aqi = ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (concentration - bp_lo) + aqi_lo
                category, health_message = self._get_health_category(aqi)

                return {
                    'pollutant': pollutant,
                    'concentration': concentration,
                    'aqi_value': int(round(aqi)),
                    'category': category,
                    'health_message': health_message,
                    'color_code': self._get_color_code(aqi)
                }

        # If concentration exceeds all breakpoints
        aqi = 500  # Maximum AQI
        return {
            'pollutant': pollutant,
            'concentration': concentration,
            'aqi_value': aqi,
            'category': 'Hazardous',
            'health_message': 'Emergency conditions: everyone more likely to be affected',
            'color_code': '#7E0023'
        }

    def _get_health_category(self, aqi: float) -> Tuple[str, str]:
        """Map AQI to health category"""
        if aqi <= 50:
            return 'Good', 'Air quality is satisfactory'
        elif aqi <= 100:
            return 'Moderate', 'Acceptable for most, unusually sensitive may be affected'
        elif aqi <= 150:
            return 'Unhealthy for Sensitive Groups', 'Sensitive groups may experience health effects'
        elif aqi <= 200:
            return 'Unhealthy', 'Everyone may begin to experience health effects'
        elif aqi <= 300:
            return 'Very Unhealthy', 'Health alert: everyone may experience more serious health effects'
        else:
            return 'Hazardous', 'Health warnings of emergency conditions'

    def _get_color_code(self, aqi: float) -> str:
        """Map AQI to EPA color code"""
        if aqi <= 50:
            return '#00E400'  # Green
        elif aqi <= 100:
            return '#FFFF00'  # Yellow
        elif aqi <= 150:
            return '#FF7E00'  # Orange
        elif aqi <= 200:
            return '#FF0000'  # Red
        elif aqi <= 300:
            return '#8F3F97'  # Purple
        else:
            return '#7E0023'  # Maroon

    def analyze_multi_pollutant(self, concentrations: Dict[str, float]) -> Dict[str, any]:
        """
        Analyze multiple pollutants and determine overall AQI

        Args:
            concentrations: Dictionary of pollutant:concentration pairs

        Returns:
            Overall AQI analysis with dominant pollutant
        """
        results = {}
        max_aqi = 0
        dominant_pollutant = None

        for pollutant, conc in concentrations.items():
            result = self.calculate_aqi(pollutant, conc)
            results[pollutant] = result

            if result['aqi_value'] > max_aqi:
                max_aqi = result['aqi_value']
                dominant_pollutant = pollutant

        return {
            'overall_aqi': max_aqi,
            'dominant_pollutant': dominant_pollutant,
            'category': results[dominant_pollutant]['category'],
            'health_message': results[dominant_pollutant]['health_message'],
            'individual_pollutants': results
        }


class GreenhouseGasSimulator:
    """
    Simulates greenhouse gas atmospheric dynamics
    Models emissions, atmospheric lifetime, and radiative effects
    """

    def simulate_co2_cycle(self, emissions_gt_per_year: float,
                          years: int = 100) -> Dict[str, any]:
        """
        Simulate CO2 concentration evolution with carbon cycle
        Uses multi-box model with ocean and terrestrial uptake

        Args:
            emissions_gt_per_year: Annual CO2 emissions in gigatonnes
            years: Simulation duration

        Returns:
            Time series of atmospheric CO2 concentration
        """
        # Initial conditions
        co2_ppm_0 = 420.0  # Current atmospheric CO2

        # Conversion: 1 ppm CO2 ≈ 2.124 GtC (gigatonnes carbon)
        ppm_per_gtc = 1.0 / 2.124

        # Carbon cycle response functions (Joos et al. 2013)
        # Fraction remaining in atmosphere after emission
        def airborne_fraction(t):
            a = [0.2173, 0.2240, 0.2824, 0.2763]
            tau = [394.4, 36.54, 4.304, 1e6]  # years (last is permanent)
            return sum(a_i * np.exp(-t / tau_i) if tau_i < 1e5 else a_i
                      for a_i, tau_i in zip(a, tau))

        # Time array
        t = np.arange(0, years + 1)

        # Calculate concentration
        co2_ppm = np.zeros(len(t))
        co2_ppm[0] = co2_ppm_0

        for i in range(1, len(t)):
            # Cumulative emissions effect
            cum_effect = 0
            for j in range(i):
                age = i - j
                cum_effect += emissions_gt_per_year * ppm_per_gtc * airborne_fraction(age)

            co2_ppm[i] = co2_ppm_0 + cum_effect

        return {
            'time_years': t.tolist(),
            'co2_ppm': co2_ppm.tolist(),
            'final_concentration_ppm': float(co2_ppm[-1]),
            'total_emissions_gt': float(emissions_gt_per_year * years),
            'atmospheric_retention_percent': float(
                (co2_ppm[-1] - co2_ppm_0) / (emissions_gt_per_year * years * ppm_per_gtc) * 100
            )
        }

    def calculate_global_warming_potential(self, gas: str,
                                          time_horizon: int = 100) -> Dict[str, float]:
        """
        Calculate Global Warming Potential (GWP) relative to CO2

        Args:
            gas: Greenhouse gas name (CH4, N2O, etc.)
            time_horizon: Integration period in years

        Returns:
            GWP value and components
        """
        if gas not in GHG_PROPERTIES:
            raise ValueError(f"Unknown gas: {gas}")

        props = GHG_PROPERTIES[gas]

        if gas == 'CO2':
            return {'gwp': 1.0, 'time_horizon_years': time_horizon}

        # Simplified GWP calculation
        # Actual GWP requires integration of radiative forcing over time

        # IPCC AR6 GWP values (100-year time horizon)
        gwp_100 = {
            'CH4': 27.9,  # Fossil CH4, excluding climate-carbon feedbacks
            'N2O': 273.0,
            'O3': None  # Short-lived, no standard GWP
        }

        if gas in gwp_100 and time_horizon == 100:
            return {
                'gas': gas,
                'gwp_100': gwp_100[gas],
                'time_horizon_years': 100,
                'radiative_efficiency_ratio': props['radiative_efficiency'] / GHG_PROPERTIES['CO2']['radiative_efficiency']
            }

        # For other time horizons, use simplified approximation
        lifetime = props.get('lifetime_years')
        if lifetime == 'variable' or lifetime is None:
            return {'error': 'GWP calculation not available for this gas/time horizon'}

        # Simplified: GWP ≈ (RE_x / RE_CO2) × (τ_x / τ_integral)
        # This is approximate; actual calculation requires full integration
        return {
            'gas': gas,
            'approximate_gwp': float(
                props['radiative_efficiency'] / GHG_PROPERTIES['CO2']['radiative_efficiency'] *
                min(lifetime, time_horizon) / time_horizon
            ),
            'time_horizon_years': time_horizon,
            'note': 'Simplified approximation; use IPCC values for policy'
        }


class WeatherPredictor:
    """
    Simplified weather prediction using atmospheric dynamics
    Implements basic numerical weather prediction concepts
    """

    def __init__(self):
        self.lapse_rate = -0.0065  # K/m - standard atmospheric lapse rate

    def calculate_atmospheric_pressure(self, altitude_m: float,
                                      surface_temp_k: float = 288.15,
                                      surface_pressure_pa: float = 101325) -> Dict[str, float]:
        """
        Calculate atmospheric pressure at altitude using barometric formula

        Args:
            altitude_m: Altitude in meters
            surface_temp_k: Surface temperature in Kelvin
            surface_pressure_pa: Surface pressure in Pascals

        Returns:
            Pressure and related atmospheric parameters
        """
        # Barometric formula for troposphere (h < 11 km)
        if altitude_m < 11000:
            temperature_k = surface_temp_k + self.lapse_rate * altitude_m
            pressure_pa = surface_pressure_pa * (temperature_k / surface_temp_k) ** (
                -G_GRAVITY / (R_SPECIFIC_DRY_AIR * self.lapse_rate)
            )
        else:
            # Isothermal stratosphere approximation
            temp_11km = surface_temp_k + self.lapse_rate * 11000
            p_11km = surface_pressure_pa * (temp_11km / surface_temp_k) ** (
                -G_GRAVITY / (R_SPECIFIC_DRY_AIR * self.lapse_rate)
            )
            temperature_k = temp_11km
            pressure_pa = p_11km * np.exp(-G_GRAVITY * (altitude_m - 11000) /
                                          (R_SPECIFIC_DRY_AIR * temp_11km))

        # Air density
        density_kg_m3 = pressure_pa / (R_SPECIFIC_DRY_AIR * temperature_k)

        return {
            'altitude_m': altitude_m,
            'pressure_pa': pressure_pa,
            'pressure_hpa': pressure_pa / 100,
            'temperature_k': temperature_k,
            'temperature_c': temperature_k - 273.15,
            'air_density_kg_m3': density_kg_m3
        }

    def estimate_dewpoint(self, temp_c: float, relative_humidity: float) -> Dict[str, float]:
        """
        Calculate dewpoint temperature using Magnus formula

        Args:
            temp_c: Air temperature in Celsius
            relative_humidity: Relative humidity as fraction (0-1)

        Returns:
            Dewpoint and related moisture parameters
        """
        # Magnus formula constants
        a = 17.27
        b = 237.7  # °C

        # Vapor pressure
        alpha = ((a * temp_c) / (b + temp_c)) + np.log(relative_humidity)
        dewpoint_c = (b * alpha) / (a - alpha)

        # Saturation vapor pressure (Tetens formula)
        es_hpa = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
        e_hpa = relative_humidity * es_hpa

        # Specific humidity
        pressure_hpa = 1013.25  # Assume sea level
        specific_humidity = 0.622 * e_hpa / (pressure_hpa - 0.378 * e_hpa)

        return {
            'temperature_c': temp_c,
            'relative_humidity_percent': relative_humidity * 100,
            'dewpoint_c': dewpoint_c,
            'vapor_pressure_hpa': e_hpa,
            'saturation_vapor_pressure_hpa': es_hpa,
            'specific_humidity_kg_kg': specific_humidity
        }

    def predict_convective_available_potential_energy(self,
                                                      surface_temp_c: float,
                                                      surface_dewpoint_c: float,
                                                      surface_pressure_hpa: float = 1013.25) -> Dict[str, any]:
        """
        Calculate CAPE - indicator of thunderstorm potential

        Args:
            surface_temp_c: Surface air temperature
            surface_dewpoint_c: Surface dewpoint
            surface_pressure_hpa: Surface pressure

        Returns:
            CAPE value and storm potential assessment
        """
        # Simplified CAPE calculation
        # Full calculation requires vertical atmospheric profile

        # Lifting Condensation Level (LCL) using approximate formula
        lcl_height_m = 125 * (surface_temp_c - surface_dewpoint_c)

        # Parcel theory: calculate buoyancy energy
        # This is simplified; actual CAPE requires integration over altitude

        temp_k = surface_temp_c + 273.15
        dewpoint_k = surface_dewpoint_c + 273.15

        # Estimate CAPE using simplified formula
        # CAPE ≈ g × (ΔT/T) × Δz where ΔT is temperature excess

        # Assume parcel rises to ~10 km with average 3°C excess
        temp_excess_estimate = max(0, (surface_temp_c - surface_dewpoint_c) * 0.3)
        cape_estimate = G_GRAVITY * (temp_excess_estimate / temp_k) * 10000  # J/kg

        # Classify storm potential
        if cape_estimate < 1000:
            potential = 'Low - unlikely severe weather'
        elif cape_estimate < 2500:
            potential = 'Moderate - isolated strong storms possible'
        elif cape_estimate < 4000:
            potential = 'High - severe storms likely'
        else:
            potential = 'Extreme - violent storms possible'

        return {
            'cape_j_kg': cape_estimate,
            'lcl_height_m': lcl_height_m,
            'storm_potential': potential,
            'surface_temp_c': surface_temp_c,
            'surface_dewpoint_c': surface_dewpoint_c,
            'note': 'Simplified calculation; use radiosonde data for accuracy'
        }


class AtmosphericScienceLab:
    """
    Main laboratory interface for atmospheric science simulations
    Integrates climate modeling, air quality, and weather prediction
    """

    def __init__(self):
        self.climate_model = ClimateModel()
        self.air_quality = AirQualityAnalyzer()
        self.ghg_simulator = GreenhouseGasSimulator()
        self.weather = WeatherPredictor()

    def run_comprehensive_climate_analysis(self,
                                          co2_ppm: float = 420,
                                          ch4_ppb: float = 1923,
                                          n2o_ppb: float = 336,
                                          projection_years: int = 100) -> Dict[str, any]:
        """
        Run complete climate analysis with current greenhouse gas levels

        Args:
            co2_ppm: CO2 concentration
            ch4_ppb: CH4 concentration
            n2o_ppb: N2O concentration
            projection_years: Years to project

        Returns:
            Comprehensive climate analysis results
        """
        # Calculate forcing
        forcing = self.climate_model.calculate_radiative_forcing(co2_ppm, ch4_ppb, n2o_ppb)

        # Project temperature
        projection = self.climate_model.project_temperature_change(
            forcing['total_forcing_wm2'], projection_years
        )

        # Calculate climate sensitivity
        sensitivity = self.climate_model.calculate_climate_sensitivity()

        return {
            'current_concentrations': {
                'co2_ppm': co2_ppm,
                'ch4_ppb': ch4_ppb,
                'n2o_ppb': n2o_ppb
            },
            'radiative_forcing': forcing,
            'temperature_projection': projection,
            'climate_sensitivity_celsius': sensitivity
        }

    def run_air_quality_assessment(self, pollutants: Dict[str, float]) -> Dict[str, any]:
        """
        Comprehensive air quality assessment

        Args:
            pollutants: Dictionary of pollutant concentrations

        Returns:
            Air quality analysis with health recommendations
        """
        return self.air_quality.analyze_multi_pollutant(pollutants)

    def run_greenhouse_gas_scenario(self, emissions_gt_per_year: float,
                                   years: int = 100) -> Dict[str, any]:
        """
        Simulate greenhouse gas emission scenario

        Args:
            emissions_gt_per_year: Annual emissions
            years: Simulation duration

        Returns:
            CO2 cycle simulation and climate impact
        """
        co2_projection = self.ghg_simulator.simulate_co2_cycle(emissions_gt_per_year, years)

        # Calculate climate impact at end of period
        final_co2 = co2_projection['co2_ppm'][-1]
        forcing = self.climate_model.calculate_radiative_forcing(final_co2, 1923, 336)

        return {
            'emissions_scenario': {
                'annual_emissions_gt': emissions_gt_per_year,
                'years': years
            },
            'co2_projection': co2_projection,
            'climate_impact': forcing
        }

    def run_weather_forecast_analysis(self,
                                     altitude_m: float,
                                     surface_temp_c: float,
                                     relative_humidity: float,
                                     surface_pressure_hpa: float = 1013.25) -> Dict[str, any]:
        """
        Comprehensive weather analysis at specified altitude

        Args:
            altitude_m: Altitude for analysis
            surface_temp_c: Surface temperature
            relative_humidity: Relative humidity (0-1)
            surface_pressure_hpa: Surface pressure

        Returns:
            Complete weather analysis
        """
        # Atmospheric profile
        atm_profile = self.weather.calculate_atmospheric_pressure(
            altitude_m, surface_temp_c + 273.15, surface_pressure_hpa * 100
        )

        # Dewpoint and moisture
        moisture = self.weather.estimate_dewpoint(surface_temp_c, relative_humidity)

        # Storm potential
        cape = self.weather.predict_convective_available_potential_energy(
            surface_temp_c, moisture['dewpoint_c'], surface_pressure_hpa
        )

        return {
            'atmospheric_profile': atm_profile,
            'moisture_analysis': moisture,
            'convective_potential': cape
        }
