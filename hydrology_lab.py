"""
Hydrology Laboratory - Production Implementation
Real rainfall-runoff modeling, groundwater flow, evapotranspiration, and flood analysis
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import math
from scipy import stats
from scipy.special import erfc


class SoilType(Enum):
    """USDA soil classification for hydrologic soil groups"""
    A = "Sand, loamy sand, or sandy loam"  # High infiltration
    B = "Silt loam or loam"  # Moderate infiltration
    C = "Sandy clay loam"  # Low infiltration
    D = "Clay loam, clay, or high water table"  # Very low infiltration


class AquiferType(Enum):
    """Aquifer types for groundwater modeling"""
    CONFINED = "confined"
    UNCONFINED = "unconfined"
    LEAKY = "leaky"
    FRACTURED = "fractured"


@dataclass
class CatchmentParameters:
    """Watershed characteristics for hydrologic modeling"""
    area: float  # km²
    slope: float  # Average slope (m/m)
    length: float  # Main channel length (km)
    cn: float  # SCS curve number
    soil_type: SoilType
    land_use: str  # agricultural, urban, forest, etc.
    imperviousness: float  # Fraction (0-1)


@dataclass
class GroundwaterParameters:
    """Aquifer and well parameters"""
    transmissivity: float  # m²/day
    storage_coefficient: float  # dimensionless
    hydraulic_conductivity: float  # m/day
    aquifer_thickness: float  # m
    porosity: float  # fraction
    specific_yield: float  # fraction
    aquifer_type: AquiferType


class HydrologyLab:
    """
    Comprehensive hydrology laboratory
    Implements real-world hydrologic modeling algorithms
    """

    def __init__(self):
        # Physical constants
        self.gravity = 9.81  # m/s²
        self.water_density = 1000  # kg/m³
        self.latent_heat_vaporization = 2.45e6  # J/kg

        # SCS curve number lookup table (typical values)
        self.cn_table = {
            'agricultural': {
                SoilType.A: 49,
                SoilType.B: 69,
                SoilType.C: 79,
                SoilType.D: 84
            },
            'urban': {
                SoilType.A: 77,
                SoilType.B: 85,
                SoilType.C: 90,
                SoilType.D: 92
            },
            'forest': {
                SoilType.A: 30,
                SoilType.B: 55,
                SoilType.C: 70,
                SoilType.D: 77
            },
            'meadow': {
                SoilType.A: 30,
                SoilType.B: 58,
                SoilType.C: 71,
                SoilType.D: 78
            }
        }

        # Regional regression coefficients for different climates
        self.regional_coefficients = {
            'humid': {'a': 2.0, 'b': 0.65, 'c': 0.5},
            'semi_arid': {'a': 1.5, 'b': 0.60, 'c': 0.45},
            'arid': {'a': 1.0, 'b': 0.55, 'c': 0.40}
        }

    def scs_runoff_model(self, rainfall: float, catchment: CatchmentParameters) -> Dict:
        """
        SCS Curve Number method for rainfall-runoff estimation
        USDA-NRCS standard methodology
        """
        # Adjust CN for antecedent moisture conditions
        # Using AMC II (average conditions) as default
        cn = catchment.cn

        # Calculate potential maximum retention
        S = (25400 / cn) - 254  # mm

        # Initial abstraction (typically 0.2S)
        Ia = 0.2 * S

        # Calculate runoff depth
        if rainfall <= Ia:
            Q = 0
        else:
            Q = ((rainfall - Ia) ** 2) / (rainfall - Ia + S)

        # Peak discharge using SCS triangular unit hydrograph
        # Time of concentration (Kirpich formula)
        tc = 0.0195 * (catchment.length * 1000) ** 0.77 * catchment.slope ** -0.385  # minutes

        # Time to peak
        tp = 0.6 * tc  # minutes

        # Peak discharge per unit depth
        qp = 0.208 * catchment.area / (tp / 60)  # m³/s per mm

        # Actual peak discharge
        Qpeak = qp * Q  # m³/s

        # Calculate lag time
        lag_time = 0.6 * tc

        # Runoff volume
        volume = Q * catchment.area * 1000  # m³

        # Runoff coefficient
        if rainfall > 0:
            runoff_coefficient = Q / rainfall
        else:
            runoff_coefficient = 0

        return {
            'runoff_depth': Q,
            'peak_discharge': Qpeak,
            'time_of_concentration': tc,
            'time_to_peak': tp,
            'lag_time': lag_time,
            'runoff_volume': volume,
            'runoff_coefficient': runoff_coefficient,
            'curve_number': cn,
            'potential_retention': S,
            'initial_abstraction': Ia
        }

    def darcy_groundwater_flow(self, gw_params: GroundwaterParameters,
                               head_gradient: float, flow_area: float) -> Dict:
        """
        Darcy's law for groundwater flow calculation
        Fundamental equation for porous media flow
        """
        # Basic Darcy flow
        Q = gw_params.hydraulic_conductivity * head_gradient * flow_area  # m³/day

        # Average linear velocity (actual water velocity)
        v_avg = Q / (flow_area * gw_params.porosity)  # m/day

        # Seepage velocity
        v_seepage = gw_params.hydraulic_conductivity * head_gradient  # m/day

        # Reynolds number for porous media
        # Using characteristic length as grain diameter (assume 1mm)
        d_grain = 0.001  # m
        kinematic_viscosity = 1e-6  # m²/s at 20°C
        Re = v_seepage * d_grain / kinematic_viscosity

        # Check if flow is laminar (Darcy valid for Re < 1-10)
        is_laminar = Re < 10

        # Calculate transmissivity if not provided
        if gw_params.aquifer_thickness > 0:
            calculated_T = gw_params.hydraulic_conductivity * gw_params.aquifer_thickness
        else:
            calculated_T = gw_params.transmissivity

        # Storage capacity
        storage_capacity = gw_params.storage_coefficient * flow_area * gw_params.aquifer_thickness

        return {
            'discharge': Q,
            'average_velocity': v_avg,
            'seepage_velocity': v_seepage,
            'reynolds_number': Re,
            'flow_is_laminar': is_laminar,
            'transmissivity': calculated_T,
            'storage_capacity': storage_capacity,
            'travel_time_per_meter': 1 / v_avg if v_avg > 0 else float('inf')
        }

    def theis_well_solution(self, gw_params: GroundwaterParameters,
                           pumping_rate: float, distance: float, time: float) -> Dict:
        """
        Theis solution for transient flow to a well in confined aquifer
        Classic analytical solution for well hydraulics
        """
        # Calculate dimensionless time parameter u
        u = (distance ** 2 * gw_params.storage_coefficient) / \
            (4 * gw_params.transmissivity * time)

        # Well function W(u) using exponential integral approximation
        if u <= 0.01:
            # Small u approximation (Jacob's method)
            W_u = -0.5772 - np.log(u)
        elif u <= 1.0:
            # Series expansion for intermediate u
            W_u = (-0.5772 - np.log(u) + u - u**2/4 + u**3/18 -
                   u**4/96 + u**5/600)
        else:
            # Large u approximation
            W_u = np.exp(-u) / u * (1 - 1/u + 2/u**2 - 6/u**3)

        # Drawdown calculation
        s = (pumping_rate / (4 * np.pi * gw_params.transmissivity)) * W_u

        # Radius of influence (approximate)
        R = 1.5 * np.sqrt(gw_params.transmissivity * time / gw_params.storage_coefficient)

        # Specific capacity
        if distance < 0.1:  # At the well
            specific_capacity = pumping_rate / s if s > 0 else float('inf')
        else:
            specific_capacity = None

        # Cone of depression volume
        cone_volume = np.pi * R ** 2 * s / 3

        return {
            'drawdown': s,
            'well_function': W_u,
            'u_parameter': u,
            'radius_of_influence': R,
            'specific_capacity': specific_capacity,
            'cone_volume': cone_volume,
            'time': time,
            'distance': distance
        }

    def penman_monteith_evapotranspiration(self, temperature: float, humidity: float,
                                          wind_speed: float, solar_radiation: float,
                                          crop_coefficient: float = 1.0) -> Dict:
        """
        FAO-56 Penman-Monteith equation for evapotranspiration
        Standard method for ET calculation
        """
        # Constants
        gamma = 0.067  # Psychrometric constant (kPa/°C)
        cp = 1.013  # Specific heat of air (kJ/kg/°C)

        # Saturation vapor pressure (Tetens formula)
        es = 0.6108 * np.exp(17.27 * temperature / (temperature + 237.3))  # kPa

        # Actual vapor pressure
        ea = es * humidity / 100  # kPa

        # Vapor pressure deficit
        vpd = es - ea

        # Slope of saturation vapor pressure curve
        delta = 4098 * es / (temperature + 237.3) ** 2  # kPa/°C

        # Net radiation (simplified - normally requires more inputs)
        # Assume 50% of solar radiation is net radiation
        Rn = 0.5 * solar_radiation  # MJ/m²/day

        # Soil heat flux (assume negligible for daily calculations)
        G = 0

        # Aerodynamic resistance (simplified for reference crop)
        ra = 208 / wind_speed  # s/m

        # Surface resistance for reference crop (grass)
        rs = 70  # s/m

        # Reference evapotranspiration (grass reference)
        ET0_numerator = (0.408 * delta * (Rn - G) +
                        gamma * (900 / (temperature + 273)) * wind_speed * vpd)
        ET0_denominator = delta + gamma * (1 + 0.34 * wind_speed)
        ET0 = ET0_numerator / ET0_denominator  # mm/day

        # Actual crop evapotranspiration
        ETc = ET0 * crop_coefficient  # mm/day

        # Water requirement per hectare
        water_volume = ETc * 10  # m³/ha/day

        # Calculate potential evaporation (open water)
        # Using simplified Penman equation
        Ep = (delta * Rn + gamma * 6.43 * (1 + 0.536 * wind_speed) * vpd) / \
             (delta + gamma)

        return {
            'reference_et': ET0,
            'crop_et': ETc,
            'potential_evaporation': Ep,
            'vapor_pressure_deficit': vpd,
            'net_radiation': Rn,
            'water_requirement_per_ha': water_volume,
            'delta_slope': delta,
            'psychrometric_constant': gamma,
            'aerodynamic_resistance': ra
        }

    def rational_method_peak_flow(self, intensity: float, catchment: CatchmentParameters) -> Dict:
        """
        Rational method for peak discharge estimation
        Q = CIA, commonly used for small urban catchments
        """
        # Runoff coefficient based on land use and imperviousness
        if catchment.land_use == 'urban':
            C = 0.3 + 0.6 * catchment.imperviousness
        elif catchment.land_use == 'agricultural':
            C = 0.2 + 0.3 * catchment.imperviousness
        elif catchment.land_use == 'forest':
            C = 0.1 + 0.2 * catchment.imperviousness
        else:
            C = 0.3  # Default

        # Adjust for slope
        if catchment.slope > 0.05:
            C = min(C * 1.1, 0.95)
        elif catchment.slope < 0.01:
            C = C * 0.9

        # Peak discharge
        Q = C * intensity * catchment.area / 3.6  # m³/s
        # Note: Division by 3.6 converts from mm/hr·km² to m³/s

        # Time of concentration (Kirpich formula)
        tc = 0.0195 * (catchment.length * 1000) ** 0.77 * catchment.slope ** -0.385

        # Rainfall intensity for design storm (using IDF curve approximation)
        # i = a / (t + b)^c where t is duration in minutes
        # Using regional coefficients
        regional = self.regional_coefficients.get('humid')  # Default to humid
        i_design = regional['a'] * 1000 / (tc + regional['b']) ** regional['c']

        # Design peak flow
        Q_design = C * i_design * catchment.area / 3.6

        return {
            'peak_discharge': Q,
            'runoff_coefficient': C,
            'time_of_concentration': tc,
            'design_intensity': i_design,
            'design_peak_flow': Q_design,
            'area': catchment.area,
            'method': 'Rational Method'
        }

    def gumbel_flood_frequency(self, annual_maxima: List[float],
                               return_periods: List[float] = [2, 5, 10, 25, 50, 100]) -> Dict:
        """
        Gumbel distribution for flood frequency analysis
        Standard method for extreme value analysis
        """
        # Convert to numpy array
        data = np.array(annual_maxima)
        n = len(data)

        # Calculate mean and standard deviation
        mean_q = np.mean(data)
        std_q = np.std(data, ddof=1)

        # Gumbel parameters using method of moments
        # For large n, yn ≈ 0.5772 and sn ≈ 1.2825
        yn = 0.5772  # Euler's constant
        sn = 1.2825

        # Calculate Gumbel parameters
        alpha = sn / std_q  # Scale parameter
        u = mean_q - yn / alpha  # Location parameter

        # Calculate flood magnitudes for different return periods
        flood_magnitudes = {}
        exceedance_probabilities = {}

        for T in return_periods:
            # Reduced variate
            yt = -np.log(-np.log(1 - 1/T))

            # Flood magnitude
            Qt = u + yt / alpha

            flood_magnitudes[T] = Qt
            exceedance_probabilities[T] = 1 / T

        # Confidence intervals (approximate 95% CI)
        confidence_intervals = {}
        for T in return_periods:
            yt = -np.log(-np.log(1 - 1/T))
            se = std_q * np.sqrt(1 + 1.14 * yt + 1.1 * yt**2) / np.sqrt(n)
            Qt = flood_magnitudes[T]
            confidence_intervals[T] = {
                'lower': Qt - 1.96 * se,
                'upper': Qt + 1.96 * se
            }

        # Goodness of fit (Kolmogorov-Smirnov test)
        sorted_data = np.sort(data)
        empirical_prob = np.arange(1, n + 1) / (n + 1)
        theoretical_prob = np.exp(-np.exp(-alpha * (sorted_data - u)))
        ks_statistic = np.max(np.abs(empirical_prob - theoretical_prob))

        return {
            'flood_magnitudes': flood_magnitudes,
            'exceedance_probabilities': exceedance_probabilities,
            'confidence_intervals': confidence_intervals,
            'gumbel_parameters': {
                'location': u,
                'scale': 1/alpha
            },
            'statistics': {
                'mean': mean_q,
                'std': std_q,
                'ks_statistic': ks_statistic,
                'sample_size': n
            }
        }

    def muskingum_routing(self, inflow: List[float], K: float, X: float,
                         dt: float = 1.0) -> Dict:
        """
        Muskingum method for flood routing through river reaches
        Linear reservoir routing model
        """
        # Muskingum coefficients
        C0 = (dt - 2 * K * X) / (2 * K * (1 - X) + dt)
        C1 = (dt + 2 * K * X) / (2 * K * (1 - X) + dt)
        C2 = (2 * K * (1 - X) - dt) / (2 * K * (1 - X) + dt)

        # Check stability (C0, C1, C2 should be positive)
        if C0 < 0 or C1 < 0 or C2 < 0:
            return {'error': 'Unstable routing parameters'}

        # Route the flood
        outflow = [inflow[0]]  # Initial condition

        for i in range(1, len(inflow)):
            Q_out = C0 * inflow[i] + C1 * inflow[i-1] + C2 * outflow[i-1]
            outflow.append(Q_out)

        # Calculate storage
        storage = []
        for i in range(len(inflow)):
            S = K * (X * inflow[i] + (1 - X) * outflow[i])
            storage.append(S)

        # Peak attenuation
        peak_inflow = max(inflow)
        peak_outflow = max(outflow)
        attenuation = (peak_inflow - peak_outflow) / peak_inflow * 100

        # Lag time
        peak_in_time = inflow.index(peak_inflow) * dt
        peak_out_time = outflow.index(peak_outflow) * dt
        lag = peak_out_time - peak_in_time

        return {
            'outflow': outflow,
            'storage': storage,
            'coefficients': {
                'C0': C0,
                'C1': C1,
                'C2': C2
            },
            'parameters': {
                'K': K,
                'X': X,
                'dt': dt
            },
            'peak_attenuation': attenuation,
            'lag_time': lag,
            'peak_inflow': peak_inflow,
            'peak_outflow': peak_outflow
        }

    def unit_hydrograph_convolution(self, rainfall: List[float],
                                   unit_hydrograph: List[float]) -> Dict:
        """
        Convolution of rainfall with unit hydrograph
        Linear systems approach to rainfall-runoff modeling
        """
        # Perform convolution
        n_rain = len(rainfall)
        n_uh = len(unit_hydrograph)
        n_out = n_rain + n_uh - 1

        discharge = np.zeros(n_out)

        for i in range(n_rain):
            for j in range(n_uh):
                if i + j < n_out:
                    discharge[i + j] += rainfall[i] * unit_hydrograph[j]

        # Calculate runoff volume
        volume = np.sum(discharge)

        # Peak discharge and time to peak
        peak_discharge = np.max(discharge)
        time_to_peak = np.argmax(discharge)

        # Calculate centroid (center of mass) of hydrograph
        time_array = np.arange(len(discharge))
        centroid = np.sum(time_array * discharge) / np.sum(discharge)

        # Calculate lag time (centroid of rainfall to centroid of runoff)
        rain_centroid = np.sum(np.arange(len(rainfall)) * rainfall) / np.sum(rainfall)
        lag_time = centroid - rain_centroid

        return {
            'discharge': discharge.tolist(),
            'peak_discharge': peak_discharge,
            'time_to_peak': time_to_peak,
            'total_volume': volume,
            'centroid': centroid,
            'lag_time': lag_time,
            'n_ordinates': len(discharge)
        }

    def green_ampt_infiltration(self, rainfall_rate: float, time: float,
                               soil_type: SoilType, initial_moisture: float = 0.2) -> Dict:
        """
        Green-Ampt infiltration model
        Physically-based infiltration calculation
        """
        # Soil hydraulic properties (typical values)
        soil_props = {
            SoilType.A: {
                'K': 11.78,  # Hydraulic conductivity (mm/hr)
                'psi': 49.5,  # Wetting front suction (mm)
                'porosity': 0.437,
                'field_capacity': 0.062
            },
            SoilType.B: {
                'K': 3.38,
                'psi': 110.1,
                'porosity': 0.453,
                'field_capacity': 0.105
            },
            SoilType.C: {
                'K': 0.38,
                'psi': 210.0,
                'porosity': 0.464,
                'field_capacity': 0.187
            },
            SoilType.D: {
                'K': 0.13,
                'psi': 273.0,
                'porosity': 0.475,
                'field_capacity': 0.265
            }
        }

        props = soil_props[soil_type]
        K = props['K']
        psi = props['psi']
        theta_e = props['porosity']
        theta_i = initial_moisture

        # Moisture deficit
        delta_theta = theta_e - theta_i

        # Calculate cumulative infiltration
        if rainfall_rate <= K:
            # All rainfall infiltrates
            F = rainfall_rate * time
            f = rainfall_rate
        else:
            # Ponding occurs
            # Time to ponding
            tp = K * psi * delta_theta / (rainfall_rate * (rainfall_rate - K))

            if time <= tp:
                F = rainfall_rate * time
                f = rainfall_rate
            else:
                # After ponding
                # Solve implicit equation for F using Newton-Raphson
                F_guess = K * (time - tp)
                for _ in range(10):
                    func = F_guess - psi * delta_theta * np.log(1 + F_guess / (psi * delta_theta)) - K * (time - tp)
                    dfunc = 1 - psi * delta_theta / (psi * delta_theta + F_guess)
                    F_new = F_guess - func / dfunc
                    if abs(F_new - F_guess) < 0.001:
                        break
                    F_guess = F_new

                F = F_new
                # Infiltration rate
                f = K * (1 + psi * delta_theta / F)

        # Runoff
        cumulative_rainfall = rainfall_rate * time
        cumulative_runoff = max(0, cumulative_rainfall - F)
        runoff_rate = max(0, rainfall_rate - f)

        return {
            'cumulative_infiltration': F,
            'infiltration_rate': f,
            'cumulative_runoff': cumulative_runoff,
            'runoff_rate': runoff_rate,
            'time_to_ponding': tp if rainfall_rate > K else None,
            'hydraulic_conductivity': K,
            'wetting_front_suction': psi,
            'moisture_deficit': delta_theta,
            'soil_type': soil_type.value
        }

    def synthetic_unit_hydrograph(self, catchment: CatchmentParameters,
                                 duration: float = 1.0) -> Dict:
        """
        Snyder's synthetic unit hydrograph
        Generate UH for ungauged basins
        """
        # Basin lag (hours)
        Ct = 1.5  # Coefficient (1.8-2.2 for natural, 0.3-1.5 for urban)
        L = catchment.length  # km
        Lc = 0.6 * L  # Distance to centroid (approximation)

        tp = Ct * (L * Lc) ** 0.3  # hours

        # Peak discharge per unit area per unit depth
        Cp = 0.75  # Coefficient (0.4-0.8, higher for steep basins)
        qp = Cp / tp  # m³/s/km²/mm

        # Time base
        tb = 5 * tp  # hours

        # Generate ordinates using gamma distribution approximation
        t = np.linspace(0, tb, int(tb / duration) + 1)

        # Shape parameter
        n = 3.5  # Typical value

        # Dimensionless hydrograph
        t_tp = t / tp
        q_qp = (t_tp ** (n - 1) * np.exp(-n * t_tp / np.e)) / (np.e ** (n - 1) / n ** n)

        # Actual discharge
        Q = q_qp * qp * catchment.area

        # Normalize to unit depth (1 mm)
        volume = np.trapz(Q, t) * 3600  # m³
        unit_volume = catchment.area * 1000  # m³ for 1 mm
        Q_unit = Q * unit_volume / volume

        return {
            'ordinates': Q_unit.tolist(),
            'time': t.tolist(),
            'lag_time': tp,
            'peak_discharge': np.max(Q_unit),
            'time_to_peak': tp,
            'time_base': tb,
            'peak_factor': qp,
            'duration': duration
        }

    def stream_flow_statistics(self, daily_flows: List[float]) -> Dict:
        """
        Calculate comprehensive streamflow statistics
        Flow duration curves, baseflow index, etc.
        """
        flows = np.array(daily_flows)

        # Basic statistics
        mean_flow = np.mean(flows)
        median_flow = np.median(flows)
        std_flow = np.std(flows)
        cv = std_flow / mean_flow  # Coefficient of variation

        # Flow duration curve
        sorted_flows = np.sort(flows)[::-1]
        n = len(flows)
        exceedance_prob = np.arange(1, n + 1) / (n + 1) * 100

        # Key percentiles
        Q95 = np.percentile(flows, 5)   # Low flow
        Q90 = np.percentile(flows, 10)  # Low flow
        Q50 = np.percentile(flows, 50)  # Median
        Q10 = np.percentile(flows, 90)  # High flow
        Q5 = np.percentile(flows, 95)   # High flow

        # 7-day low flow
        window = 7
        if len(flows) >= window:
            rolling_mean = np.convolve(flows, np.ones(window)/window, mode='valid')
            Q7_10 = np.percentile(rolling_mean, 10)
        else:
            Q7_10 = Q90

        # Baseflow separation (simple digital filter)
        alpha = 0.925  # Filter parameter
        baseflow = np.zeros_like(flows)
        baseflow[0] = flows[0]

        for i in range(1, len(flows)):
            baseflow[i] = alpha * baseflow[i-1] + (1 - alpha) * flows[i]
            baseflow[i] = min(baseflow[i], flows[i])

        # Baseflow index
        BFI = np.sum(baseflow) / np.sum(flows)

        # Richards-Baker flashiness index
        sum_abs_changes = np.sum(np.abs(np.diff(flows)))
        sum_flows = np.sum(flows)
        RB_index = sum_abs_changes / sum_flows if sum_flows > 0 else 0

        return {
            'mean': mean_flow,
            'median': median_flow,
            'std': std_flow,
            'cv': cv,
            'percentiles': {
                'Q95': Q95,
                'Q90': Q90,
                'Q50': Q50,
                'Q10': Q10,
                'Q5': Q5
            },
            'Q7_10': Q7_10,
            'baseflow_index': BFI,
            'flashiness_index': RB_index,
            'flow_duration': {
                'flows': sorted_flows.tolist()[:100],  # First 100 for display
                'exceedance': exceedance_prob.tolist()[:100]
            }
        }


def demo():
    """Demonstration of Hydrology Lab capabilities"""
    print("Hydrology Laboratory - Production Demo")
    print("=" * 60)

    lab = HydrologyLab()

    # Demo 1: SCS Curve Number Runoff
    print("\n1. SCS Curve Number Rainfall-Runoff Model:")
    catchment = CatchmentParameters(
        area=50.0,  # km²
        slope=0.02,  # 2% slope
        length=10.0,  # km
        cn=75,  # Curve number
        soil_type=SoilType.B,
        land_use='agricultural',
        imperviousness=0.1
    )

    runoff = lab.scs_runoff_model(50.0, catchment)  # 50mm rainfall
    print(f"   Runoff Depth: {runoff['runoff_depth']:.2f} mm")
    print(f"   Peak Discharge: {runoff['peak_discharge']:.2f} m³/s")
    print(f"   Time of Concentration: {runoff['time_of_concentration']:.1f} min")
    print(f"   Runoff Coefficient: {runoff['runoff_coefficient']:.3f}")

    # Demo 2: Groundwater Flow
    print("\n2. Darcy Groundwater Flow:")
    gw_params = GroundwaterParameters(
        transmissivity=500,  # m²/day
        storage_coefficient=0.0001,
        hydraulic_conductivity=10,  # m/day
        aquifer_thickness=50,  # m
        porosity=0.25,
        specific_yield=0.15,
        aquifer_type=AquiferType.CONFINED
    )

    flow = lab.darcy_groundwater_flow(gw_params, 0.001, 100)  # gradient=0.001, area=100m²
    print(f"   Discharge: {flow['discharge']:.3f} m³/day")
    print(f"   Average Velocity: {flow['average_velocity']:.4f} m/day")
    print(f"   Flow is Laminar: {flow['flow_is_laminar']}")

    # Demo 3: Penman-Monteith ET
    print("\n3. Penman-Monteith Evapotranspiration:")
    et_result = lab.penman_monteith_evapotranspiration(
        temperature=25,  # °C
        humidity=60,  # %
        wind_speed=2,  # m/s
        solar_radiation=20,  # MJ/m²/day
        crop_coefficient=1.15  # Corn
    )
    print(f"   Reference ET: {et_result['reference_et']:.2f} mm/day")
    print(f"   Crop ET: {et_result['crop_et']:.2f} mm/day")
    print(f"   Water Requirement: {et_result['water_requirement_per_ha']:.1f} m³/ha/day")

    # Demo 4: Flood Frequency Analysis
    print("\n4. Gumbel Flood Frequency Analysis:")
    # Example annual maximum flows (m³/s)
    annual_maxima = [120, 150, 180, 95, 210, 165, 145, 190, 175, 160,
                     140, 155, 170, 185, 130, 195, 125, 175, 165, 155]

    flood_freq = lab.gumbel_flood_frequency(annual_maxima)
    print("   Return Period Floods:")
    for T, Q in flood_freq['flood_magnitudes'].items():
        ci = flood_freq['confidence_intervals'][T]
        print(f"     {T}-year: {Q:.1f} m³/s (95% CI: {ci['lower']:.1f}-{ci['upper']:.1f})")

    # Demo 5: Green-Ampt Infiltration
    print("\n5. Green-Ampt Infiltration Model:")
    infiltration = lab.green_ampt_infiltration(
        rainfall_rate=20,  # mm/hr
        time=2.0,  # hours
        soil_type=SoilType.B,
        initial_moisture=0.15
    )
    print(f"   Cumulative Infiltration: {infiltration['cumulative_infiltration']:.1f} mm")
    print(f"   Infiltration Rate: {infiltration['infiltration_rate']:.2f} mm/hr")
    print(f"   Cumulative Runoff: {infiltration['cumulative_runoff']:.1f} mm")

    print("\nLab ready for production use!")
    return lab


if __name__ == "__main__":
    demo()