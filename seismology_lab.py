"""
Seismology Laboratory - Production Implementation
Real P/S wave propagation, magnitude calculations, site amplification, and hazard analysis
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import math
from scipy import signal
from scipy.special import jn  # Bessel functions


class WaveType(Enum):
    """Seismic wave types"""
    P_WAVE = "P-wave"  # Primary/compression
    S_WAVE = "S-wave"  # Secondary/shear
    SURFACE_WAVE = "Surface"
    RAYLEIGH = "Rayleigh"
    LOVE = "Love"


class SoilClass(Enum):
    """NEHRP soil classifications for site amplification"""
    A = "Hard rock"  # Vs30 > 1500 m/s
    B = "Rock"  # 760 < Vs30 <= 1500 m/s
    C = "Very dense soil/soft rock"  # 360 < Vs30 <= 760 m/s
    D = "Stiff soil"  # 180 < Vs30 <= 360 m/s
    E = "Soft soil"  # Vs30 < 180 m/s


@dataclass
class EarthquakeSource:
    """Earthquake source parameters"""
    latitude: float
    longitude: float
    depth: float  # km
    magnitude: float  # Moment magnitude
    strike: float  # degrees
    dip: float  # degrees
    rake: float  # degrees
    origin_time: float  # seconds since reference


@dataclass
class SeismicStation:
    """Seismic station parameters"""
    station_code: str
    latitude: float
    longitude: float
    elevation: float  # m
    soil_class: SoilClass


class SeismologyLab:
    """
    Comprehensive seismology laboratory
    Implements real seismic wave propagation and hazard analysis
    """

    def __init__(self):
        # Earth parameters
        self.earth_radius = 6371.0  # km
        self.gravity = 9.81  # m/s²

        # Typical crustal velocities
        self.crustal_velocities = {
            'vp': 6.0,  # P-wave velocity km/s
            'vs': 3.5,  # S-wave velocity km/s
            'density': 2700  # kg/m³
        }

        # Attenuation parameters
        self.q_factor = {
            'P': 600,  # Quality factor for P-waves
            'S': 300   # Quality factor for S-waves
        }

    def p_wave_propagation(self, source: EarthquakeSource, station: SeismicStation,
                           earth_model: str = 'ak135') -> Dict:
        """
        Calculate P-wave propagation characteristics
        Using ray theory and travel time tables
        """
        # Calculate epicentral distance (great circle)
        distance = self._calculate_distance(
            source.latitude, source.longitude,
            station.latitude, station.longitude
        )

        # Hypocentral distance
        hypo_distance = np.sqrt(distance ** 2 + source.depth ** 2)

        # P-wave travel time (simplified 1D model)
        if earth_model == 'ak135':
            # AK135 travel time approximation
            if source.depth < 35:  # Crustal
                vp = 6.0 + 0.02 * source.depth
            elif source.depth < 210:  # Upper mantle
                vp = 8.0 + 0.003 * source.depth
            else:  # Transition zone
                vp = 10.0 + 0.001 * source.depth
        else:
            vp = self.crustal_velocities['vp']

        travel_time = hypo_distance / vp

        # Ray parameter (horizontal slowness)
        ray_param = distance / (vp * hypo_distance)

        # Takeoff angle
        takeoff_angle = np.arcsin(ray_param * vp) * 180 / np.pi

        # Incident angle at station
        if source.depth > 0:
            incident_angle = np.arcsin(distance / hypo_distance) * 180 / np.pi
        else:
            incident_angle = 90.0

        # Amplitude decay with distance (geometric spreading + attenuation)
        geometric_spreading = 1 / hypo_distance
        freq = 1.0  # 1 Hz reference frequency
        attenuation = np.exp(-np.pi * freq * hypo_distance / (self.q_factor['P'] * vp))
        amplitude_factor = geometric_spreading * attenuation

        # First motion polarity (simplified)
        # Based on focal mechanism and takeoff angle
        polarity = self._calculate_first_motion(
            source.strike, source.dip, source.rake, takeoff_angle
        )

        return {
            'travel_time': travel_time,
            'epicentral_distance': distance,
            'hypocentral_distance': hypo_distance,
            'ray_parameter': ray_param,
            'takeoff_angle': takeoff_angle,
            'incident_angle': incident_angle,
            'velocity': vp,
            'amplitude_factor': amplitude_factor,
            'first_motion': polarity,
            'arrival_time': source.origin_time + travel_time
        }

    def s_wave_propagation(self, source: EarthquakeSource, station: SeismicStation) -> Dict:
        """
        Calculate S-wave propagation characteristics
        S-waves are slower and don't travel through liquid
        """
        # Calculate distances
        distance = self._calculate_distance(
            source.latitude, source.longitude,
            station.latitude, station.longitude
        )

        hypo_distance = np.sqrt(distance ** 2 + source.depth ** 2)

        # S-wave velocity (approximately Vp/√3)
        if source.depth < 35:
            vs = 3.5 + 0.01 * source.depth
        else:
            vs = 4.5 + 0.002 * source.depth

        travel_time = hypo_distance / vs

        # S-wave specific parameters
        ray_param = distance / (vs * hypo_distance)
        takeoff_angle = np.arcsin(ray_param * vs) * 180 / np.pi

        # S-waves have stronger attenuation
        geometric_spreading = 1 / hypo_distance
        freq = 1.0
        attenuation = np.exp(-np.pi * freq * hypo_distance / (self.q_factor['S'] * vs))
        amplitude_factor = geometric_spreading * attenuation

        # S-waves are polarized (SH and SV components)
        sh_amplitude = amplitude_factor * np.sin(source.rake * np.pi / 180)
        sv_amplitude = amplitude_factor * np.cos(source.rake * np.pi / 180)

        # S-P time (important for earthquake location)
        p_travel = hypo_distance / (1.73 * vs)  # Vp ≈ 1.73 * Vs
        sp_time = travel_time - p_travel

        return {
            'travel_time': travel_time,
            'velocity': vs,
            'sp_time': sp_time,
            'ray_parameter': ray_param,
            'takeoff_angle': takeoff_angle,
            'amplitude_factor': amplitude_factor,
            'sh_amplitude': sh_amplitude,
            'sv_amplitude': sv_amplitude,
            'arrival_time': source.origin_time + travel_time,
            'hypocentral_distance': hypo_distance
        }

    def richter_magnitude(self, amplitude: float, distance: float,
                         station_correction: float = 0) -> Dict:
        """
        Calculate Richter (Local) Magnitude ML
        ML = log10(A) + distance_correction + station_correction
        """
        # Wood-Anderson seismograph amplitude in mm
        # Modern instruments need conversion
        wa_amplitude = amplitude * 2080  # Typical gain correction

        # Distance correction (Richter's original formula for S. California)
        if distance < 600:
            distance_correction = 1.11 * np.log10(distance / 100) + 0.00189 * (distance - 100) + 3.0
        else:
            distance_correction = 3.5  # Beyond 600 km, use different scale

        # Calculate magnitude
        ML = np.log10(wa_amplitude) + distance_correction + station_correction

        # Saturation check
        if ML > 6.5:
            saturated = True
            saturation_note = "ML saturates above ~6.5, use Mw for large events"
        else:
            saturated = False
            saturation_note = "Within valid range"

        return {
            'magnitude': ML,
            'amplitude_mm': wa_amplitude,
            'distance_km': distance,
            'distance_correction': distance_correction,
            'station_correction': station_correction,
            'saturated': saturated,
            'note': saturation_note
        }

    def moment_magnitude(self, moment: float) -> Dict:
        """
        Calculate Moment Magnitude (Mw)
        Mw = (2/3) * log10(M0) - 10.7
        where M0 is seismic moment in dyne·cm
        """
        # Convert to dyne·cm if given in N·m
        moment_dyne_cm = moment * 1e7  # N·m to dyne·cm

        # Hanks-Kanamori formula
        Mw = (2/3) * np.log10(moment_dyne_cm) - 10.7

        # Calculate fault parameters (assuming typical values)
        # M0 = μ * A * D (rigidity * area * displacement)
        mu = 3e10  # Typical rigidity in dyne/cm² (30 GPa)

        # Estimate fault area from moment (assuming D = 0.0001 * L)
        # M0 = μ * L² * 0.0001 * L = μ * 0.0001 * L³
        L = (moment_dyne_cm / (mu * 0.0001)) ** (1/3) / 1e5  # Convert to km
        area = L ** 2  # km²
        displacement = 0.0001 * L * 1000  # meters

        # Energy release (Gutenberg-Richter relation)
        # log E = 11.8 + 1.5 * M
        energy_joules = 10 ** (11.8 + 1.5 * Mw) * 1e-7  # ergs to joules

        return {
            'moment_magnitude': Mw,
            'seismic_moment_Nm': moment,
            'seismic_moment_dyne_cm': moment_dyne_cm,
            'estimated_fault_length_km': L,
            'estimated_fault_area_km2': area,
            'estimated_displacement_m': displacement,
            'energy_release_joules': energy_joules,
            'energy_tnt_equivalent': energy_joules / 4.184e9  # kilotons TNT
        }

    def site_amplification(self, input_motion: np.ndarray, frequency: np.ndarray,
                          soil_class: SoilClass, depth_to_bedrock: float = 30) -> Dict:
        """
        Calculate site amplification effects
        Based on NEHRP provisions and Vs30
        """
        # Get shear wave velocity for soil class
        vs30_ranges = {
            SoilClass.A: 1500,
            SoilClass.B: 900,
            SoilClass.C: 500,
            SoilClass.D: 250,
            SoilClass.E: 150
        }

        vs30 = vs30_ranges[soil_class]

        # Site amplification factors (simplified NEHRP)
        if soil_class == SoilClass.A:
            fa = 0.8  # Short period
            fv = 0.8  # Long period
        elif soil_class == SoilClass.B:
            fa = 1.0
            fv = 1.0
        elif soil_class == SoilClass.C:
            fa = 1.2
            fv = 1.3
        elif soil_class == SoilClass.D:
            fa = 1.6
            fv = 1.5
        else:  # Class E
            fa = 2.5
            fv = 1.7

        # Frequency-dependent amplification
        # Using quarter-wavelength method
        fundamental_freq = vs30 / (4 * depth_to_bedrock)

        # Transfer function (simplified)
        amplification = np.ones_like(frequency)
        for i, f in enumerate(frequency):
            if f < 0.1:
                amplification[i] = fv
            elif f > 10:
                amplification[i] = fa
            else:
                # Resonance at fundamental frequency
                Q = 20  # Quality factor for soil
                amplification[i] = 1 + (fa - 1) * fundamental_freq ** 2 / \
                                  (fundamental_freq ** 2 + (f - fundamental_freq) ** 2 / Q ** 2)

        # Apply amplification to input motion
        output_motion = input_motion * np.mean(amplification)

        # Peak ground acceleration amplification
        pga_amplification = fa
        pgv_amplification = fv

        # Duration effects (softer soils extend duration)
        duration_factor = 1.0 + 0.3 * (5 - list(SoilClass).index(soil_class))

        return {
            'soil_class': soil_class.value,
            'vs30': vs30,
            'fundamental_frequency': fundamental_freq,
            'amplification_factors': {
                'short_period': fa,
                'long_period': fv,
                'pga': pga_amplification,
                'pgv': pgv_amplification
            },
            'frequency': frequency.tolist(),
            'amplification_spectrum': amplification.tolist(),
            'output_motion': output_motion.tolist(),
            'duration_factor': duration_factor,
            'depth_to_bedrock': depth_to_bedrock
        }

    def seismic_hazard_curve(self, site_lat: float, site_lon: float,
                            return_periods: List[float] = [475, 975, 2475]) -> Dict:
        """
        Calculate probabilistic seismic hazard curve
        Simplified PSHA calculation
        """
        # Example source parameters (would normally come from catalog)
        # Using Gutenberg-Richter recurrence: log N = a - b*M
        a = 4.0  # Activity rate
        b = 1.0  # B-value (typically 0.8-1.2)
        mmax = 7.5  # Maximum magnitude
        mmin = 4.0  # Minimum magnitude of interest

        # Annual rate of exceedance for different PGA levels
        pga_levels = np.logspace(-2, 0, 20)  # 0.01g to 1g
        annual_rates = []

        for pga in pga_levels:
            # Find magnitude that produces this PGA at the site
            # Using simplified attenuation relation: PGA = c1 * exp(c2*M) * R^c3
            # Where R is distance
            distance = 50  # Assume 50 km source distance

            # Boore-Atkinson 2008 GMPE (simplified)
            # log(PGA) = c1 + c2*M - c3*log(R) - c4*R
            c1, c2, c3, c4 = -3.5, 0.9, 1.5, 0.006

            # Solve for M given PGA
            M_required = (np.log10(pga) + c3 * np.log10(distance) + c4 * distance - c1) / c2

            if M_required < mmin:
                M_required = mmin
            elif M_required > mmax:
                rate = 0
                annual_rates.append(rate)
                continue

            # Annual rate of M >= M_required (Gutenberg-Richter)
            rate = 10 ** (a - b * M_required) - 10 ** (a - b * mmax)
            annual_rates.append(max(0, rate))

        annual_rates = np.array(annual_rates)

        # Calculate PGA for different return periods
        hazard_results = {}
        for rp in return_periods:
            annual_prob = 1 / rp
            # Find PGA corresponding to this annual probability
            if np.any(annual_rates > annual_prob):
                pga_rp = np.interp(annual_prob, annual_rates[::-1], pga_levels[::-1])
            else:
                pga_rp = pga_levels[-1]

            hazard_results[f'{rp}_year'] = {
                'pga': pga_rp,
                'annual_probability': annual_prob,
                'probability_50_years': 1 - np.exp(-50 * annual_prob)
            }

        # Design spectrum (simplified Type 1 from Eurocode 8)
        periods = np.logspace(-2, 1, 50)  # 0.01 to 10 seconds
        pga_design = hazard_results['475_year']['pga']

        spectrum = []
        for T in periods:
            if T < 0.15:
                Sa = pga_design * (1 + T / 0.15 * 1.5)
            elif T < 0.4:
                Sa = pga_design * 2.5
            elif T < 2.0:
                Sa = pga_design * 2.5 * (0.4 / T)
            else:
                Sa = pga_design * 2.5 * (0.4 * 2.0) / (T ** 2)
            spectrum.append(Sa)

        return {
            'site_location': {'lat': site_lat, 'lon': site_lon},
            'pga_levels': pga_levels.tolist(),
            'annual_rates': annual_rates.tolist(),
            'return_period_results': hazard_results,
            'design_spectrum': {
                'periods': periods.tolist(),
                'spectral_acceleration': spectrum
            },
            'seismic_parameters': {
                'a_value': a,
                'b_value': b,
                'mmax': mmax,
                'mmin': mmin
            }
        }

    def ground_motion_prediction(self, magnitude: float, distance: float,
                                site_class: SoilClass = SoilClass.C,
                                gmpe_model: str = 'boore_atkinson_2008') -> Dict:
        """
        Ground Motion Prediction Equation (GMPE)
        Estimates PGA, PGV, and response spectra
        """
        # Boore-Atkinson 2008 NGA model (simplified)
        if gmpe_model == 'boore_atkinson_2008':
            # Coefficients for PGA
            c1, c2, c3 = -3.512, 0.904, -1.328
            c4, c5, c6 = 0.149, -0.0031, 1.0

            # Site amplification factor
            site_factors = {
                SoilClass.A: 0.8,
                SoilClass.B: 1.0,
                SoilClass.C: 1.3,
                SoilClass.D: 1.6,
                SoilClass.E: 2.0
            }
            fs = site_factors.get(site_class, 1.3)

            # Calculate median PGA (g)
            log_pga = (c1 + c2 * magnitude + c3 * np.log10(distance) +
                      c4 * (magnitude - 6) ** 2 + c5 * distance + np.log10(fs))
            pga = 10 ** log_pga

            # PGV estimation (cm/s)
            # Approximate relation: PGV = PGA * g * T / (2*pi) where T ~ 0.5s
            pgv = pga * 981 * 0.5 / (2 * np.pi)

            # PGD estimation (cm)
            # Approximate: PGD = PGV * T / (2*pi) where T ~ 2s
            pgd = pgv * 2 / (2 * np.pi)

            # Standard deviation (total sigma)
            if magnitude < 5:
                sigma = 0.5
            elif magnitude < 7:
                sigma = 0.4
            else:
                sigma = 0.35

        # Response spectrum (simplified)
        periods = np.array([0.01, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0])
        sa_spectrum = []

        for T in periods:
            if T < 0.1:
                sa = pga
            elif T < 0.5:
                sa = pga * (2.5 - 1.5 * (T - 0.1) / 0.4)
            elif T < 2.0:
                sa = pga * 2.5 * (0.5 / T) ** 0.7
            else:
                sa = pga * 2.5 * (0.5 * 2.0 / T ** 2) ** 0.5

            sa_spectrum.append(sa)

        # Duration (Boore et al. 1989)
        if magnitude < 5.5:
            duration = 2
        elif magnitude < 7.5:
            duration = 10 + 20 * (magnitude - 5.5) / 2
        else:
            duration = 30

        # Arias Intensity (m/s)
        Ia = pga ** 2 * duration * np.pi / (2 * 9.81)

        return {
            'magnitude': magnitude,
            'distance': distance,
            'site_class': site_class.value,
            'pga': pga,
            'pgv': pgv,
            'pgd': pgd,
            'sigma': sigma,
            'response_spectrum': {
                'periods': periods.tolist(),
                'spectral_acceleration': sa_spectrum
            },
            'duration': duration,
            'arias_intensity': Ia,
            'gmpe_model': gmpe_model
        }

    def earthquake_early_warning(self, p_arrival_time: float, s_arrival_time: float,
                                p_amplitude: float, frequency: float) -> Dict:
        """
        Earthquake Early Warning calculations
        Based on P-wave parameters to estimate magnitude and intensity
        """
        # S-P time
        sp_time = s_arrival_time - p_arrival_time

        # Estimate distance (assuming Vp=6km/s, Vs=3.5km/s)
        vp, vs = 6.0, 3.5
        distance = sp_time * vp * vs / (vp - vs)

        # τc method (Characteristic period)
        # Larger earthquakes have longer period P-waves
        tau_c = 1 / frequency  # Simplified

        # Magnitude estimation from τc (Kanamori 2005)
        if tau_c < 0.5:
            mag_estimate = 4.0 + 2 * tau_c
        elif tau_c < 2.0:
            mag_estimate = 5.0 + 1.5 * (tau_c - 0.5)
        else:
            mag_estimate = 7.25 + 0.5 * (tau_c - 2.0)

        # Pd method (Peak displacement in first 3 seconds)
        # Simplified: use amplitude as proxy
        pd = p_amplitude * tau_c  # cm

        # Magnitude from Pd (Wu & Zhao 2006)
        mag_pd = 5.15 + 1.23 * np.log10(pd) if pd > 0 else mag_estimate

        # Average magnitude estimates
        magnitude = (mag_estimate + mag_pd) / 2

        # Predict PGA at site using GMPE
        gmpe = self.ground_motion_prediction(magnitude, distance)
        predicted_pga = gmpe['pga']
        predicted_intensity = self._pga_to_intensity(predicted_pga)

        # Warning time before S-wave
        current_time = p_arrival_time + 3  # 3 seconds for P-wave analysis
        warning_time = max(0, s_arrival_time - current_time)

        # Alert level
        if predicted_intensity < 4:
            alert_level = "No alert"
            action = "No action needed"
        elif predicted_intensity < 5:
            alert_level = "Information"
            action = "Be aware"
        elif predicted_intensity < 6:
            alert_level = "Weak shaking"
            action = "Drop, Cover, Hold On"
        elif predicted_intensity < 7:
            alert_level = "Moderate shaking"
            action = "Take protective action immediately"
        else:
            alert_level = "Strong shaking"
            action = "EMERGENCY - Take cover immediately"

        return {
            'estimated_magnitude': magnitude,
            'estimated_distance': distance,
            'warning_time': warning_time,
            'predicted_pga': predicted_pga,
            'predicted_intensity': predicted_intensity,
            'alert_level': alert_level,
            'recommended_action': action,
            'tau_c': tau_c,
            'pd': pd,
            'methods_used': ['tau_c', 'pd'],
            'sp_time': sp_time
        }

    def fault_rupture_model(self, magnitude: float, fault_type: str = 'strike_slip') -> Dict:
        """
        Empirical fault rupture scaling relationships
        Wells and Coppersmith (1994) relations
        """
        # Fault type coefficients for rupture length
        # log(L) = a + b*M
        length_params = {
            'strike_slip': {'a': -3.55, 'b': 0.74},
            'reverse': {'a': -2.86, 'b': 0.63},
            'normal': {'a': -2.01, 'b': 0.50},
            'all': {'a': -3.22, 'b': 0.69}
        }

        # Fault type coefficients for rupture area
        # log(A) = a + b*M
        area_params = {
            'strike_slip': {'a': -3.42, 'b': 0.90},
            'reverse': {'a': -3.99, 'b': 0.98},
            'normal': {'a': -2.87, 'b': 0.82},
            'all': {'a': -3.49, 'b': 0.91}
        }

        # Fault type coefficients for displacement
        # log(D) = a + b*M
        disp_params = {
            'strike_slip': {'a': -7.03, 'b': 1.03},
            'reverse': {'a': -1.84, 'b': 0.29},
            'normal': {'a': -4.45, 'b': 0.63},
            'all': {'a': -4.80, 'b': 0.69}
        }

        # Get parameters for fault type
        l_params = length_params.get(fault_type, length_params['all'])
        a_params = area_params.get(fault_type, area_params['all'])
        d_params = disp_params.get(fault_type, disp_params['all'])

        # Calculate rupture dimensions
        log_length = l_params['a'] + l_params['b'] * magnitude
        length = 10 ** log_length  # km

        log_area = a_params['a'] + a_params['b'] * magnitude
        area = 10 ** log_area  # km²

        log_disp = d_params['a'] + d_params['b'] * magnitude
        max_displacement = 10 ** log_disp  # m

        # Calculate width from area and length
        width = area / length  # km

        # Average displacement
        avg_displacement = max_displacement / 2  # Approximation

        # Seismic moment
        mu = 3e10  # N/m² (30 GPa)
        M0 = mu * area * 1e6 * avg_displacement  # N·m

        # Stress drop (Brune 1970)
        # Δσ = 7/16 * M0 / r³ where r is radius of circular fault
        radius = np.sqrt(area / np.pi)
        stress_drop = (7/16) * M0 / (radius * 1000) ** 3 / 1e6  # MPa

        # Rupture velocity (typically 0.7-0.9 of S-wave velocity)
        rupture_velocity = 0.8 * 3.5  # km/s

        # Rupture duration
        rupture_duration = length / rupture_velocity

        return {
            'magnitude': magnitude,
            'fault_type': fault_type,
            'rupture_length_km': length,
            'rupture_width_km': width,
            'rupture_area_km2': area,
            'max_displacement_m': max_displacement,
            'avg_displacement_m': avg_displacement,
            'seismic_moment_Nm': M0,
            'stress_drop_MPa': stress_drop,
            'rupture_velocity_km_s': rupture_velocity,
            'rupture_duration_s': rupture_duration
        }

    def surface_wave_dispersion(self, periods: np.ndarray, layer_model: Dict) -> Dict:
        """
        Calculate surface wave dispersion curves
        For Rayleigh and Love waves
        """
        # Simple two-layer model
        h1 = layer_model.get('layer1_thickness', 10)  # km
        vs1 = layer_model.get('layer1_vs', 3.0)  # km/s
        vs2 = layer_model.get('layer2_vs', 4.0)  # km/s
        vp1 = vs1 * 1.73  # Approximate Vp/Vs ratio
        vp2 = vs2 * 1.73

        # Rayleigh wave dispersion (simplified)
        rayleigh_velocities = []
        for T in periods:
            wavelength = T * vs1  # Approximate wavelength

            if wavelength < h1:
                # Short period - sensitive to shallow layer
                c_rayleigh = 0.92 * vs1
            elif wavelength > 4 * h1:
                # Long period - sensitive to deeper layer
                c_rayleigh = 0.92 * vs2
            else:
                # Transition zone
                weight = (wavelength - h1) / (3 * h1)
                c_rayleigh = 0.92 * (vs1 * (1 - weight) + vs2 * weight)

            rayleigh_velocities.append(c_rayleigh)

        # Love wave dispersion
        love_velocities = []
        for T in periods:
            wavelength = T * vs1

            if wavelength < h1:
                c_love = vs1
            elif wavelength > 4 * h1:
                c_love = vs2
            else:
                weight = (wavelength - h1) / (3 * h1)
                c_love = vs1 * (1 - weight) + vs2 * weight

            love_velocities.append(c_love)

        # Group velocities (simplified - typically 0.9 * phase velocity)
        rayleigh_group = [0.9 * c for c in rayleigh_velocities]
        love_group = [0.9 * c for c in love_velocities]

        # Penetration depth (approximately λ/3 for Rayleigh waves)
        penetration_depths = [T * c / 3 for T, c in zip(periods, rayleigh_velocities)]

        return {
            'periods': periods.tolist(),
            'rayleigh_phase_velocity': rayleigh_velocities,
            'rayleigh_group_velocity': rayleigh_group,
            'love_phase_velocity': love_velocities,
            'love_group_velocity': love_group,
            'penetration_depths': penetration_depths,
            'layer_model': layer_model
        }

    def _calculate_distance(self, lat1: float, lon1: float,
                          lat2: float, lon2: float) -> float:
        """Calculate great circle distance between two points"""
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        return self.earth_radius * c

    def _calculate_first_motion(self, strike: float, dip: float,
                               rake: float, takeoff: float) -> str:
        """Calculate P-wave first motion polarity"""
        # Simplified calculation
        # Actual calculation requires full moment tensor
        if rake > 0:
            if takeoff < 45:
                return "UP"
            else:
                return "DOWN"
        else:
            if takeoff < 45:
                return "DOWN"
            else:
                return "UP"

    def _pga_to_intensity(self, pga: float) -> float:
        """Convert PGA to Modified Mercalli Intensity"""
        # Wald et al. (1999) relations
        if pga < 0.0017:
            return 1
        elif pga < 0.014:
            return 2 + (pga - 0.0017) / 0.0123 * 2
        elif pga < 0.039:
            return 4 + (pga - 0.014) / 0.025 * 1
        elif pga < 0.092:
            return 5 + (pga - 0.039) / 0.053 * 1
        elif pga < 0.18:
            return 6 + (pga - 0.092) / 0.088 * 1
        elif pga < 0.34:
            return 7 + (pga - 0.18) / 0.16 * 1
        elif pga < 0.65:
            return 8 + (pga - 0.34) / 0.31 * 1
        elif pga < 1.24:
            return 9 + (pga - 0.65) / 0.59 * 1
        else:
            return min(10 + (pga - 1.24) / 0.5, 12)


def demo():
    """Demonstration of Seismology Lab capabilities"""
    print("Seismology Laboratory - Production Demo")
    print("=" * 60)

    lab = SeismologyLab()

    # Demo 1: P-Wave Propagation
    print("\n1. P-Wave Propagation Analysis:")
    source = EarthquakeSource(
        latitude=35.0,
        longitude=-118.0,
        depth=10.0,
        magnitude=6.5,
        strike=180,
        dip=90,
        rake=0,
        origin_time=0
    )

    station = SeismicStation(
        station_code='TEST',
        latitude=35.5,
        longitude=-117.5,
        elevation=1000,
        soil_class=SoilClass.C
    )

    p_wave = lab.p_wave_propagation(source, station)
    print(f"   Travel Time: {p_wave['travel_time']:.2f} seconds")
    print(f"   Distance: {p_wave['epicentral_distance']:.1f} km")
    print(f"   Velocity: {p_wave['velocity']:.2f} km/s")
    print(f"   First Motion: {p_wave['first_motion']}")

    # Demo 2: Magnitude Calculations
    print("\n2. Moment Magnitude Calculation:")
    moment = 1e18  # N·m
    mw = lab.moment_magnitude(moment)
    print(f"   Moment Magnitude (Mw): {mw['moment_magnitude']:.2f}")
    print(f"   Estimated Fault Length: {mw['estimated_fault_length_km']:.1f} km")
    print(f"   Energy Release: {mw['energy_tnt_equivalent']:.2f} kilotons TNT")

    # Demo 3: Site Amplification
    print("\n3. Site Amplification Effects:")
    frequencies = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    input_motion = np.ones(100)
    site_amp = lab.site_amplification(input_motion, frequencies, SoilClass.D)
    print(f"   Soil Class: {site_amp['soil_class']}")
    print(f"   Vs30: {site_amp['vs30']} m/s")
    print(f"   Fundamental Frequency: {site_amp['fundamental_frequency']:.2f} Hz")
    print(f"   PGA Amplification: {site_amp['amplification_factors']['pga']:.1f}x")

    # Demo 4: Ground Motion Prediction
    print("\n4. Ground Motion Prediction (GMPE):")
    gmpe = lab.ground_motion_prediction(
        magnitude=7.0,
        distance=50,
        site_class=SoilClass.C
    )
    print(f"   PGA: {gmpe['pga']:.3f} g")
    print(f"   PGV: {gmpe['pgv']:.1f} cm/s")
    print(f"   Duration: {gmpe['duration']:.1f} seconds")

    # Demo 5: Earthquake Early Warning
    print("\n5. Earthquake Early Warning:")
    eew = lab.earthquake_early_warning(
        p_arrival_time=10.0,
        s_arrival_time=18.0,
        p_amplitude=0.001,
        frequency=2.0
    )
    print(f"   Estimated Magnitude: {eew['estimated_magnitude']:.1f}")
    print(f"   Warning Time: {eew['warning_time']:.1f} seconds")
    print(f"   Alert Level: {eew['alert_level']}")
    print(f"   Action: {eew['recommended_action']}")

    print("\nLab ready for production use!")
    return lab


if __name__ == "__main__":
    demo()