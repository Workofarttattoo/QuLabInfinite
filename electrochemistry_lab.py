"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

ELECTROCHEMISTRY LAB
Production-ready electrochemistry algorithms and simulations.
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union, Callable
import math
from scipy.integrate import odeint, solve_ivp, cumtrapz
from scipy.optimize import minimize, curve_fit
from scipy.signal import find_peaks, savgol_filter
from scipy.constants import R, F, k, e
from enum import Enum

# Electrochemical constants
FARADAY_CONSTANT = 96485.3321  # C/mol
GAS_CONSTANT = 8.314462618  # J/(mol·K)
ELEMENTARY_CHARGE = 1.602176634e-19  # C

class ElectrochemicalTechnique(Enum):
    """Common electrochemical techniques"""
    CV = "Cyclic Voltammetry"
    LSV = "Linear Sweep Voltammetry"
    DPV = "Differential Pulse Voltammetry"
    SWV = "Square Wave Voltammetry"
    CA = "Chronoamperometry"
    CP = "Chronopotentiometry"
    EIS = "Electrochemical Impedance Spectroscopy"
    RDE = "Rotating Disk Electrode"
    RRDE = "Rotating Ring-Disk Electrode"

class ElectrodeType(Enum):
    """Common electrode materials"""
    PLATINUM = "Pt"
    GOLD = "Au"
    GLASSY_CARBON = "GC"
    CARBON_PASTE = "CP"
    MERCURY = "Hg"
    SILVER_SILVER_CHLORIDE = "Ag/AgCl"
    SATURATED_CALOMEL = "SCE"
    SCREEN_PRINTED = "SPE"
    BORON_DOPED_DIAMOND = "BDD"

@dataclass
class ElectrochemicalCell:
    """Electrochemical cell configuration"""
    working_electrode: ElectrodeType
    reference_electrode: ElectrodeType
    counter_electrode: ElectrodeType
    temperature: float = 298.15  # K
    area: float = 0.0707  # cm² (3mm diameter disk)
    solution_resistance: float = 100  # Ohm
    double_layer_capacitance: float = 20e-6  # F/cm²

@dataclass
class RedoxCouple:
    """Redox couple properties"""
    name: str
    E0: float  # Formal potential (V)
    n: int  # Number of electrons
    D_ox: float  # Diffusion coefficient of oxidized form (cm²/s)
    D_red: float  # Diffusion coefficient of reduced form (cm²/s)
    k0: float  # Standard rate constant (cm/s)
    alpha: float = 0.5  # Transfer coefficient
    concentration: float = 1e-3  # mol/cm³

@dataclass
class Voltammogram:
    """Voltammetric data"""
    potential: np.ndarray  # V
    current: np.ndarray  # A
    time: np.ndarray = None
    scan_rate: float = 0.1  # V/s
    technique: ElectrochemicalTechnique = ElectrochemicalTechnique.CV

class ElectrochemistryLab:
    """Production-ready electrochemistry laboratory"""

    def __init__(self, cell: ElectrochemicalCell = None):
        if cell is None:
            cell = ElectrochemicalCell(
                working_electrode=ElectrodeType.GLASSY_CARBON,
                reference_electrode=ElectrodeType.SILVER_SILVER_CHLORIDE,
                counter_electrode=ElectrodeType.PLATINUM
            )
        self.cell = cell
        self.F = FARADAY_CONSTANT
        self.R = GAS_CONSTANT
        self.T = cell.temperature

    # === NERNST EQUATION AND THERMODYNAMICS ===

    def nernst_potential(self, E0: float, n: int,
                        activity_ox: float, activity_red: float) -> float:
        """Calculate Nernst potential"""
        # E = E0 + (RT/nF) * ln(a_ox/a_red)
        if activity_red > 0:
            E = E0 + (self.R * self.T) / (n * self.F) * np.log(activity_ox / activity_red)
        else:
            E = E0
        return E

    def calculate_gibbs_energy(self, E_cell: float, n: int) -> float:
        """Calculate Gibbs free energy from cell potential"""
        # ΔG = -nFE
        delta_G = -n * self.F * E_cell
        return delta_G  # J/mol

    def calculate_equilibrium_constant(self, E0_cell: float, n: int) -> float:
        """Calculate equilibrium constant from standard potential"""
        # K = exp(nFE°/RT)
        K = np.exp(n * self.F * E0_cell / (self.R * self.T))
        return K

    def pourbaix_diagram_point(self, pH: float, E0: float,
                              n_electrons: int, n_protons: int) -> float:
        """Calculate potential for Pourbaix diagram at given pH"""
        # E = E0 - (2.303RT/F) * (n_H+/n_e-) * pH
        E = E0 - (2.303 * self.R * self.T / self.F) * (n_protons / n_electrons) * pH
        return E

    # === BUTLER-VOLMER KINETICS ===

    def butler_volmer_current(self, E: float, redox: RedoxCouple) -> float:
        """Calculate current using Butler-Volmer equation"""
        # i = nFA k0 [C_ox * exp(-αnfη) - C_red * exp((1-α)nfη)]
        # where f = F/RT and η = E - E_eq

        f = self.F / (self.R * self.T)
        eta = E - redox.E0  # Overpotential

        # Exchange current density
        i0 = redox.n * self.F * self.cell.area * redox.k0 * redox.concentration

        # Butler-Volmer equation
        i = i0 * (np.exp(-redox.alpha * redox.n * f * eta) -
                  np.exp((1 - redox.alpha) * redox.n * f * eta))

        return i

    def tafel_analysis(self, overpotential: np.ndarray,
                       current: np.ndarray) -> Dict:
        """Perform Tafel analysis to extract kinetic parameters"""
        # Tafel equation: η = a + b*log(i)
        # where b = 2.303RT/(αnF) is Tafel slope

        # Take log of absolute current
        log_i = np.log10(np.abs(current))

        # Remove invalid values
        valid = np.isfinite(log_i) & np.isfinite(overpotential)
        log_i = log_i[valid]
        eta = overpotential[valid]

        if len(log_i) < 2:
            return {'tafel_slope': 0, 'exchange_current': 0, 'alpha': 0.5}

        # Linear fit in Tafel region (typically η > 50 mV)
        tafel_region = np.abs(eta) > 0.05
        if np.sum(tafel_region) > 2:
            coeffs = np.polyfit(log_i[tafel_region], eta[tafel_region], 1)
            tafel_slope = coeffs[0]  # V/decade
            log_i0 = -coeffs[1] / coeffs[0]
            i0 = 10 ** log_i0

            # Calculate transfer coefficient
            # b = 2.303RT/(αnF)
            alpha = 2.303 * self.R * self.T / (abs(tafel_slope) * self.F)
        else:
            tafel_slope = 0.120  # Default 120 mV/decade
            i0 = 1e-6
            alpha = 0.5

        return {
            'tafel_slope': tafel_slope,
            'exchange_current': i0,
            'alpha': alpha,
            'corrosion_rate': i0 / (self.F * self.cell.area)  # mol/(cm²·s)
        }

    # === CYCLIC VOLTAMMETRY ===

    def simulate_cyclic_voltammetry(self, redox: RedoxCouple,
                                   E_initial: float, E_vertex1: float,
                                   E_vertex2: float, scan_rate: float,
                                   n_cycles: int = 1) -> Voltammogram:
        """Simulate cyclic voltammogram for reversible system"""

        # Create potential waveform
        n_points = int(abs(E_vertex1 - E_initial) / (scan_rate * 0.001))  # 1 mV steps
        E_forward = np.linspace(E_initial, E_vertex1, n_points)
        E_reverse = np.linspace(E_vertex1, E_vertex2, n_points)

        potential = [E_forward]
        for cycle in range(n_cycles):
            if cycle == 0:
                potential.append(E_reverse)
            else:
                potential.append(np.linspace(E_vertex2, E_vertex1, n_points))
                potential.append(E_reverse)

        potential = np.concatenate(potential)
        time = np.arange(len(potential)) * 0.001 / scan_rate

        # Calculate current using Randles-Sevcik equation for reversible system
        current = []
        for E in potential:
            # Simplified model - full simulation would solve diffusion equations
            # ip = 0.4463 * n * F * A * C * sqrt(n * F * v * D / RT)

            # Distance from E0
            delta_E = E - redox.E0

            # Peak current (Randles-Sevcik)
            ip = 0.4463 * redox.n * self.F * self.cell.area * redox.concentration * \
                 np.sqrt(redox.n * self.F * scan_rate * redox.D_ox / (self.R * self.T))

            # Current shape (approximation)
            f = self.F / (self.R * self.T)
            i = ip * np.exp(-redox.n * f * abs(delta_E)) * np.sign(-delta_E)

            # Add capacitive current
            i += self.cell.double_layer_capacitance * self.cell.area * scan_rate

            current.append(i)

        current = np.array(current)

        return Voltammogram(
            potential=potential,
            current=current,
            time=time,
            scan_rate=scan_rate,
            technique=ElectrochemicalTechnique.CV
        )

    def analyze_cv_peaks(self, voltammogram: Voltammogram) -> Dict:
        """Extract peak parameters from cyclic voltammogram"""

        # Find peaks
        anodic_peaks, _ = find_peaks(voltammogram.current, height=0)
        cathodic_peaks, _ = find_peaks(-voltammogram.current, height=0)

        analysis = {
            'E_pa': [],  # Anodic peak potentials
            'E_pc': [],  # Cathodic peak potentials
            'i_pa': [],  # Anodic peak currents
            'i_pc': [],  # Cathodic peak currents
            'E_1/2': [],  # Half-wave potentials
            'delta_E_p': [],  # Peak separation
            'reversibility': []
        }

        # Extract peak parameters
        for peak in anodic_peaks:
            analysis['E_pa'].append(voltammogram.potential[peak])
            analysis['i_pa'].append(voltammogram.current[peak])

        for peak in cathodic_peaks:
            analysis['E_pc'].append(voltammogram.potential[peak])
            analysis['i_pc'].append(-voltammogram.current[peak])

        # Calculate derived parameters
        if len(analysis['E_pa']) > 0 and len(analysis['E_pc']) > 0:
            # Take first peaks
            E_pa = analysis['E_pa'][0]
            E_pc = analysis['E_pc'][0]
            i_pa = analysis['i_pa'][0]
            i_pc = analysis['i_pc'][0]

            analysis['E_1/2'] = [(E_pa + E_pc) / 2]
            analysis['delta_E_p'] = [abs(E_pa - E_pc)]

            # Assess reversibility
            # For reversible system at 25°C: ΔE_p = 59/n mV
            if analysis['delta_E_p'][0] < 0.070:  # < 70 mV
                analysis['reversibility'] = ['reversible']
            elif analysis['delta_E_p'][0] < 0.200:  # < 200 mV
                analysis['reversibility'] = ['quasi-reversible']
            else:
                analysis['reversibility'] = ['irreversible']

            # Peak current ratio
            analysis['i_ratio'] = abs(i_pc / i_pa) if i_pa != 0 else 0

        return analysis

    def randles_sevcik_analysis(self, peak_currents: np.ndarray,
                               scan_rates: np.ndarray,
                               n_electrons: int = 1) -> Dict:
        """Analyze scan rate dependence using Randles-Sevcik equation"""
        # ip = 0.4463 * n^(3/2) * F^(3/2) * A * C * D^(1/2) * v^(1/2) / (RT)^(1/2)

        # Plot ip vs v^(1/2)
        sqrt_v = np.sqrt(scan_rates)

        # Linear fit
        coeffs = np.polyfit(sqrt_v, peak_currents, 1)
        slope = coeffs[0]
        intercept = coeffs[1]

        # Calculate diffusion coefficient
        # slope = 0.4463 * n^(3/2) * F^(3/2) * A * C * D^(1/2) / (RT)^(1/2)
        const = 0.4463 * n_electrons ** 1.5 * self.F ** 1.5 / np.sqrt(self.R * self.T)

        # Assume concentration = 1 mM
        concentration = 1e-6  # mol/cm³
        D = (slope / (const * self.cell.area * concentration)) ** 2

        # R-squared
        y_fit = np.polyval(coeffs, sqrt_v)
        ss_res = np.sum((peak_currents - y_fit) ** 2)
        ss_tot = np.sum((peak_currents - np.mean(peak_currents)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            'diffusion_coefficient': D,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'process_type': 'diffusion-controlled' if abs(intercept) < abs(slope) * 0.1 else 'mixed'
        }

    # === IMPEDANCE SPECTROSCOPY ===

    def simulate_eis_spectrum(self, frequencies: np.ndarray,
                             R_s: float, R_ct: float, C_dl: float,
                             W_coefficient: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate electrochemical impedance spectrum"""
        # Randles circuit: Rs + (Rct || Cdl) + W

        omega = 2 * np.pi * frequencies
        Z_real = []
        Z_imag = []

        for w in omega:
            # Double layer capacitance impedance
            Z_cdl = 1 / (1j * w * C_dl)

            # Charge transfer resistance
            Z_ct = R_ct

            # Parallel combination of Rct and Cdl
            Z_parallel = (Z_ct * Z_cdl) / (Z_ct + Z_cdl)

            # Warburg impedance (semi-infinite diffusion)
            if W_coefficient is not None:
                Z_w = W_coefficient / np.sqrt(w) - 1j * W_coefficient / np.sqrt(w)
            else:
                Z_w = 0

            # Total impedance
            Z_total = R_s + Z_parallel + Z_w

            Z_real.append(Z_total.real)
            Z_imag.append(-Z_total.imag)  # Convention: negative imaginary

        return np.array(Z_real), np.array(Z_imag)

    def fit_randles_circuit(self, Z_real: np.ndarray, Z_imag: np.ndarray,
                           frequencies: np.ndarray) -> Dict:
        """Fit impedance data to Randles equivalent circuit"""

        def randles_model(freq, R_s, R_ct, C_dl, W):
            Z_r, Z_i = self.simulate_eis_spectrum(freq, R_s, R_ct, C_dl, W)
            return np.concatenate([Z_r, Z_i])

        # Initial guesses
        R_s_guess = np.min(Z_real)
        R_ct_guess = np.max(Z_real) - np.min(Z_real)
        C_dl_guess = 1e-5
        W_guess = 100

        # Combine real and imaginary parts for fitting
        Z_data = np.concatenate([Z_real, Z_imag])

        try:
            popt, pcov = curve_fit(randles_model, frequencies, Z_data,
                                  p0=[R_s_guess, R_ct_guess, C_dl_guess, W_guess],
                                  bounds=([0, 0, 1e-9, 0], [1e6, 1e6, 1e-3, 1e6]))

            R_s, R_ct, C_dl, W = popt

            # Calculate fit quality
            Z_fit = randles_model(frequencies, *popt)
            residuals = Z_data - Z_fit
            chi_squared = np.sum(residuals ** 2) / len(residuals)

            return {
                'R_solution': R_s,
                'R_charge_transfer': R_ct,
                'C_double_layer': C_dl,
                'Warburg_coefficient': W,
                'chi_squared': chi_squared,
                'time_constant': R_ct * C_dl
            }
        except:
            return {
                'R_solution': R_s_guess,
                'R_charge_transfer': R_ct_guess,
                'C_double_layer': C_dl_guess,
                'Warburg_coefficient': W_guess,
                'chi_squared': float('inf'),
                'time_constant': R_ct_guess * C_dl_guess
            }

    def calculate_exchange_current_from_eis(self, R_ct: float, n: int = 1) -> float:
        """Calculate exchange current density from charge transfer resistance"""
        # i0 = RT / (nF * Rct * A)
        i0 = self.R * self.T / (n * self.F * R_ct * self.cell.area)
        return i0

    # === CHRONOAMPEROMETRY ===

    def cottrell_equation(self, time: np.ndarray, n: int,
                         concentration: float, D: float) -> np.ndarray:
        """Calculate current using Cottrell equation"""
        # i(t) = nFAC√(D/πt)
        # Avoid division by zero at t=0
        time_safe = np.maximum(time, 1e-6)
        current = n * self.F * self.cell.area * concentration * \
                 np.sqrt(D / (np.pi * time_safe))
        return current

    def simulate_chronoamperometry(self, E_step: float, redox: RedoxCouple,
                                  duration: float, n_points: int = 1000) -> Voltammogram:
        """Simulate chronoamperometric response"""

        time = np.linspace(0, duration, n_points)
        time[0] = 1e-6  # Avoid t=0

        # Cottrell current (diffusion-limited)
        i_cottrell = self.cottrell_equation(time, redox.n,
                                           redox.concentration, redox.D_ox)

        # Add capacitive current (exponential decay)
        tau = self.cell.solution_resistance * self.cell.double_layer_capacitance * self.cell.area
        i_capacitive = (E_step / self.cell.solution_resistance) * np.exp(-time / tau)

        # Total current
        current = i_cottrell + i_capacitive

        # Constant potential
        potential = np.full_like(time, E_step)

        return Voltammogram(
            potential=potential,
            current=current,
            time=time,
            technique=ElectrochemicalTechnique.CA
        )

    def analyze_cottrell_plot(self, time: np.ndarray, current: np.ndarray,
                             n_electrons: int = 1,
                             concentration: float = 1e-6) -> Dict:
        """Analyze Cottrell plot to extract diffusion coefficient"""

        # Cottrell equation: i = nFAC√(D/πt)
        # Plot i vs 1/√t should be linear

        # Remove initial points (capacitive contribution)
        mask = time > 0.1  # After 100 ms
        time_fit = time[mask]
        current_fit = current[mask]

        if len(time_fit) < 2:
            return {'diffusion_coefficient': 0, 'r_squared': 0}

        # Transform variables
        x = 1 / np.sqrt(time_fit)
        y = current_fit

        # Linear fit
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]

        # Extract diffusion coefficient
        # slope = nFAC√(D/π)
        D = (slope / (n_electrons * self.F * self.cell.area * concentration)) ** 2 * np.pi

        # Calculate R²
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            'diffusion_coefficient': D,
            'slope': slope,
            'intercept': coeffs[1],
            'r_squared': r_squared
        }

    # === BATTERY MODELING ===

    def battery_capacity(self, n_electrons: int, mass_active: float,
                        molecular_weight: float) -> float:
        """Calculate theoretical battery capacity"""
        # Q = nF * (m/M)
        # Returns capacity in mAh/g
        capacity_C = n_electrons * self.F * (1 / molecular_weight)  # C/g
        capacity_mAh = capacity_C / 3.6  # mAh/g
        return capacity_mAh

    def battery_energy_density(self, voltage: float, capacity: float) -> float:
        """Calculate energy density"""
        # E = V * Q
        # capacity in mAh/g, voltage in V
        # Returns Wh/kg
        energy = voltage * capacity  # mWh/g = Wh/kg
        return energy

    def simulate_battery_discharge(self, capacity: float, current: float,
                                  internal_resistance: float,
                                  nominal_voltage: float,
                                  cutoff_voltage: float) -> Dict:
        """Simulate battery discharge curve"""

        # Time to discharge
        time_hours = capacity / (current * 1000)  # capacity in mAh, current in A
        time = np.linspace(0, time_hours, 100)

        # State of charge
        soc = 1 - time / time_hours

        # Voltage vs SOC (simplified linear + exponential)
        voltage = nominal_voltage * (0.9 + 0.1 * soc) * np.exp(-2 * (1 - soc))

        # IR drop
        voltage -= current * internal_resistance

        # Find cutoff point
        cutoff_idx = np.where(voltage <= cutoff_voltage)[0]
        if len(cutoff_idx) > 0:
            actual_capacity = capacity * (cutoff_idx[0] / len(time))
            actual_time = time[cutoff_idx[0]]
        else:
            actual_capacity = capacity
            actual_time = time_hours

        return {
            'time': time[:cutoff_idx[0]] if len(cutoff_idx) > 0 else time,
            'voltage': voltage[:cutoff_idx[0]] if len(cutoff_idx) > 0 else voltage,
            'soc': soc[:cutoff_idx[0]] if len(cutoff_idx) > 0 else soc,
            'actual_capacity': actual_capacity,
            'discharge_time': actual_time,
            'energy': actual_capacity * np.mean(voltage[:cutoff_idx[0]] if len(cutoff_idx) > 0 else voltage)
        }

    def battery_internal_resistance(self, ocv: float, load_voltage: float,
                                   current: float) -> float:
        """Calculate internal resistance from load test"""
        if current > 0:
            R_internal = (ocv - load_voltage) / current
        else:
            R_internal = 0
        return R_internal

    # === CORROSION ===

    def corrosion_rate(self, i_corr: float, n: int,
                      molecular_weight: float, density: float) -> Dict:
        """Calculate corrosion rate from corrosion current"""
        # Using Faraday's law

        # Mass loss rate (g/s)
        mass_rate = (i_corr * molecular_weight) / (n * self.F)

        # Penetration rate (mm/year)
        # 1 year = 31536000 s
        penetration_rate = (mass_rate * 31536000 * 10) / (self.cell.area * density)

        # Corrosion rate in mpy (mils per year)
        mpy = penetration_rate * 39.37  # mm to mils

        return {
            'mass_loss_rate': mass_rate * 1000,  # mg/s
            'penetration_rate': penetration_rate,  # mm/year
            'corrosion_rate_mpy': mpy,  # mils/year
            'corrosion_current': i_corr  # A
        }

    def evans_diagram(self, E_range: np.ndarray,
                     metal_redox: RedoxCouple,
                     oxidant_redox: RedoxCouple) -> Dict:
        """Generate Evans diagram for corrosion analysis"""

        # Calculate currents for each half-reaction
        i_metal = []
        i_oxidant = []

        for E in E_range:
            i_m = self.butler_volmer_current(E, metal_redox)
            i_o = self.butler_volmer_current(E, oxidant_redox)
            i_metal.append(i_m)
            i_oxidant.append(-i_o)  # Opposite sign for reduction

        i_metal = np.array(i_metal)
        i_oxidant = np.array(i_oxidant)

        # Find intersection (corrosion potential and current)
        diff = abs(i_metal + i_oxidant)
        corr_idx = np.argmin(diff)
        E_corr = E_range[corr_idx]
        i_corr = abs(i_metal[corr_idx])

        return {
            'E_corrosion': E_corr,
            'i_corrosion': i_corr,
            'metal_current': i_metal,
            'oxidant_current': i_oxidant,
            'potential_range': E_range
        }

    def passivation_analysis(self, potential: np.ndarray,
                           current: np.ndarray) -> Dict:
        """Analyze passivation behavior from polarization curve"""

        # Find critical passivation potential (peak current)
        peaks, _ = find_peaks(current)

        if len(peaks) > 0:
            i_critical = current[peaks[0]]
            E_critical = potential[peaks[0]]

            # Find passive region (minimum current after peak)
            passive_idx = peaks[0] + np.argmin(current[peaks[0]:])
            i_passive = current[passive_idx]
            E_passive = potential[passive_idx]

            # Find transpassive potential (current increases again)
            trans_idx = passive_idx + np.where(current[passive_idx:] > 2 * i_passive)[0]
            if len(trans_idx) > 0:
                E_trans = potential[passive_idx + trans_idx[0]]
            else:
                E_trans = potential[-1]

            passive_range = E_trans - E_passive

        else:
            # No passivation observed
            i_critical = np.max(current)
            E_critical = potential[np.argmax(current)]
            i_passive = i_critical
            E_passive = E_critical
            E_trans = potential[-1]
            passive_range = 0

        return {
            'critical_current': i_critical,
            'critical_potential': E_critical,
            'passive_current': i_passive,
            'passive_potential': E_passive,
            'transpassive_potential': E_trans,
            'passive_range': passive_range,
            'passivation_observed': len(peaks) > 0
        }

def run_demo():
    """Demonstrate electrochemistry lab capabilities"""
    print("=" * 80)
    print("ELECTROCHEMISTRY LAB - Production Demo")
    print("Copyright (c) 2025 Joshua Hendricks Cole")
    print("=" * 80)

    # Initialize lab
    cell = ElectrochemicalCell(
        working_electrode=ElectrodeType.GLASSY_CARBON,
        reference_electrode=ElectrodeType.SILVER_SILVER_CHLORIDE,
        counter_electrode=ElectrodeType.PLATINUM
    )
    lab = ElectrochemistryLab(cell)

    # 1. Nernst equation
    print("\n1. Nernst equation and thermodynamics...")
    E0 = 0.77  # Fe³⁺/Fe²⁺
    E_nernst = lab.nernst_potential(E0, n=1, activity_ox=0.01, activity_red=0.1)
    print(f"   Nernst potential: {E_nernst:.3f} V")

    delta_G = lab.calculate_gibbs_energy(E_nernst, n=1)
    print(f"   Gibbs energy: {delta_G/1000:.1f} kJ/mol")

    K_eq = lab.calculate_equilibrium_constant(E0, n=1)
    print(f"   Equilibrium constant: {K_eq:.2e}")

    # 2. Butler-Volmer kinetics
    print("\n2. Butler-Volmer kinetics...")
    redox = RedoxCouple(
        name="Fe(CN)6³⁻/⁴⁻",
        E0=0.36,
        n=1,
        D_ox=7.6e-6,
        D_red=6.3e-6,
        k0=0.01,
        concentration=5e-6
    )

    E_test = 0.4  # 40 mV overpotential
    i_bv = lab.butler_volmer_current(E_test, redox)
    print(f"   Current at {E_test}V: {i_bv*1e6:.2f} μA")

    # 3. Cyclic voltammetry simulation
    print("\n3. Simulating cyclic voltammetry...")
    cv = lab.simulate_cyclic_voltammetry(
        redox=redox,
        E_initial=0.0,
        E_vertex1=0.6,
        E_vertex2=0.0,
        scan_rate=0.1,  # V/s
        n_cycles=1
    )

    # Analyze CV
    cv_analysis = lab.analyze_cv_peaks(cv)
    if len(cv_analysis['E_pa']) > 0:
        print(f"   E_pa: {cv_analysis['E_pa'][0]:.3f} V")
        print(f"   E_pc: {cv_analysis['E_pc'][0]:.3f} V")
        print(f"   ΔE_p: {cv_analysis['delta_E_p'][0]*1000:.1f} mV")
        print(f"   Reversibility: {cv_analysis['reversibility'][0]}")

    # 4. Scan rate analysis
    print("\n4. Randles-Sevcik analysis...")
    scan_rates = np.array([0.01, 0.025, 0.05, 0.1, 0.2, 0.5])  # V/s
    peak_currents = np.array([5.2, 8.1, 11.5, 16.3, 23.0, 36.4]) * 1e-6  # A

    rs_analysis = lab.randles_sevcik_analysis(peak_currents, scan_rates, n_electrons=1)
    print(f"   Diffusion coefficient: {rs_analysis['diffusion_coefficient']*1e6:.1f} × 10⁻⁶ cm²/s")
    print(f"   R²: {rs_analysis['r_squared']:.4f}")
    print(f"   Process: {rs_analysis['process_type']}")

    # 5. Impedance spectroscopy
    print("\n5. Electrochemical impedance spectroscopy...")
    frequencies = np.logspace(-1, 5, 50)  # 0.1 Hz to 100 kHz
    Z_real, Z_imag = lab.simulate_eis_spectrum(
        frequencies=frequencies,
        R_s=100,  # Ω
        R_ct=1000,  # Ω
        C_dl=20e-6,  # F
        W_coefficient=500
    )

    # Fit circuit
    fit = lab.fit_randles_circuit(Z_real, Z_imag, frequencies)
    print(f"   R_solution: {fit['R_solution']:.1f} Ω")
    print(f"   R_charge_transfer: {fit['R_charge_transfer']:.1f} Ω")
    print(f"   C_double_layer: {fit['C_double_layer']*1e6:.1f} μF")
    print(f"   Time constant: {fit['time_constant']*1e3:.2f} ms")

    # 6. Chronoamperometry
    print("\n6. Chronoamperometry simulation...")
    ca = lab.simulate_chronoamperometry(
        E_step=0.5,
        redox=redox,
        duration=10.0  # seconds
    )

    # Analyze Cottrell behavior
    cottrell = lab.analyze_cottrell_plot(
        ca.time[10:],  # Skip initial points
        ca.current[10:],
        n_electrons=1,
        concentration=redox.concentration
    )
    print(f"   Diffusion coefficient: {cottrell['diffusion_coefficient']*1e6:.1f} × 10⁻⁶ cm²/s")
    print(f"   R²: {cottrell['r_squared']:.3f}")

    # 7. Tafel analysis
    print("\n7. Tafel analysis...")
    overpotentials = np.linspace(-0.2, 0.2, 50)
    currents = [lab.butler_volmer_current(redox.E0 + eta, redox) for eta in overpotentials]

    tafel = lab.tafel_analysis(overpotentials, np.array(currents))
    print(f"   Tafel slope: {tafel['tafel_slope']*1000:.1f} mV/decade")
    print(f"   Exchange current: {tafel['exchange_current']*1e6:.2f} μA")
    print(f"   Transfer coefficient: {tafel['alpha']:.2f}")

    # 8. Battery calculations
    print("\n8. Battery modeling...")

    # Li-ion battery example
    capacity = lab.battery_capacity(n_electrons=1, mass_active=1.0,
                                   molecular_weight=97.0)  # LiCoO2
    print(f"   Theoretical capacity: {capacity:.1f} mAh/g")

    energy = lab.battery_energy_density(voltage=3.7, capacity=capacity)
    print(f"   Energy density: {energy:.1f} Wh/kg")

    # Discharge simulation
    discharge = lab.simulate_battery_discharge(
        capacity=2000,  # mAh
        current=1.0,  # A
        internal_resistance=0.05,  # Ω
        nominal_voltage=3.7,  # V
        cutoff_voltage=3.0  # V
    )
    print(f"   Actual capacity: {discharge['actual_capacity']:.0f} mAh")
    print(f"   Discharge time: {discharge['discharge_time']:.2f} hours")

    # 9. Corrosion analysis
    print("\n9. Corrosion analysis...")

    # Iron corrosion example
    i_corr = 1e-5  # A
    corrosion = lab.corrosion_rate(
        i_corr=i_corr,
        n=2,  # Fe → Fe²⁺ + 2e⁻
        molecular_weight=55.845,
        density=7.87  # g/cm³
    )
    print(f"   Corrosion current: {i_corr*1e6:.1f} μA")
    print(f"   Mass loss rate: {corrosion['mass_loss_rate']:.3f} mg/s")
    print(f"   Penetration rate: {corrosion['penetration_rate']:.2f} mm/year")
    print(f"   Corrosion rate: {corrosion['corrosion_rate_mpy']:.1f} mpy")

    # 10. Pourbaix diagram point
    print("\n10. Pourbaix diagram...")
    E_pourbaix = lab.pourbaix_diagram_point(
        pH=7.0,
        E0=1.23,  # O2/H2O
        n_electrons=4,
        n_protons=4
    )
    print(f"   E at pH 7: {E_pourbaix:.3f} V")

    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("Visit: https://aios.is | https://thegavl.com")
    print("=" * 80)

if __name__ == '__main__':
    run_demo()