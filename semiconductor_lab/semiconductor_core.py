# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Semiconductor Core Module
NIST-validated constants and scientifically accurate semiconductor physics
"""

import numpy as np
from scipy.constants import k as k_B, h, c, e, m_e, epsilon_0, pi
from scipy.special import erfc
from typing import Dict, List, Tuple, Optional


# Semiconductor constants
H_BAR = h / (2 * pi)
Q = e  # Elementary charge
KT_300K = k_B * 300  # Thermal energy at 300K


class TransistorPhysics:
    """
    MOSFET and BJT device physics
    I-V characteristics, threshold voltage, transconductance
    """

    def __init__(self):
        self.name = "Transistor Physics Simulator"

    def mosfet_iv_characteristic(self,
                                V_gs_array: np.ndarray,
                                V_ds_array: np.ndarray,
                                V_th: float,
                                mu_n: float,
                                C_ox: float,
                                W_L_ratio: float) -> Dict:
        """
        MOSFET I-V characteristics (long-channel)
        Linear: I_D = μ_n * C_ox * (W/L) * [(V_gs - V_th)*V_ds - V_ds²/2]
        Saturation: I_D = (μ_n * C_ox * W)/(2L) * (V_gs - V_th)²

        Args:
            V_gs_array: Gate-source voltage array (V)
            V_ds_array: Drain-source voltage array (V)
            V_th: Threshold voltage (V)
            mu_n: Electron mobility (cm²/V·s)
            C_ox: Oxide capacitance per unit area (F/cm²)
            W_L_ratio: Width/Length ratio

        Returns:
            Dictionary with I-V curves
        """
        # Convert mobility to SI (m²/V·s)
        mu_n_SI = mu_n * 1e-4
        C_ox_SI = C_ox * 1e4  # F/m²

        I_D_curves = []

        for V_gs in V_gs_array:
            I_D = np.zeros_like(V_ds_array)

            if V_gs <= V_th:
                # Subthreshold/cutoff
                I_D = np.zeros_like(V_ds_array)
            else:
                for i, V_ds in enumerate(V_ds_array):
                    V_ds_sat = V_gs - V_th

                    if V_ds < V_ds_sat:
                        # Linear/triode region
                        I_D[i] = mu_n_SI * C_ox_SI * W_L_ratio * ((V_gs - V_th) * V_ds - V_ds**2 / 2)
                    else:
                        # Saturation region
                        I_D[i] = (mu_n_SI * C_ox_SI * W_L_ratio / 2) * (V_gs - V_th)**2

            I_D_curves.append(I_D.tolist())

        # Transconductance at V_ds = V_dd (saturation)
        V_dd = V_ds_array[-1]
        g_m = mu_n_SI * C_ox_SI * W_L_ratio * (V_gs_array - V_th)
        g_m[V_gs_array <= V_th] = 0

        return {
            'V_gs_array': V_gs_array.tolist(),
            'V_ds_array': V_ds_array.tolist(),
            'I_D_curves_A': I_D_curves,
            'transconductance_S': g_m.tolist(),
            'V_th': V_th,
            'model': 'Long-Channel MOSFET'
        }

    def threshold_voltage_calculation(self,
                                     oxide_thickness_nm: float,
                                     substrate_doping_cm3: float,
                                     oxide_charge_cm2: float = 1e10,
                                     metal_work_function_eV: float = 4.5,
                                     temperature_K: float = 300) -> Dict:
        """
        Calculate MOSFET threshold voltage
        V_th = V_fb + 2*Φ_f + Q_dep/C_ox

        Args:
            oxide_thickness_nm: Gate oxide thickness (nm)
            substrate_doping_cm3: Substrate doping concentration (cm⁻³)
            oxide_charge_cm2: Fixed oxide charge (cm⁻²)
            metal_work_function_eV: Metal gate work function (eV)
            temperature_K: Temperature (K)

        Returns:
            Dictionary with threshold voltage components
        """
        t_ox = oxide_thickness_nm * 1e-9  # m
        N_A = substrate_doping_cm3 * 1e6  # m⁻³
        Q_ox = oxide_charge_cm2 * 1e4 * Q  # C/m²

        # Silicon parameters
        epsilon_si = 11.7 * epsilon_0
        epsilon_ox = 3.9 * epsilon_0
        n_i = 1.5e10 * 1e6  # Intrinsic carrier concentration (m⁻³) at 300K

        # Oxide capacitance
        C_ox = epsilon_ox / t_ox

        # Fermi potential
        phi_f = (k_B * temperature_K / Q) * np.log(N_A / n_i)

        # Flatband voltage
        phi_ms = metal_work_function_eV - (4.05 + 0.56)  # Si work function ~4.6eV
        V_fb = phi_ms - Q_ox / C_ox

        # Depletion charge
        Q_dep = -np.sqrt(2 * epsilon_si * Q * N_A * 2 * phi_f)

        # Threshold voltage
        V_th = V_fb + 2 * phi_f + abs(Q_dep) / C_ox

        return {
            'threshold_voltage_V': V_th,
            'flatband_voltage_V': V_fb,
            'fermi_potential_V': phi_f,
            'oxide_capacitance_F_per_m2': C_ox,
            'depletion_charge_C_per_m2': Q_dep,
            'oxide_thickness_nm': oxide_thickness_nm,
            'substrate_doping_cm3': substrate_doping_cm3,
            'model': 'MOSFET Threshold Voltage'
        }

    def bjt_ebers_moll(self,
                      V_be: float,
                      V_bc: float,
                      I_s: float = 1e-15,
                      beta_f: float = 100,
                      beta_r: float = 1,
                      temperature_K: float = 300) -> Dict:
        """
        Ebers-Moll BJT model
        I_C = I_s * (exp(V_be/V_t) - 1) - (I_s/β_r) * (exp(V_bc/V_t) - 1)

        Args:
            V_be: Base-emitter voltage (V)
            V_bc: Base-collector voltage (V)
            I_s: Saturation current (A)
            beta_f: Forward current gain
            beta_r: Reverse current gain
            temperature_K: Temperature (K)

        Returns:
            Dictionary with BJT currents
        """
        V_t = k_B * temperature_K / Q  # Thermal voltage

        # Forward and reverse currents
        I_f = I_s * (np.exp(V_be / V_t) - 1)
        I_r = I_s * (np.exp(V_bc / V_t) - 1)

        # Collector and emitter currents
        I_C = I_f - I_r / beta_r
        I_E = I_f / beta_f - I_r
        I_B = I_E - I_C

        # Operating region
        if V_be > 0.6 and V_bc < 0:
            region = "Forward Active"
        elif V_be > 0.6 and V_bc > 0:
            region = "Saturation"
        elif V_be < 0.6:
            region = "Cutoff"
        else:
            region = "Reverse Active"

        return {
            'collector_current_A': I_C,
            'emitter_current_A': I_E,
            'base_current_A': I_B,
            'current_gain_beta': I_C / I_B if abs(I_B) > 1e-20 else beta_f,
            'operating_region': region,
            'V_be': V_be,
            'V_bc': V_bc,
            'model': 'Ebers-Moll BJT'
        }


class BandStructure:
    """
    Electronic band structure calculations
    Band diagrams, density of states, effective mass
    """

    def __init__(self):
        self.name = "Band Structure Calculator"

    def intrinsic_carrier_concentration(self,
                                       temperature_K: np.ndarray,
                                       bandgap_eV: float = 1.12,
                                       m_e_star: float = 1.08,
                                       m_h_star: float = 0.81) -> Dict:
        """
        Calculate intrinsic carrier concentration
        n_i = sqrt(N_c * N_v) * exp(-E_g / (2kT))

        Args:
            temperature_K: Temperature array (K)
            bandgap_eV: Bandgap energy (eV) [Si: 1.12eV]
            m_e_star: Electron effective mass (m*/m_e)
            m_h_star: Hole effective mass (m*/m_e)

        Returns:
            Dictionary with carrier concentration
        """
        T = temperature_K
        E_g = bandgap_eV * Q  # Convert to Joules

        # Effective density of states
        N_c = 2 * (2 * pi * m_e_star * m_e * k_B * T / h**2)**(3/2)
        N_v = 2 * (2 * pi * m_h_star * m_e * k_B * T / h**2)**(3/2)

        # Intrinsic carrier concentration
        n_i = np.sqrt(N_c * N_v) * np.exp(-E_g / (2 * k_B * T))

        return {
            'temperature_K': T.tolist(),
            'intrinsic_carrier_concentration_per_m3': n_i.tolist(),
            'intrinsic_carrier_concentration_per_cm3': (n_i * 1e-6).tolist(),
            'bandgap_eV': bandgap_eV,
            'N_c_per_m3': N_c.tolist() if isinstance(N_c, np.ndarray) else [N_c],
            'N_v_per_m3': N_v.tolist() if isinstance(N_v, np.ndarray) else [N_v],
            'model': 'Intrinsic Carrier Concentration'
        }

    def pn_junction_built_in_potential(self,
                                      N_a_cm3: float,
                                      N_d_cm3: float,
                                      temperature_K: float = 300) -> Dict:
        """
        Calculate built-in potential of p-n junction
        V_bi = (kT/q) * ln(N_a * N_d / n_i²)

        Args:
            N_a_cm3: Acceptor doping (cm⁻³)
            N_d_cm3: Donor doping (cm⁻³)
            temperature_K: Temperature (K)

        Returns:
            Dictionary with junction parameters
        """
        N_a = N_a_cm3 * 1e6  # Convert to m⁻³
        N_d = N_d_cm3 * 1e6

        # Intrinsic carrier concentration (Si at 300K)
        n_i = 1.5e10 * 1e6  # m⁻³

        # Built-in potential
        V_t = k_B * temperature_K / Q
        V_bi = V_t * np.log(N_a * N_d / n_i**2)

        # Depletion width
        epsilon_si = 11.7 * epsilon_0
        W_dep = np.sqrt(2 * epsilon_si * V_bi / Q * (N_a + N_d) / (N_a * N_d))

        # Depletion widths on each side
        x_n = W_dep * N_a / (N_a + N_d)  # n-side
        x_p = W_dep * N_d / (N_a + N_d)  # p-side

        # Junction capacitance per unit area
        C_j = epsilon_si / W_dep

        return {
            'built_in_potential_V': V_bi,
            'depletion_width_um': W_dep * 1e6,
            'n_side_depletion_um': x_n * 1e6,
            'p_side_depletion_um': x_p * 1e6,
            'junction_capacitance_F_per_m2': C_j,
            'N_a_cm3': N_a_cm3,
            'N_d_cm3': N_d_cm3,
            'model': 'Abrupt p-n Junction'
        }

    def quantum_well_energy_levels(self,
                                   well_width_nm: float,
                                   barrier_height_eV: float,
                                   effective_mass_ratio: float = 0.067) -> Dict:
        """
        Energy levels in quantum well (infinite square well approximation)
        E_n = (n² * ℏ² * π²) / (2 * m* * L²)

        Args:
            well_width_nm: Well width (nm)
            barrier_height_eV: Barrier height (eV)
            effective_mass_ratio: Effective mass (m*/m_e)

        Returns:
            Dictionary with energy levels
        """
        L = well_width_nm * 1e-9  # m
        m_star = effective_mass_ratio * m_e

        # Energy levels (first 5)
        n_levels = 5
        E_n = []
        for n in range(1, n_levels + 1):
            E_J = (n**2 * H_BAR**2 * pi**2) / (2 * m_star * L**2)
            E_eV = E_J / Q
            E_n.append(E_eV)

        # Spacing between levels
        delta_E = E_n[1] - E_n[0] if len(E_n) > 1 else 0

        return {
            'energy_levels_eV': E_n,
            'ground_state_eV': E_n[0],
            'level_spacing_eV': delta_E,
            'well_width_nm': well_width_nm,
            'barrier_height_eV': barrier_height_eV,
            'model': 'Infinite Square Well Quantum Confinement'
        }


class DopingAnalysis:
    """
    Doping profile analysis
    Diffusion, ion implantation, carrier statistics
    """

    def __init__(self):
        self.name = "Doping Analysis"

    def gaussian_implantation_profile(self,
                                     depth_nm: np.ndarray,
                                     dose_cm2: float,
                                     projected_range_nm: float,
                                     straggle_nm: float) -> Dict:
        """
        Gaussian ion implantation profile
        N(x) = (Q / (sqrt(2π) * ΔR_p)) * exp(-(x - R_p)² / (2 * ΔR_p²))

        Args:
            depth_nm: Depth array (nm)
            dose_cm2: Implanted dose (ions/cm²)
            projected_range_nm: Projected range R_p (nm)
            straggle_nm: Range straggle ΔR_p (nm)

        Returns:
            Dictionary with doping profile
        """
        x = depth_nm
        Q_dose = dose_cm2
        R_p = projected_range_nm
        delta_R_p = straggle_nm

        # Gaussian distribution
        N_x = (Q_dose / (np.sqrt(2 * pi) * delta_R_p)) * np.exp(-(x - R_p)**2 / (2 * delta_R_p**2))

        # Peak concentration
        N_peak = Q_dose / (np.sqrt(2 * pi) * delta_R_p)

        # Junction depth (where N drops to background doping, assume 1e15 cm⁻³)
        N_background = 1e15
        junction_depth_nm = R_p + delta_R_p * np.sqrt(2 * np.log(N_peak / N_background))

        return {
            'depth_nm': x.tolist(),
            'concentration_cm3': N_x.tolist(),
            'peak_concentration_cm3': N_peak,
            'projected_range_nm': R_p,
            'straggle_nm': delta_R_p,
            'junction_depth_nm': junction_depth_nm,
            'model': 'Gaussian Ion Implantation'
        }

    def diffusion_profile(self,
                         depth_um: np.ndarray,
                         surface_concentration_cm3: float,
                         diffusion_time_hours: float,
                         diffusion_coefficient_cm2_per_s: float = 1e-13,
                         temperature_K: float = 1273) -> Dict:
        """
        Diffusion profile (complementary error function)
        N(x,t) = N_s * erfc(x / (2 * sqrt(D*t)))

        Args:
            depth_um: Depth array (μm)
            surface_concentration_cm3: Surface concentration (cm⁻³)
            diffusion_time_hours: Diffusion time (hours)
            diffusion_coefficient_cm2_per_s: Diffusion coefficient (cm²/s)
            temperature_K: Temperature (K)

        Returns:
            Dictionary with diffusion profile
        """
        x_cm = depth_um * 1e-4  # Convert to cm
        N_s = surface_concentration_cm3
        t_s = diffusion_time_hours * 3600  # Convert to seconds
        D = diffusion_coefficient_cm2_per_s

        # Complementary error function solution
        N_x = N_s * erfc(x_cm / (2 * np.sqrt(D * t_s)))

        # Diffusion length
        L_D = np.sqrt(D * t_s)

        # Junction depth (where N drops to background)
        N_background = 1e15
        x_j_cm = 2 * np.sqrt(D * t_s) * erfc(N_background / N_s)**(-1) if N_s > N_background else 0
        x_j_um = x_j_cm * 1e4

        return {
            'depth_um': depth_um.tolist(),
            'concentration_cm3': N_x.tolist(),
            'surface_concentration_cm3': N_s,
            'diffusion_length_um': L_D * 1e4,
            'junction_depth_um': x_j_um,
            'diffusion_coefficient_cm2_per_s': D,
            'model': 'Constant Source Diffusion (erfc)'
        }


class DeviceSimulation:
    """
    Device-level simulation
    SPICE parameters, power dissipation, frequency response
    """

    def __init__(self):
        self.name = "Device Simulation"

    def mosfet_small_signal_model(self,
                                  I_D: float,
                                  V_gs: float,
                                  V_th: float,
                                  mu_n: float,
                                  C_ox: float,
                                  W_L_ratio: float,
                                  C_gs: float,
                                  C_gd: float) -> Dict:
        """
        MOSFET small-signal parameters

        Args:
            I_D: Drain current (A)
            V_gs: Gate-source voltage (V)
            V_th: Threshold voltage (V)
            mu_n: Mobility (cm²/V·s)
            C_ox: Oxide capacitance (F/cm²)
            W_L_ratio: W/L ratio
            C_gs: Gate-source capacitance (F)
            C_gd: Gate-drain capacitance (F)

        Returns:
            Dictionary with small-signal parameters
        """
        # Convert units
        mu_n_SI = mu_n * 1e-4
        C_ox_SI = C_ox * 1e4

        # Transconductance
        if V_gs > V_th:
            g_m = mu_n_SI * C_ox_SI * W_L_ratio * (V_gs - V_th)
        else:
            g_m = 0

        # Output conductance (assume lambda = 0.01 V^-1)
        lambda_param = 0.01
        g_ds = lambda_param * I_D

        # Transit frequency
        f_T = g_m / (2 * pi * (C_gs + C_gd))

        # Maximum oscillation frequency
        f_max = f_T / (2 * np.sqrt(g_ds / g_m))

        return {
            'transconductance_S': g_m,
            'output_conductance_S': g_ds,
            'transit_frequency_Hz': f_T,
            'max_oscillation_frequency_Hz': f_max,
            'voltage_gain': g_m / g_ds,
            'input_capacitance_F': C_gs + C_gd,
            'model': 'MOSFET Small-Signal'
        }

    def power_dissipation(self,
                         supply_voltage_V: float,
                         switching_frequency_Hz: float,
                         load_capacitance_pF: float,
                         leakage_current_nA: float = 10) -> Dict:
        """
        CMOS power dissipation
        P = P_dynamic + P_static
        P_dynamic = C_L * V_dd² * f
        P_static = I_leak * V_dd

        Args:
            supply_voltage_V: Supply voltage (V)
            switching_frequency_Hz: Switching frequency (Hz)
            load_capacitance_pF: Load capacitance (pF)
            leakage_current_nA: Leakage current (nA)

        Returns:
            Dictionary with power analysis
        """
        V_dd = supply_voltage_V
        f = switching_frequency_Hz
        C_L = load_capacitance_pF * 1e-12  # Convert to F
        I_leak = leakage_current_nA * 1e-9  # Convert to A

        # Dynamic power
        P_dynamic = C_L * V_dd**2 * f

        # Static power
        P_static = I_leak * V_dd

        # Total power
        P_total = P_dynamic + P_static

        return {
            'total_power_W': P_total,
            'dynamic_power_W': P_dynamic,
            'static_power_W': P_static,
            'dynamic_energy_per_cycle_J': C_L * V_dd**2,
            'power_density_W_per_MHz': P_total / (f * 1e-6),
            'model': 'CMOS Power Dissipation'
        }
