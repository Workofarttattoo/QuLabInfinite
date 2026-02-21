"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

ELECTRICAL ENGINEERING LAB
Production-ready electrical engineering calculations for circuit analysis, filters, control systems, and semiconductors.
Free gift to the scientific community from QuLabInfinite.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy import constants, signal
from typing import Dict, Tuple, List, Optional, Complex
import cmath

@dataclass
class ElectricalEngineeringLab:
    """
    Comprehensive electrical engineering calculations including:
    - AC/DC circuit analysis
    - Filter design (Butterworth, Chebyshev, Elliptic)
    - Power calculations (real, reactive, apparent)
    - Control system stability analysis
    - Signal processing basics
    - Semiconductor modeling
    """

    # Standard values
    frequency: float = 60.0  # Hz (standard power frequency)
    temperature: float = 300  # K (room temperature)
    boltzmann: float = constants.k  # Boltzmann constant
    electron_charge: float = constants.e  # Elementary charge

    # Component tolerances
    resistor_tolerance: float = 0.05  # 5% tolerance
    capacitor_tolerance: float = 0.10  # 10% tolerance

    def __post_init__(self):
        """Initialize derived constants"""
        self.omega = 2 * np.pi * self.frequency  # Angular frequency
        self.thermal_voltage = self.boltzmann * self.temperature / self.electron_charge  # ~26mV at room temp

    def ohms_law(self, voltage: Optional[float] = None,
                 current: Optional[float] = None,
                 resistance: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate missing value using Ohm's Law: V = I * R

        Args:
            voltage: Voltage in volts
            current: Current in amperes
            resistance: Resistance in ohms

        Returns:
            Dictionary with all three values
        """
        if voltage is None:
            if current is None or resistance is None:
                raise ValueError("Need two values to calculate the third")
            voltage = current * resistance
        elif current is None:
            if voltage is None or resistance is None:
                raise ValueError("Need two values to calculate the third")
            current = voltage / resistance
        elif resistance is None:
            if voltage is None or current is None:
                raise ValueError("Need two values to calculate the third")
            resistance = voltage / current

        return {
            'voltage': voltage,
            'current': current,
            'resistance': resistance,
            'power': voltage * current
        }

    def series_resistance(self, resistances: List[float]) -> float:
        """Calculate total resistance of series resistors: R_total = R1 + R2 + ..."""
        return sum(resistances)

    def parallel_resistance(self, resistances: List[float]) -> float:
        """Calculate total resistance of parallel resistors: 1/R_total = 1/R1 + 1/R2 + ..."""
        if not resistances or any(r <= 0 for r in resistances):
            raise ValueError("All resistances must be positive")
        return 1 / sum(1/r for r in resistances)

    def voltage_divider(self, v_in: float, r1: float, r2: float) -> float:
        """
        Calculate output voltage of voltage divider: V_out = V_in * R2 / (R1 + R2)

        Args:
            v_in: Input voltage in volts
            r1: First resistor (connected to V_in) in ohms
            r2: Second resistor (connected to ground) in ohms

        Returns:
            Output voltage in volts
        """
        return v_in * r2 / (r1 + r2)

    def current_divider(self, i_total: float, r1: float, r2: float) -> Tuple[float, float]:
        """
        Calculate branch currents in current divider

        Args:
            i_total: Total current in amperes
            r1: First branch resistance in ohms
            r2: Second branch resistance in ohms

        Returns:
            Tuple of (i1, i2) branch currents
        """
        r_total = self.parallel_resistance([r1, r2])
        i1 = i_total * r_total / r1
        i2 = i_total * r_total / r2
        return i1, i2

    def ac_impedance(self, resistance: float, inductance: float = 0,
                    capacitance: float = 0, frequency: Optional[float] = None) -> complex:
        """
        Calculate complex impedance: Z = R + j(XL - XC)

        Args:
            resistance: Resistance in ohms
            inductance: Inductance in henries
            capacitance: Capacitance in farads
            frequency: Frequency in Hz (default: self.frequency)

        Returns:
            Complex impedance in ohms
        """
        if frequency is None:
            frequency = self.frequency

        omega = 2 * np.pi * frequency

        # Inductive reactance: XL = ωL
        x_l = omega * inductance if inductance > 0 else 0

        # Capacitive reactance: XC = 1/(ωC)
        x_c = 1 / (omega * capacitance) if capacitance > 0 else 0

        # Total impedance
        return complex(resistance, x_l - x_c)

    def ac_power(self, voltage: float, current: float, phase_angle: float) -> Dict[str, float]:
        """
        Calculate AC power components

        Args:
            voltage: RMS voltage in volts
            current: RMS current in amperes
            phase_angle: Phase angle in radians

        Returns:
            Dictionary with real, reactive, apparent power and power factor
        """
        apparent_power = voltage * current  # S in VA
        real_power = apparent_power * np.cos(phase_angle)  # P in W
        reactive_power = apparent_power * np.sin(phase_angle)  # Q in VAR
        power_factor = np.cos(phase_angle)

        return {
            'real_power': real_power,  # W
            'reactive_power': reactive_power,  # VAR
            'apparent_power': apparent_power,  # VA
            'power_factor': power_factor,
            'phase_angle_deg': np.degrees(phase_angle)
        }

    def three_phase_power(self, voltage_line: float, current_line: float,
                         phase_angle: float, configuration: str = 'wye') -> Dict[str, float]:
        """
        Calculate three-phase power

        Args:
            voltage_line: Line voltage in volts
            current_line: Line current in amperes
            phase_angle: Phase angle in radians
            configuration: 'wye' or 'delta'

        Returns:
            Dictionary with power calculations
        """
        if configuration == 'wye':
            voltage_phase = voltage_line / np.sqrt(3)
            current_phase = current_line
        elif configuration == 'delta':
            voltage_phase = voltage_line
            current_phase = current_line / np.sqrt(3)
        else:
            raise ValueError("Configuration must be 'wye' or 'delta'")

        # Total three-phase power
        total_real_power = np.sqrt(3) * voltage_line * current_line * np.cos(phase_angle)
        total_reactive_power = np.sqrt(3) * voltage_line * current_line * np.sin(phase_angle)
        total_apparent_power = np.sqrt(3) * voltage_line * current_line

        return {
            'voltage_phase': voltage_phase,
            'current_phase': current_phase,
            'total_real_power': total_real_power,  # W
            'total_reactive_power': total_reactive_power,  # VAR
            'total_apparent_power': total_apparent_power,  # VA
            'power_factor': np.cos(phase_angle),
            'configuration': configuration
        }

    def rc_time_constant(self, resistance: float, capacitance: float) -> float:
        """Calculate RC circuit time constant: τ = R * C"""
        return resistance * capacitance

    def rl_time_constant(self, resistance: float, inductance: float) -> float:
        """Calculate RL circuit time constant: τ = L / R"""
        if resistance <= 0:
            raise ValueError("Resistance must be positive")
        return inductance / resistance

    def resonant_frequency(self, inductance: float, capacitance: float) -> float:
        """
        Calculate resonant frequency of LC circuit: f_0 = 1 / (2π√(LC))

        Args:
            inductance: Inductance in henries
            capacitance: Capacitance in farads

        Returns:
            Resonant frequency in Hz
        """
        return 1 / (2 * np.pi * np.sqrt(inductance * capacitance))

    def quality_factor(self, resistance: float, inductance: float,
                      capacitance: float) -> float:
        """
        Calculate Q factor of RLC circuit: Q = (1/R) * √(L/C)

        Args:
            resistance: Resistance in ohms
            inductance: Inductance in henries
            capacitance: Capacitance in farads

        Returns:
            Quality factor (dimensionless)
        """
        if resistance <= 0:
            return float('inf')
        return (1 / resistance) * np.sqrt(inductance / capacitance)

    def butterworth_filter(self, order: int, cutoff_freq: float,
                          filter_type: str = 'low') -> Tuple[np.ndarray, np.ndarray]:
        """
        Design Butterworth filter

        Args:
            order: Filter order
            cutoff_freq: Cutoff frequency in Hz
            filter_type: 'low', 'high', 'band', or 'stop'

        Returns:
            Tuple of (numerator coefficients, denominator coefficients)
        """
        nyquist = 0.5 * 1000  # Assuming 1kHz sampling rate
        normalized_cutoff = cutoff_freq / nyquist

        if filter_type in ['low', 'high']:
            b, a = signal.butter(order, normalized_cutoff, btype=filter_type, analog=False)
        elif filter_type == 'band':
            # For bandpass, use a range around cutoff
            low = max(0.01, normalized_cutoff - 0.1)
            high = min(0.99, normalized_cutoff + 0.1)
            b, a = signal.butter(order, [low, high], btype='band', analog=False)
        elif filter_type == 'stop':
            # For bandstop, use a range around cutoff
            low = max(0.01, normalized_cutoff - 0.1)
            high = min(0.99, normalized_cutoff + 0.1)
            b, a = signal.butter(order, [low, high], btype='bandstop', analog=False)
        else:
            raise ValueError("filter_type must be 'low', 'high', 'band', or 'stop'")

        return b, a

    def transfer_function_poles_zeros(self, numerator: List[float],
                                     denominator: List[float]) -> Dict[str, np.ndarray]:
        """
        Find poles and zeros of transfer function

        Args:
            numerator: Numerator polynomial coefficients
            denominator: Denominator polynomial coefficients

        Returns:
            Dictionary with poles, zeros, and stability status
        """
        zeros = np.roots(numerator)
        poles = np.roots(denominator)

        # System is stable if all poles have negative real parts
        is_stable = all(np.real(pole) < 0 for pole in poles)

        return {
            'zeros': zeros,
            'poles': poles,
            'is_stable': is_stable,
            'damping_ratios': -np.real(poles) / np.abs(poles) if len(poles) > 0 else []
        }

    def pid_controller(self, error: float, error_integral: float,
                      error_derivative: float, kp: float = 1.0,
                      ki: float = 0.1, kd: float = 0.01) -> float:
        """
        Calculate PID controller output

        Args:
            error: Current error signal
            error_integral: Integral of error over time
            error_derivative: Derivative of error
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain

        Returns:
            Controller output
        """
        return kp * error + ki * error_integral + kd * error_derivative

    def bode_plot_data(self, numerator: List[float], denominator: List[float],
                      frequencies: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Generate data for Bode plot

        Args:
            numerator: Transfer function numerator coefficients
            denominator: Transfer function denominator coefficients
            frequencies: Frequency array in Hz (default: logspace from 0.1 to 10kHz)

        Returns:
            Dictionary with frequency, magnitude (dB), and phase (degrees)
        """
        if frequencies is None:
            frequencies = np.logspace(-1, 4, 1000)  # 0.1 Hz to 10 kHz

        # Convert to angular frequency
        omega = 2 * np.pi * frequencies

        # Create transfer function
        system = signal.TransferFunction(numerator, denominator)

        # Calculate frequency response
        w, h = signal.freqresp(system, omega)

        # Convert to magnitude in dB and phase in degrees
        magnitude_db = 20 * np.log10(np.abs(h))
        phase_deg = np.degrees(np.angle(h))

        return {
            'frequencies': frequencies,
            'magnitude_db': magnitude_db,
            'phase_deg': phase_deg,
            'gain_margin': self._calculate_gain_margin(magnitude_db, phase_deg),
            'phase_margin': self._calculate_phase_margin(magnitude_db, phase_deg)
        }

    def _calculate_gain_margin(self, magnitude_db: np.ndarray,
                               phase_deg: np.ndarray) -> float:
        """Calculate gain margin from frequency response"""
        # Find where phase crosses -180 degrees
        phase_crossings = np.where(np.diff(np.sign(phase_deg + 180)))[0]

        if len(phase_crossings) > 0:
            # Gain margin is negative of magnitude at phase crossing
            return -magnitude_db[phase_crossings[0]]
        else:
            return float('inf')

    def _calculate_phase_margin(self, magnitude_db: np.ndarray,
                                phase_deg: np.ndarray) -> float:
        """Calculate phase margin from frequency response"""
        # Find where magnitude crosses 0 dB
        gain_crossings = np.where(np.diff(np.sign(magnitude_db)))[0]

        if len(gain_crossings) > 0:
            # Phase margin is 180 + phase at gain crossing
            return 180 + phase_deg[gain_crossings[0]]
        else:
            return 180.0

    def diode_current(self, voltage: float, saturation_current: float = 1e-12,
                     ideality_factor: float = 1.0) -> float:
        """
        Calculate diode current using Shockley equation

        I = Is * (e^(V/(n*Vt)) - 1)

        Args:
            voltage: Applied voltage in volts
            saturation_current: Saturation current in amperes
            ideality_factor: Ideality factor (1-2)

        Returns:
            Diode current in amperes
        """
        vt = self.thermal_voltage

        if voltage > 0.7:  # Forward bias limit to prevent overflow
            return saturation_current * np.exp(0.7 / (ideality_factor * vt))

        return saturation_current * (np.exp(voltage / (ideality_factor * vt)) - 1)

    def transistor_amplifier(self, vbe: float, beta: float = 100,
                           vcc: float = 12, rc: float = 1000,
                           re: float = 100) -> Dict[str, float]:
        """
        Calculate common-emitter amplifier parameters

        Args:
            vbe: Base-emitter voltage in volts
            beta: Current gain (hFE)
            vcc: Collector supply voltage in volts
            rc: Collector resistance in ohms
            re: Emitter resistance in ohms

        Returns:
            Dictionary with amplifier parameters
        """
        # Simplified analysis
        vt = self.thermal_voltage

        # Base current (simplified)
        ib = (vbe - 0.7) / (beta * re) if vbe > 0.7 else 0

        # Collector current
        ic = beta * ib

        # Emitter current
        ie = ic + ib

        # Voltages
        vc = vcc - ic * rc
        ve = ie * re
        vce = vc - ve

        # Small-signal parameters
        gm = ic / vt if ic > 0 else 0  # Transconductance
        r_in = beta * vt / ic if ic > 0 else float('inf')  # Input resistance
        av = -gm * rc  # Voltage gain

        return {
            'base_current': ib,
            'collector_current': ic,
            'emitter_current': ie,
            'collector_voltage': vc,
            'emitter_voltage': ve,
            'vce': vce,
            'transconductance': gm,
            'input_resistance': r_in,
            'voltage_gain': av
        }

    def op_amp_gain(self, rf: float, r1: float, config: str = 'inverting') -> float:
        """
        Calculate operational amplifier gain

        Args:
            rf: Feedback resistance in ohms
            r1: Input resistance in ohms
            config: 'inverting' or 'non_inverting'

        Returns:
            Voltage gain
        """
        if config == 'inverting':
            return -rf / r1
        elif config == 'non_inverting':
            return 1 + rf / r1
        else:
            raise ValueError("config must be 'inverting' or 'non_inverting'")

    def transmission_line_impedance(self, inductance_per_length: float,
                                   capacitance_per_length: float) -> float:
        """
        Calculate characteristic impedance of transmission line

        Z0 = √(L/C)

        Args:
            inductance_per_length: Inductance per unit length in H/m
            capacitance_per_length: Capacitance per unit length in F/m

        Returns:
            Characteristic impedance in ohms
        """
        return np.sqrt(inductance_per_length / capacitance_per_length)

    def signal_to_noise_ratio(self, signal_power: float, noise_power: float) -> float:
        """
        Calculate signal-to-noise ratio in dB

        SNR = 10 * log10(P_signal / P_noise)

        Args:
            signal_power: Signal power in watts
            noise_power: Noise power in watts

        Returns:
            SNR in decibels
        """
        if noise_power <= 0:
            return float('inf')
        return 10 * np.log10(signal_power / noise_power)

    def fourier_series_coefficients(self, signal_func, period: float,
                                   n_harmonics: int = 10) -> Dict[str, np.ndarray]:
        """
        Calculate Fourier series coefficients

        Args:
            signal_func: Function representing the signal
            period: Signal period in seconds
            n_harmonics: Number of harmonics to calculate

        Returns:
            Dictionary with DC component and harmonic coefficients
        """
        t = np.linspace(0, period, 1000)
        signal = signal_func(t)

        # DC component
        a0 = (2 / period) * np.trapz(signal, t)

        # Harmonic coefficients
        an = np.zeros(n_harmonics)
        bn = np.zeros(n_harmonics)

        for n in range(1, n_harmonics + 1):
            an[n-1] = (2 / period) * np.trapz(signal * np.cos(2 * np.pi * n * t / period), t)
            bn[n-1] = (2 / period) * np.trapz(signal * np.sin(2 * np.pi * n * t / period), t)

        # Magnitude and phase
        magnitude = np.sqrt(an**2 + bn**2)
        phase = np.arctan2(bn, an)

        return {
            'dc_component': a0 / 2,
            'cosine_coefficients': an,
            'sine_coefficients': bn,
            'magnitude': magnitude,
            'phase_rad': phase,
            'phase_deg': np.degrees(phase)
        }


def run_demo():
    """Demonstrate electrical engineering calculations"""
    print("=" * 70)
    print("ELECTRICAL ENGINEERING LAB - Production Demo")
    print("=" * 70)

    lab = ElectricalEngineeringLab()

    # Basic Circuit Analysis
    print("\n1. BASIC CIRCUIT ANALYSIS")
    print("-" * 40)
    result = lab.ohms_law(voltage=12, resistance=100)
    print(f"12V across 100Ω:")
    print(f"  Current: {result['current']:.3f} A")
    print(f"  Power: {result['power']:.2f} W")

    v_out = lab.voltage_divider(12, 1000, 1000)
    print(f"Voltage divider (12V, 1kΩ, 1kΩ): {v_out:.2f} V")

    # AC Circuit Analysis
    print("\n2. AC CIRCUIT ANALYSIS")
    print("-" * 40)
    z = lab.ac_impedance(100, 0.1, 10e-6, 60)
    print(f"RLC impedance at 60Hz:")
    print(f"  Magnitude: {abs(z):.2f} Ω")
    print(f"  Phase: {np.degrees(cmath.phase(z)):.2f}°")

    ac_power = lab.ac_power(120, 10, np.radians(30))
    print(f"AC Power (120V, 10A, 30° phase):")
    print(f"  Real Power: {ac_power['real_power']:.2f} W")
    print(f"  Power Factor: {ac_power['power_factor']:.3f}")

    # Three-Phase Power
    print("\n3. THREE-PHASE POWER")
    print("-" * 40)
    three_phase = lab.three_phase_power(480, 50, np.radians(25), 'wye')
    print(f"Three-phase (480V, 50A, Wye):")
    print(f"  Total Power: {three_phase['total_real_power']/1000:.2f} kW")
    print(f"  Power Factor: {three_phase['power_factor']:.3f}")

    # Filter Design
    print("\n4. FILTER DESIGN")
    print("-" * 40)
    b, a = lab.butterworth_filter(4, 100, 'low')
    print(f"4th order Butterworth lowpass (100Hz cutoff):")
    print(f"  Numerator coeffs: {len(b)} terms")
    print(f"  Denominator coeffs: {len(a)} terms")

    f_res = lab.resonant_frequency(0.1, 10e-6)
    print(f"LC Resonant frequency (0.1H, 10μF): {f_res:.2f} Hz")

    # Control Systems
    print("\n5. CONTROL SYSTEM ANALYSIS")
    print("-" * 40)
    tf_analysis = lab.transfer_function_poles_zeros([1, 0], [1, 5, 6])
    print(f"Transfer function s/(s² + 5s + 6):")
    print(f"  Poles: {tf_analysis['poles']}")
    print(f"  System stable: {tf_analysis['is_stable']}")

    pid_output = lab.pid_controller(0.5, 0.1, 0.01)
    print(f"PID controller output: {pid_output:.3f}")

    # Semiconductor Devices
    print("\n6. SEMICONDUCTOR MODELING")
    print("-" * 40)
    i_diode = lab.diode_current(0.65)
    print(f"Diode current at 0.65V: {i_diode*1e6:.2f} μA")

    amp = lab.transistor_amplifier(0.75, beta=150, vcc=15)
    print(f"CE Amplifier (Vbe=0.75V, β=150):")
    print(f"  Ic: {amp['collector_current']*1e3:.2f} mA")
    print(f"  Voltage gain: {amp['voltage_gain']:.1f}")

    op_gain = lab.op_amp_gain(10000, 1000, 'inverting')
    print(f"Op-amp gain (10kΩ/1kΩ inverting): {op_gain}")

    print("\n" + "=" * 70)
    print("Demo completed successfully!")


if __name__ == '__main__':
    run_demo()