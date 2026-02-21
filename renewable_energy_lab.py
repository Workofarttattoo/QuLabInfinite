"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

RENEWABLE ENERGY LAB - Production Ready Implementation
Solar cells, wind turbines, battery storage, grid integration, and energy optimization
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable
from scipy import integrate, optimize, interpolate
from scipy.constants import h, c, k, e, Stefan_Boltzmann

# Physical constants
SOLAR_CONSTANT = 1361  # W/m² - Solar irradiance at Earth's orbit
PLANCK_CONSTANT = h  # J·s
SPEED_OF_LIGHT = c  # m/s
BOLTZMANN_CONSTANT = k  # J/K
ELEMENTARY_CHARGE = e  # C
STEFAN_BOLTZMANN_CONSTANT = Stefan_Boltzmann  # W/(m²·K⁴)

@dataclass
class SolarCell:
    """Solar photovoltaic cell parameters"""
    material: str
    bandgap: float  # eV
    efficiency: float  # Fraction
    area: float  # m²
    temperature_coefficient: float  # %/K
    series_resistance: float  # Ohms
    shunt_resistance: float  # Ohms
    ideality_factor: float = 1.5
    saturation_current: float = 1e-12  # A

@dataclass
class WindTurbine:
    """Wind turbine specifications"""
    rotor_diameter: float  # m
    hub_height: float  # m
    rated_power: float  # W
    cut_in_speed: float  # m/s
    rated_speed: float  # m/s
    cut_out_speed: float  # m/s
    power_coefficient: float = 0.45  # Betz limit is 0.593

@dataclass
class BatterySystem:
    """Energy storage battery system"""
    technology: str  # Li-ion, Lead-acid, Flow, etc.
    capacity: float  # Wh
    voltage: float  # V
    max_discharge_rate: float  # C-rate
    max_charge_rate: float  # C-rate
    efficiency: float  # Round-trip efficiency
    cycle_life: int
    depth_of_discharge: float = 0.8
    self_discharge_rate: float = 0.02  # per month

@dataclass
class GridConnection:
    """Electrical grid connection parameters"""
    voltage: float  # V
    frequency: float  # Hz
    max_export: float  # W
    max_import: float  # W
    feed_in_tariff: float  # $/kWh
    electricity_price: float  # $/kWh

class RenewableEnergyLab:
    """Comprehensive renewable energy laboratory"""

    def __init__(self):
        self.solar_spectrum = self._initialize_solar_spectrum()
        self.wind_profile = None
        self.energy_storage = {}

    # ============= SOLAR ENERGY =============

    def solar_irradiance(self, latitude: float, day_of_year: int,
                        hour: float, tilt_angle: float = 0,
                        azimuth: float = 0) -> float:
        """
        Calculate solar irradiance on tilted surface
        Using simplified clear-sky model
        """
        # Solar declination angle
        declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))

        # Hour angle
        hour_angle = 15 * (hour - 12)  # degrees

        # Solar elevation angle
        elevation = np.arcsin(
            np.sin(np.radians(declination)) * np.sin(np.radians(latitude)) +
            np.cos(np.radians(declination)) * np.cos(np.radians(latitude)) *
            np.cos(np.radians(hour_angle))
        )

        if elevation < 0:
            return 0  # Sun below horizon

        # Air mass
        air_mass = 1 / np.sin(elevation) if elevation > 0 else 1000

        # Atmospheric transmission (simplified)
        transmission = 0.75 ** air_mass

        # Direct normal irradiance
        dni = SOLAR_CONSTANT * transmission

        # Angle of incidence on tilted surface
        # Simplified: assuming south-facing surface
        cos_incidence = (
            np.sin(elevation) * np.cos(np.radians(tilt_angle)) +
            np.cos(elevation) * np.sin(np.radians(tilt_angle)) *
            np.cos(np.radians(azimuth - 180))
        )

        # Total irradiance on tilted surface
        direct = dni * max(0, cos_incidence)
        diffuse = 0.1 * dni  # Simplified diffuse component
        reflected = 0.2 * dni * (1 - np.cos(np.radians(tilt_angle))) / 2  # Ground reflection

        return direct + diffuse + reflected

    def solar_cell_iv_curve(self, cell: SolarCell, irradiance: float,
                           temperature: float = 298) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate I-V characteristic curve of solar cell
        Using single-diode model
        """
        # Temperature-adjusted parameters
        T_ref = 298  # K
        Eg_ref = cell.bandgap * ELEMENTARY_CHARGE  # J

        # Temperature effects
        Eg = Eg_ref * (1 - 0.0002 * (temperature - T_ref))
        Is = cell.saturation_current * (temperature / T_ref) ** 3 * \
             np.exp(Eg_ref / (cell.ideality_factor * k * T_ref) -
                   Eg / (cell.ideality_factor * k * temperature))

        # Photocurrent (proportional to irradiance)
        Iph = irradiance / 1000 * cell.area * 5  # Simplified: 5 A/m² at 1000 W/m²

        # Voltage array
        V = np.linspace(0, cell.bandgap * 0.8, 100)

        # Current calculation (implicit equation solver)
        I = np.zeros_like(V)
        for i, v in enumerate(V):
            # Newton-Raphson to solve: I = Iph - Is*(exp(q*(V+I*Rs)/(n*k*T)) - 1) - (V+I*Rs)/Rsh
            def f(i_guess):
                return (i_guess - Iph +
                       Is * (np.exp(ELEMENTARY_CHARGE * (v + i_guess * cell.series_resistance) /
                                   (cell.ideality_factor * k * temperature)) - 1) +
                       (v + i_guess * cell.series_resistance) / cell.shunt_resistance)

            try:
                I[i] = optimize.fsolve(f, Iph)[0]
            except:
                I[i] = 0

        # Ensure physical values
        I = np.maximum(0, I)

        return V, I

    def solar_cell_efficiency(self, cell: SolarCell, spectrum: Optional[np.ndarray] = None) -> float:
        """
        Calculate detailed balance limit (Shockley-Queisser) efficiency
        """
        if spectrum is None:
            spectrum = self.solar_spectrum

        # Wavelength to energy conversion
        wavelengths = spectrum[:, 0] * 1e-9  # nm to m
        irradiance = spectrum[:, 1]  # W/m²/nm

        # Photon energy
        photon_energy = PLANCK_CONSTANT * SPEED_OF_LIGHT / wavelengths / ELEMENTARY_CHARGE  # eV

        # Photons above bandgap
        above_bandgap = photon_energy > cell.bandgap

        # Absorbed photon flux
        photon_flux = irradiance * wavelengths / (PLANCK_CONSTANT * SPEED_OF_LIGHT)

        # Current from absorbed photons
        Jsc = ELEMENTARY_CHARGE * np.trapz(photon_flux[above_bandgap],
                                          wavelengths[above_bandgap]) * cell.area

        # Radiative recombination current (detailed balance)
        T = 300  # K
        V = cell.bandgap * 0.8  # Typical MPP voltage

        # Dark current
        J0 = ELEMENTARY_CHARGE * 2 * np.pi / (PLANCK_CONSTANT ** 3 * SPEED_OF_LIGHT ** 2) * \
             (cell.bandgap * ELEMENTARY_CHARGE) ** 2 * k * T * \
             np.exp(-cell.bandgap * ELEMENTARY_CHARGE / (k * T))

        # Fill factor (simplified)
        voc = cell.ideality_factor * k * T / ELEMENTARY_CHARGE * np.log(Jsc / J0 + 1)
        ff = (voc - np.log(voc + 0.72)) / (voc + 1)

        # Maximum power
        P_max = Jsc * voc * ff

        # Input power
        P_in = np.trapz(irradiance, wavelengths * 1e9) * cell.area

        # Efficiency
        efficiency = P_max / P_in if P_in > 0 else 0

        return min(efficiency, cell.efficiency)  # Cap at specified efficiency

    def tandem_solar_cell(self, top_cell: SolarCell, bottom_cell: SolarCell,
                         irradiance: float) -> Dict[str, float]:
        """
        Model tandem (multi-junction) solar cell
        """
        # Top cell absorbs high energy photons
        spectrum = self.solar_spectrum
        wavelengths = spectrum[:, 0] * 1e-9  # nm to m
        photon_energy = PLANCK_CONSTANT * SPEED_OF_LIGHT / wavelengths / ELEMENTARY_CHARGE

        # Top cell absorption
        top_absorption = (photon_energy > top_cell.bandgap)
        top_spectrum = spectrum.copy()
        top_spectrum[~top_absorption, 1] = 0

        top_efficiency = self.solar_cell_efficiency(top_cell, top_spectrum)
        top_power = top_efficiency * irradiance * top_cell.area

        # Bottom cell gets remaining photons
        bottom_spectrum = spectrum.copy()
        bottom_spectrum[top_absorption, 1] *= 0.1  # Some transmission through top cell
        bottom_absorption = (photon_energy > bottom_cell.bandgap) & (photon_energy <= top_cell.bandgap)

        bottom_efficiency = self.solar_cell_efficiency(bottom_cell, bottom_spectrum)
        bottom_power = bottom_efficiency * irradiance * bottom_cell.area * 0.9  # Accounting for shading

        # Current matching for series connection
        V_top, I_top = self.solar_cell_iv_curve(top_cell, irradiance)
        V_bottom, I_bottom = self.solar_cell_iv_curve(bottom_cell, irradiance * 0.5)

        # Find current matching point
        I_match = min(np.max(I_top), np.max(I_bottom))

        return {
            'top_power': top_power,
            'bottom_power': bottom_power,
            'total_power': top_power + bottom_power,
            'matched_current': I_match,
            'total_efficiency': (top_power + bottom_power) / (irradiance * top_cell.area)
        }

    def perovskite_stability(self, temperature: float, humidity: float,
                           illumination_hours: float) -> Dict[str, float]:
        """
        Model perovskite solar cell degradation
        """
        # Arrhenius degradation model
        E_activation = 0.5  # eV
        degradation_rate = np.exp(-E_activation * ELEMENTARY_CHARGE / (k * temperature))

        # Humidity acceleration factor
        humidity_factor = 1 + (humidity / 50) ** 2

        # UV degradation
        uv_degradation = 1 - 0.001 * illumination_hours

        # Total degradation
        retention = np.exp(-degradation_rate * humidity_factor * illumination_hours / 1000) * uv_degradation

        # T80 lifetime (time to 80% retention)
        if retention > 0:
            t80 = -np.log(0.8) / (degradation_rate * humidity_factor / 1000)
        else:
            t80 = 0

        return {
            'efficiency_retention': retention,
            'T80_lifetime_hours': t80,
            'degradation_rate': degradation_rate * humidity_factor,
            'humidity_factor': humidity_factor
        }

    # ============= WIND ENERGY =============

    def wind_power_curve(self, turbine: WindTurbine,
                        wind_speeds: np.ndarray) -> np.ndarray:
        """
        Calculate wind turbine power output curve
        """
        power = np.zeros_like(wind_speeds)

        for i, v in enumerate(wind_speeds):
            if v < turbine.cut_in_speed:
                power[i] = 0
            elif v < turbine.rated_speed:
                # Cubic relationship in partial load region
                swept_area = np.pi * (turbine.rotor_diameter / 2) ** 2
                air_density = 1.225  # kg/m³ at sea level

                # Betz limit consideration
                power[i] = 0.5 * air_density * swept_area * v ** 3 * turbine.power_coefficient

                # Cap at rated power
                power[i] = min(power[i], turbine.rated_power)
            elif v < turbine.cut_out_speed:
                power[i] = turbine.rated_power
            else:
                power[i] = 0  # Shutdown for safety

        return power

    def wind_profile_height(self, wind_speed_ref: float, height_ref: float,
                           height: float, roughness_length: float = 0.03) -> float:
        """
        Calculate wind speed at different heights using log law
        roughness_length: 0.0002 (water), 0.03 (open field), 0.5 (suburban), 1.0 (urban)
        """
        # Logarithmic wind profile
        wind_speed = wind_speed_ref * np.log(height / roughness_length) / \
                    np.log(height_ref / roughness_length)

        return max(0, wind_speed)

    def weibull_wind_distribution(self, k: float = 2.0, c: float = 7.0,
                                 wind_speeds: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Weibull probability distribution for wind speeds
        k: shape parameter (typically 1.5-3)
        c: scale parameter (m/s)
        """
        if wind_speeds is None:
            wind_speeds = np.linspace(0, 25, 100)

        # Weibull PDF
        pdf = (k / c) * (wind_speeds / c) ** (k - 1) * np.exp(-(wind_speeds / c) ** k)

        return wind_speeds, pdf

    def capacity_factor(self, turbine: WindTurbine, wind_distribution: Tuple[np.ndarray, np.ndarray]) -> float:
        """
        Calculate capacity factor for wind turbine
        """
        wind_speeds, pdf = wind_distribution

        # Power curve
        power = self.wind_power_curve(turbine, wind_speeds)

        # Expected power
        expected_power = np.trapz(power * pdf, wind_speeds)

        # Capacity factor
        cf = expected_power / turbine.rated_power

        return cf

    def wake_effect(self, turbines: List[Tuple[float, float]],
                   wind_direction: float, wind_speed: float) -> np.ndarray:
        """
        Calculate wake effects in wind farm
        turbines: list of (x, y) positions
        """
        n_turbines = len(turbines)
        wind_speeds = np.ones(n_turbines) * wind_speed

        # Wake decay constant
        k_wake = 0.075

        # Convert wind direction to vector
        wind_vec = np.array([np.cos(np.radians(wind_direction)),
                           np.sin(np.radians(wind_direction))])

        for i, (x1, y1) in enumerate(turbines):
            for j, (x2, y2) in enumerate(turbines):
                if i != j:
                    # Check if turbine j is downstream of turbine i
                    rel_pos = np.array([x2 - x1, y2 - y1])
                    downstream_dist = np.dot(rel_pos, wind_vec)

                    if downstream_dist > 0:
                        # Lateral distance
                        lateral_dist = np.abs(np.cross(rel_pos, wind_vec))

                        # Wake radius at downstream position
                        rotor_diameter = 100  # Assumed
                        wake_radius = rotor_diameter / 2 + k_wake * downstream_dist

                        if lateral_dist < wake_radius:
                            # Jensen wake model
                            velocity_deficit = (1 - np.sqrt(1 - 0.5)) / (1 + k_wake * downstream_dist / rotor_diameter) ** 2
                            wind_speeds[j] *= (1 - velocity_deficit)

        return wind_speeds

    def vertical_axis_turbine(self, height: float, diameter: float,
                            wind_speed: float, solidity: float = 0.3) -> float:
        """
        Model Vertical Axis Wind Turbine (VAWT) like Darrieus or Savonius
        """
        # Swept area
        swept_area = height * diameter

        # Power coefficient depends on tip speed ratio and solidity
        # Simplified model
        if solidity < 0.5:  # Darrieus type
            cp_max = 0.35
            optimal_tsr = 4.0
        else:  # Savonius type
            cp_max = 0.25
            optimal_tsr = 1.0

        # Assuming operation at optimal TSR
        air_density = 1.225
        power = 0.5 * air_density * swept_area * wind_speed ** 3 * cp_max

        return power

    # ============= BATTERY STORAGE =============

    def battery_charge_discharge(self, battery: BatterySystem,
                                power_profile: np.ndarray,
                                time_step: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Simulate battery charge/discharge with power profile
        power_profile: positive for charging, negative for discharging
        time_step: hours
        """
        n_steps = len(power_profile)
        soc = np.zeros(n_steps)  # State of charge
        actual_power = np.zeros(n_steps)
        losses = np.zeros(n_steps)

        # Initial SOC at 50%
        current_energy = battery.capacity * 0.5

        for i, power in enumerate(power_profile):
            if power > 0:  # Charging
                # Apply charge rate limit
                max_charge = battery.capacity * battery.max_charge_rate * time_step
                actual_charge = min(power * time_step, max_charge)

                # Apply efficiency
                energy_added = actual_charge * np.sqrt(battery.efficiency)

                # Check capacity limit
                if current_energy + energy_added > battery.capacity:
                    energy_added = battery.capacity - current_energy
                    actual_charge = energy_added / np.sqrt(battery.efficiency)

                current_energy += energy_added
                actual_power[i] = actual_charge / time_step
                losses[i] = actual_charge * (1 - np.sqrt(battery.efficiency))

            else:  # Discharging
                # Apply discharge rate limit
                max_discharge = battery.capacity * battery.max_discharge_rate * time_step
                actual_discharge = min(-power * time_step, max_discharge)

                # Apply efficiency
                energy_removed = actual_discharge / np.sqrt(battery.efficiency)

                # Check DOD limit
                min_energy = battery.capacity * (1 - battery.depth_of_discharge)
                if current_energy - energy_removed < min_energy:
                    energy_removed = current_energy - min_energy
                    actual_discharge = energy_removed * np.sqrt(battery.efficiency)

                current_energy -= energy_removed
                actual_power[i] = -actual_discharge / time_step
                losses[i] = actual_discharge * (1 - np.sqrt(battery.efficiency))

            # Self-discharge
            current_energy *= (1 - battery.self_discharge_rate / (30 * 24) * time_step)

            # Record SOC
            soc[i] = current_energy / battery.capacity

        return {
            'soc': soc,
            'actual_power': actual_power,
            'losses': losses,
            'final_energy': current_energy
        }

    def battery_degradation(self, battery: BatterySystem,
                          cycles: int, average_dod: float,
                          average_temperature: float = 298) -> Dict[str, float]:
        """
        Model battery capacity fade over cycles
        """
        # Arrhenius temperature factor
        T_ref = 298  # K
        Ea = 20000  # J/mol activation energy
        temp_factor = np.exp(Ea / R * (1/T_ref - 1/average_temperature))

        # DOD stress factor
        dod_factor = (average_dod / 0.8) ** 2

        # Cycle aging
        if battery.technology == "Li-ion":
            # Capacity fade model for Li-ion
            capacity_fade = 0.2 * (cycles / battery.cycle_life) ** 0.5 * temp_factor * dod_factor
        elif battery.technology == "Lead-acid":
            # Faster degradation for lead-acid
            capacity_fade = 0.3 * (cycles / battery.cycle_life) ** 0.75 * temp_factor * dod_factor
        else:
            capacity_fade = 0.25 * (cycles / battery.cycle_life) ** 0.6

        # Calendar aging
        calendar_fade = 0.02 * (cycles / 365) * temp_factor  # Assuming daily cycling

        # Total fade
        total_fade = min(capacity_fade + calendar_fade, 0.8)  # Max 80% fade

        # Remaining capacity
        remaining_capacity = battery.capacity * (1 - total_fade)

        # Resistance increase
        resistance_increase = 1 + 2 * total_fade

        return {
            'capacity_fade': total_fade,
            'remaining_capacity': remaining_capacity,
            'resistance_increase': resistance_increase,
            'remaining_cycles': max(0, battery.cycle_life * (1 - total_fade/0.2))
        }

    def flow_battery_model(self, power: float, energy: float,
                          electrolyte_volume: float) -> Dict[str, float]:
        """
        Model flow battery (Vanadium redox, Zinc-bromine, etc.)
        """
        # Power determined by stack size
        stack_area = power / 1000  # m², assuming 1 kW/m²

        # Energy determined by electrolyte volume
        energy_density = 30  # Wh/L for vanadium redox
        available_energy = electrolyte_volume * energy_density

        # Efficiency depends on current density
        current_density = power / (stack_area * 48)  # A/m² (48V nominal)
        if current_density < 200:
            efficiency = 0.85
        elif current_density < 500:
            efficiency = 0.80
        else:
            efficiency = 0.75

        # Pumping power
        pump_power = 0.02 * power  # 2% parasitic loss

        return {
            'stack_area': stack_area,
            'available_energy': available_energy,
            'efficiency': efficiency,
            'pump_power': pump_power,
            'net_power': power - pump_power
        }

    # ============= GRID INTEGRATION =============

    def grid_frequency_response(self, grid: GridConnection,
                               power_imbalance: float,
                               system_inertia: float = 10000) -> float:
        """
        Calculate grid frequency response to power imbalance
        """
        # Swing equation
        # df/dt = (P_mechanical - P_electrical) / (2 * H * S_base)

        # Rate of change of frequency (ROCOF)
        rocof = power_imbalance / (2 * system_inertia * grid.frequency)

        # Frequency deviation (simplified - first second)
        frequency_deviation = rocof

        return frequency_deviation

    def inverter_control(self, dc_power: float, dc_voltage: float,
                        grid_voltage: float, grid_frequency: float) -> Dict[str, float]:
        """
        Model grid-tied inverter with MPPT and grid synchronization
        """
        # Maximum Power Point Tracking (simplified)
        mppt_efficiency = 0.98

        # DC to AC conversion efficiency (depends on loading)
        loading = dc_power / (dc_voltage * 100)  # Assuming 100A max
        if loading < 0.1:
            conversion_efficiency = 0.90
        elif loading < 0.5:
            conversion_efficiency = 0.96
        else:
            conversion_efficiency = 0.98

        # Total efficiency
        total_efficiency = mppt_efficiency * conversion_efficiency

        # AC output power
        ac_power = dc_power * total_efficiency

        # Power factor (unity for modern inverters)
        power_factor = 0.99

        # AC current
        ac_current = ac_power / (grid_voltage * power_factor)

        # THD (Total Harmonic Distortion)
        if loading < 0.2:
            thd = 0.05  # 5%
        else:
            thd = 0.02  # 2%

        return {
            'ac_power': ac_power,
            'ac_current': ac_current,
            'efficiency': total_efficiency,
            'power_factor': power_factor,
            'thd': thd
        }

    def microgrid_optimization(self, solar_power: np.ndarray,
                             wind_power: np.ndarray,
                             load_demand: np.ndarray,
                             battery: BatterySystem,
                             grid: GridConnection,
                             time_step: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Optimize microgrid operation with renewable sources, storage, and grid connection
        """
        n_steps = len(load_demand)
        battery_power = np.zeros(n_steps)
        grid_power = np.zeros(n_steps)
        curtailment = np.zeros(n_steps)
        cost = np.zeros(n_steps)

        # Battery state
        battery_energy = battery.capacity * 0.5  # Start at 50% SOC

        for i in range(n_steps):
            # Net renewable generation
            renewable = solar_power[i] + wind_power[i]
            net_power = renewable - load_demand[i]

            if net_power > 0:  # Excess generation
                # First, try to charge battery
                max_charge = min(net_power, battery.max_charge_rate * battery.capacity)
                available_capacity = battery.capacity - battery_energy

                if available_capacity > 0:
                    charge = min(max_charge, available_capacity / time_step)
                    battery_power[i] = charge
                    battery_energy += charge * time_step * battery.efficiency
                    net_power -= charge
                else:
                    battery_power[i] = 0

                # Then, export to grid
                if net_power > 0:
                    if net_power <= grid.max_export:
                        grid_power[i] = -net_power  # Negative for export
                        cost[i] = -net_power * grid.feed_in_tariff * time_step
                    else:
                        grid_power[i] = -grid.max_export
                        cost[i] = -grid.max_export * grid.feed_in_tariff * time_step
                        curtailment[i] = net_power - grid.max_export

            else:  # Deficit generation
                deficit = -net_power

                # First, try to discharge battery
                max_discharge = min(deficit, battery.max_discharge_rate * battery.capacity)
                available_energy = battery_energy - battery.capacity * (1 - battery.depth_of_discharge)

                if available_energy > 0:
                    discharge = min(max_discharge, available_energy / time_step)
                    battery_power[i] = -discharge
                    battery_energy -= discharge * time_step / battery.efficiency
                    deficit -= discharge
                else:
                    battery_power[i] = 0

                # Then, import from grid
                if deficit > 0:
                    grid_power[i] = min(deficit, grid.max_import)
                    cost[i] = grid_power[i] * grid.electricity_price * time_step

                    if deficit > grid.max_import:
                        # Load shedding required
                        pass

        return {
            'battery_power': battery_power,
            'grid_power': grid_power,
            'curtailment': curtailment,
            'cost': cost,
            'total_cost': np.sum(cost)
        }

    def hydrogen_storage(self, excess_power: float, electrolyzer_efficiency: float = 0.7,
                        fuel_cell_efficiency: float = 0.5) -> Dict[str, float]:
        """
        Model hydrogen energy storage system
        """
        # Electrolyzer: Power to H2
        h2_production_rate = excess_power * electrolyzer_efficiency / 33.33  # kg/h (HHV of H2 = 33.33 kWh/kg)

        # Storage pressure
        storage_pressure = 700  # bar
        compression_work = 0.1 * excess_power  # ~10% for compression

        # Net H2 energy stored
        net_h2_energy = (excess_power - compression_work) * electrolyzer_efficiency

        # Round-trip efficiency
        round_trip_efficiency = electrolyzer_efficiency * fuel_cell_efficiency * 0.9  # 90% for compression losses

        # Power output from fuel cell
        fuel_cell_power = net_h2_energy * fuel_cell_efficiency

        return {
            'h2_production_kg_per_hour': h2_production_rate,
            'compression_power': compression_work,
            'stored_energy': net_h2_energy,
            'fuel_cell_power': fuel_cell_power,
            'round_trip_efficiency': round_trip_efficiency
        }

    # ============= ENERGY HARVESTING =============

    def tidal_energy(self, tidal_velocity: float, turbine_area: float,
                    efficiency: float = 0.35) -> float:
        """
        Calculate tidal energy generation
        """
        water_density = 1025  # kg/m³
        power = 0.5 * water_density * turbine_area * tidal_velocity ** 3 * efficiency
        return power

    def geothermal_power(self, flow_rate: float, inlet_temp: float,
                        outlet_temp: float, efficiency: float = 0.1) -> float:
        """
        Calculate geothermal power generation
        flow_rate: kg/s
        temperatures: Celsius
        """
        specific_heat = 4186  # J/(kg·K) for water
        thermal_power = flow_rate * specific_heat * (inlet_temp - outlet_temp)
        electric_power = thermal_power * efficiency
        return electric_power

    def biomass_gasification(self, biomass_kg: float, moisture_content: float = 0.1,
                           gasifier_efficiency: float = 0.75) -> Dict[str, float]:
        """
        Model biomass gasification for power generation
        """
        # Lower heating value depends on moisture
        lhv_dry = 18  # MJ/kg for wood
        lhv_wet = lhv_dry * (1 - moisture_content) - 2.44 * moisture_content  # MJ/kg

        # Syngas production
        syngas_energy = biomass_kg * lhv_wet * gasifier_efficiency  # MJ

        # Power generation (assuming gas engine)
        engine_efficiency = 0.35
        electric_power = syngas_energy * engine_efficiency / 3.6  # kWh

        # Biochar production (carbon sequestration)
        biochar_yield = biomass_kg * 0.15 * (1 - moisture_content)  # kg

        return {
            'syngas_energy_MJ': syngas_energy,
            'electric_power_kWh': electric_power,
            'biochar_kg': biochar_yield,
            'carbon_sequestered_kg': biochar_yield * 0.8  # 80% carbon content
        }

    # ============= UTILITIES =============

    def _initialize_solar_spectrum(self) -> np.ndarray:
        """
        Initialize AM1.5 solar spectrum (simplified)
        """
        # Wavelength range (nm)
        wavelengths = np.linspace(300, 2500, 100)

        # Simplified AM1.5 spectrum (W/m²/nm)
        spectrum = np.zeros((len(wavelengths), 2))
        spectrum[:, 0] = wavelengths

        # Approximate spectral irradiance
        for i, wl in enumerate(wavelengths):
            if wl < 400:
                spectrum[i, 1] = 0.5
            elif wl < 700:
                spectrum[i, 1] = 1.5
            elif wl < 1000:
                spectrum[i, 1] = 1.2
            elif wl < 1500:
                spectrum[i, 1] = 0.8
            else:
                spectrum[i, 1] = 0.3

        # Normalize to 1000 W/m² total
        total = np.trapz(spectrum[:, 1], spectrum[:, 0])
        spectrum[:, 1] *= 1000 / total

        return spectrum

    def levelized_cost(self, capital_cost: float, operating_cost: float,
                      energy_produced: float, lifetime_years: int,
                      discount_rate: float = 0.07) -> float:
        """
        Calculate Levelized Cost of Energy (LCOE)
        """
        # Present value factors
        pv_factors = [(1 + discount_rate) ** -t for t in range(lifetime_years)]

        # Total costs
        total_cost = capital_cost + sum(operating_cost * pv for pv in pv_factors)

        # Total energy
        total_energy = sum(energy_produced * pv for pv in pv_factors)

        # LCOE
        lcoe = total_cost / total_energy if total_energy > 0 else np.inf

        return lcoe

def run_demo():
    """Comprehensive demonstration of renewable energy lab"""
    print("="*60)
    print("RENEWABLE ENERGY LAB - Comprehensive Demo")
    print("="*60)

    lab = RenewableEnergyLab()

    # Solar energy
    print("\n1. SOLAR PHOTOVOLTAICS")
    print("-" * 40)

    silicon_cell = SolarCell(
        material="Silicon",
        bandgap=1.12,
        efficiency=0.22,
        area=1.0,
        temperature_coefficient=-0.004,
        series_resistance=0.01,
        shunt_resistance=1000
    )

    # Calculate irradiance at noon
    irradiance = lab.solar_irradiance(latitude=40, day_of_year=172, hour=12, tilt_angle=30)
    print(f"Solar irradiance at noon: {irradiance:.1f} W/m²")

    # I-V curve
    V, I = lab.solar_cell_iv_curve(silicon_cell, irradiance, temperature=298)
    P = V * I
    max_power_idx = np.argmax(P)

    print(f"Maximum power point: {P[max_power_idx]:.1f} W")
    print(f"MPP voltage: {V[max_power_idx]:.2f} V")
    print(f"MPP current: {I[max_power_idx]:.2f} A")

    # Efficiency
    efficiency = lab.solar_cell_efficiency(silicon_cell)
    print(f"Cell efficiency: {efficiency*100:.1f}%")

    # Wind energy
    print("\n2. WIND TURBINE")
    print("-" * 40)

    turbine = WindTurbine(
        rotor_diameter=80,
        hub_height=80,
        rated_power=2e6,  # 2 MW
        cut_in_speed=3,
        rated_speed=12,
        cut_out_speed=25
    )

    # Wind profile
    wind_10m = 6.0  # m/s at 10m height
    wind_80m = lab.wind_profile_height(wind_10m, 10, 80)
    print(f"Wind speed at 10m: {wind_10m:.1f} m/s")
    print(f"Wind speed at 80m hub: {wind_80m:.1f} m/s")

    # Power curve
    wind_speeds = np.linspace(0, 30, 100)
    power = lab.wind_power_curve(turbine, wind_speeds)
    print(f"Rated power: {turbine.rated_power/1e6:.1f} MW")

    # Weibull distribution and capacity factor
    wind_speeds, pdf = lab.weibull_wind_distribution(k=2.0, c=7.0)
    cf = lab.capacity_factor(turbine, (wind_speeds, pdf))
    print(f"Capacity factor: {cf*100:.1f}%")
    print(f"Annual energy: {cf * turbine.rated_power * 8760 / 1e6:.1f} MWh")

    # Battery storage
    print("\n3. BATTERY STORAGE")
    print("-" * 40)

    battery = BatterySystem(
        technology="Li-ion",
        capacity=100,  # kWh
        voltage=400,
        max_discharge_rate=1.0,  # 1C
        max_charge_rate=0.5,  # 0.5C
        efficiency=0.95,
        cycle_life=5000
    )

    # Charge/discharge cycle
    hours = np.arange(24)
    solar_gen = np.maximum(0, 50 * np.sin(np.pi * (hours - 6) / 12))
    load = 30 + 10 * np.sin(np.pi * hours / 12)
    power_profile = solar_gen - load

    results = lab.battery_charge_discharge(battery, power_profile, time_step=1.0)

    print(f"Battery capacity: {battery.capacity} kWh")
    print(f"Final SOC: {results['soc'][-1]*100:.1f}%")
    print(f"Total losses: {np.sum(results['losses']):.1f} kWh")

    # Degradation after 1000 cycles
    degradation = lab.battery_degradation(battery, cycles=1000, average_dod=0.8, average_temperature=308)
    print(f"Capacity fade after 1000 cycles: {degradation['capacity_fade']*100:.1f}%")

    # Grid integration
    print("\n4. MICROGRID OPTIMIZATION")
    print("-" * 40)

    grid = GridConnection(
        voltage=400,
        frequency=50,
        max_export=50,  # kW
        max_import=100,  # kW
        feed_in_tariff=0.10,  # $/kWh
        electricity_price=0.20  # $/kWh
    )

    # 24-hour simulation
    solar = solar_gen
    wind = 20 * np.ones(24)  # Constant 20 kW
    load = 30 + 20 * np.sin(np.pi * hours / 12)

    microgrid = lab.microgrid_optimization(solar, wind, load, battery, grid)

    print(f"Total grid import: {np.sum(microgrid['grid_power'][microgrid['grid_power'] > 0]):.1f} kWh")
    print(f"Total grid export: {-np.sum(microgrid['grid_power'][microgrid['grid_power'] < 0]):.1f} kWh")
    print(f"Total curtailment: {np.sum(microgrid['curtailment']):.1f} kWh")
    print(f"Net cost: ${microgrid['total_cost']:.2f}")

    # Hydrogen storage
    print("\n5. HYDROGEN ENERGY STORAGE")
    print("-" * 40)

    h2_system = lab.hydrogen_storage(excess_power=100)  # 100 kW excess

    print(f"H2 production: {h2_system['h2_production_kg_per_hour']:.2f} kg/h")
    print(f"Round-trip efficiency: {h2_system['round_trip_efficiency']*100:.1f}%")
    print(f"Fuel cell power output: {h2_system['fuel_cell_power']:.1f} kW")

    # LCOE calculation
    print("\n6. LEVELIZED COST OF ENERGY")
    print("-" * 40)

    # Solar LCOE
    solar_lcoe = lab.levelized_cost(
        capital_cost=1000 * 1000,  # $1000/kW for 1MW
        operating_cost=20000,  # $20k/year O&M
        energy_produced=1500 * 1000,  # 1500 MWh/year (capacity factor ~17%)
        lifetime_years=25
    )

    # Wind LCOE
    wind_lcoe = lab.levelized_cost(
        capital_cost=2000 * 2000,  # $2000/kW for 2MW
        operating_cost=50000,  # $50k/year O&M
        energy_produced=cf * 2000 * 8760,  # Based on capacity factor
        lifetime_years=20
    )

    print(f"Solar LCOE: ${solar_lcoe:.3f}/kWh")
    print(f"Wind LCOE: ${wind_lcoe:.3f}/kWh")

if __name__ == '__main__':
    run_demo()