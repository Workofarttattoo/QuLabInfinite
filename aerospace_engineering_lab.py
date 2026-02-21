"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

AEROSPACE ENGINEERING LAB
Production-ready aerospace engineering calculations for aerodynamics, propulsion, orbital mechanics, and flight dynamics.
Free gift to the scientific community from QuLabInfinite.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy import constants, optimize, integrate
from typing import Dict, Tuple, List, Optional
import warnings

@dataclass
class AerospaceEngineeringLab:
    """
    Comprehensive aerospace engineering calculations including:
    - Lift and drag calculations
    - Orbital mechanics and trajectory planning
    - Propulsion system analysis
    - Structural analysis for aircraft and spacecraft
    - Stability and control analysis
    - Trajectory optimization
    """

    # Standard atmosphere at sea level
    temperature_sl: float = 288.15  # K (15°C)
    pressure_sl: float = 101325  # Pa
    density_sl: float = 1.225  # kg/m³
    gravity: float = constants.g  # m/s²

    # Earth parameters
    earth_radius: float = 6371000  # m
    earth_mass: float = 5.972e24  # kg
    earth_mu: float = 3.986e14  # m³/s² (gravitational parameter)

    # Universal constants
    gas_constant: float = 287.05  # J/(kg·K) for air
    gamma: float = 1.4  # Specific heat ratio for air
    universal_gas_constant: float = constants.R  # J/(mol·K)

    def __post_init__(self):
        """Initialize derived constants"""
        self.gravitational_constant = constants.G
        self.stefan_boltzmann = constants.stefan_boltzmann

    def standard_atmosphere(self, altitude: float) -> Dict[str, float]:
        """
        Calculate atmospheric properties using International Standard Atmosphere model

        Args:
            altitude: Altitude in meters (0-85000m)

        Returns:
            Dictionary with temperature, pressure, density, and speed of sound
        """
        if altitude < 0:
            raise ValueError("Altitude must be non-negative")

        if altitude <= 11000:  # Troposphere
            temp = self.temperature_sl - 0.0065 * altitude
            pressure = self.pressure_sl * (temp / self.temperature_sl)**5.2561
        elif altitude <= 25000:  # Lower stratosphere
            temp = 216.65
            pressure = 22632 * np.exp(-self.gravity * (altitude - 11000) / (self.gas_constant * temp))
        elif altitude <= 47000:  # Upper stratosphere
            temp = 216.65 + 0.001 * (altitude - 25000)
            pressure = 2488.84 * (temp / 216.65)**(-34.1632)
        elif altitude <= 53000:  # Lower mesosphere
            temp = 270.65
            pressure = 120.45 * np.exp(-self.gravity * (altitude - 47000) / (self.gas_constant * temp))
        elif altitude <= 79000:  # Upper mesosphere
            temp = 270.65 - 0.0028 * (altitude - 53000)
            pressure = 58.323 * (temp / 270.65)**(12.2011)
        elif altitude <= 90000:  # Lower thermosphere
            temp = 186.946
            pressure = 1.0313 * np.exp(-self.gravity * (altitude - 79000) / (self.gas_constant * temp))
        else:  # Upper atmosphere
            temp = 186.946
            pressure = 0.0

        density = pressure / (self.gas_constant * temp) if temp > 0 else 0
        speed_of_sound = np.sqrt(self.gamma * self.gas_constant * temp) if temp > 0 else 0

        return {
            'temperature': temp,  # K
            'pressure': pressure,  # Pa
            'density': density,  # kg/m³
            'speed_of_sound': speed_of_sound  # m/s
        }

    def lift_coefficient(self, angle_of_attack: float, aspect_ratio: float,
                        sweep_angle: float = 0, efficiency: float = 0.9) -> float:
        """
        Calculate lift coefficient using lifting-line theory

        CL = 2π * α * (AR / (AR + 2))

        Args:
            angle_of_attack: Angle of attack in radians
            aspect_ratio: Wing aspect ratio (span²/area)
            sweep_angle: Wing sweep angle in radians
            efficiency: Span efficiency factor (0.8-1.0)

        Returns:
            Lift coefficient
        """
        # Account for sweep angle
        effective_ar = aspect_ratio * np.cos(sweep_angle)

        # Basic lifting-line theory
        cl_slope = 2 * np.pi * effective_ar / (effective_ar + 2)

        # Apply efficiency
        cl = cl_slope * angle_of_attack * efficiency

        return cl

    def drag_coefficient(self, lift_coefficient: float, aspect_ratio: float,
                        cd_zero: float = 0.02, efficiency: float = 0.85) -> float:
        """
        Calculate drag coefficient using drag polar

        CD = CD0 + CL²/(π*e*AR)

        Args:
            lift_coefficient: Lift coefficient
            aspect_ratio: Wing aspect ratio
            cd_zero: Zero-lift drag coefficient
            efficiency: Oswald efficiency factor

        Returns:
            Total drag coefficient
        """
        induced_drag = lift_coefficient**2 / (np.pi * efficiency * aspect_ratio)
        return cd_zero + induced_drag

    def lift_drag_forces(self, velocity: float, altitude: float, wing_area: float,
                        cl: float, cd: float) -> Dict[str, float]:
        """
        Calculate aerodynamic forces

        Args:
            velocity: Airspeed in m/s
            altitude: Altitude in meters
            wing_area: Wing reference area in m²
            cl: Lift coefficient
            cd: Drag coefficient

        Returns:
            Dictionary with lift, drag, and L/D ratio
        """
        atm = self.standard_atmosphere(altitude)
        dynamic_pressure = 0.5 * atm['density'] * velocity**2

        lift = dynamic_pressure * wing_area * cl
        drag = dynamic_pressure * wing_area * cd

        return {
            'lift': lift,  # N
            'drag': drag,  # N
            'dynamic_pressure': dynamic_pressure,  # Pa
            'lift_to_drag': lift / drag if drag > 0 else float('inf')
        }

    def mach_number(self, velocity: float, altitude: float) -> float:
        """
        Calculate Mach number

        Args:
            velocity: Airspeed in m/s
            altitude: Altitude in meters

        Returns:
            Mach number
        """
        atm = self.standard_atmosphere(altitude)
        return velocity / atm['speed_of_sound'] if atm['speed_of_sound'] > 0 else 0

    def orbital_velocity(self, radius: float, central_mass: Optional[float] = None) -> float:
        """
        Calculate circular orbital velocity

        v = √(μ/r)

        Args:
            radius: Orbital radius in meters
            central_mass: Central body mass in kg (default: Earth)

        Returns:
            Orbital velocity in m/s
        """
        if central_mass is None:
            mu = self.earth_mu
        else:
            mu = self.gravitational_constant * central_mass

        return np.sqrt(mu / radius)

    def orbital_period(self, semi_major_axis: float, central_mass: Optional[float] = None) -> float:
        """
        Calculate orbital period using Kepler's third law

        T = 2π√(a³/μ)

        Args:
            semi_major_axis: Semi-major axis in meters
            central_mass: Central body mass in kg (default: Earth)

        Returns:
            Period in seconds
        """
        if central_mass is None:
            mu = self.earth_mu
        else:
            mu = self.gravitational_constant * central_mass

        return 2 * np.pi * np.sqrt(semi_major_axis**3 / mu)

    def hohmann_transfer(self, r1: float, r2: float) -> Dict[str, float]:
        """
        Calculate Hohmann transfer orbit parameters

        Args:
            r1: Initial orbit radius in meters
            r2: Final orbit radius in meters

        Returns:
            Dictionary with delta-v requirements and transfer time
        """
        # Circular velocities
        v1 = self.orbital_velocity(r1)
        v2 = self.orbital_velocity(r2)

        # Transfer orbit semi-major axis
        a_transfer = (r1 + r2) / 2

        # Transfer velocities at periapsis and apoapsis
        v_transfer_1 = np.sqrt(self.earth_mu * (2/r1 - 1/a_transfer))
        v_transfer_2 = np.sqrt(self.earth_mu * (2/r2 - 1/a_transfer))

        # Delta-v requirements
        delta_v1 = abs(v_transfer_1 - v1)
        delta_v2 = abs(v2 - v_transfer_2)
        delta_v_total = delta_v1 + delta_v2

        # Transfer time (half period of transfer orbit)
        transfer_time = np.pi * np.sqrt(a_transfer**3 / self.earth_mu)

        return {
            'delta_v1': delta_v1,  # m/s
            'delta_v2': delta_v2,  # m/s
            'delta_v_total': delta_v_total,  # m/s
            'transfer_time': transfer_time,  # seconds
            'transfer_time_hours': transfer_time / 3600  # hours
        }

    def vis_viva_equation(self, radius: float, semi_major_axis: float) -> float:
        """
        Calculate orbital velocity at any point using vis-viva equation

        v² = μ(2/r - 1/a)

        Args:
            radius: Current radius in meters
            semi_major_axis: Semi-major axis in meters

        Returns:
            Velocity in m/s
        """
        return np.sqrt(self.earth_mu * (2/radius - 1/semi_major_axis))

    def escape_velocity(self, radius: float, central_mass: Optional[float] = None) -> float:
        """
        Calculate escape velocity

        v_escape = √(2μ/r)

        Args:
            radius: Distance from center in meters
            central_mass: Central body mass in kg (default: Earth)

        Returns:
            Escape velocity in m/s
        """
        if central_mass is None:
            mu = self.earth_mu
        else:
            mu = self.gravitational_constant * central_mass

        return np.sqrt(2 * mu / radius)

    def specific_impulse(self, exhaust_velocity: float) -> float:
        """
        Calculate specific impulse from exhaust velocity

        Isp = v_e / g0

        Args:
            exhaust_velocity: Exhaust velocity in m/s

        Returns:
            Specific impulse in seconds
        """
        return exhaust_velocity / self.gravity

    def tsiolkovsky_equation(self, delta_v: float, specific_impulse: float,
                           initial_mass: float) -> float:
        """
        Calculate final mass using rocket equation

        Δv = Isp * g0 * ln(m0/mf)

        Args:
            delta_v: Required velocity change in m/s
            specific_impulse: Specific impulse in seconds
            initial_mass: Initial mass in kg

        Returns:
            Final mass in kg
        """
        exhaust_velocity = specific_impulse * self.gravity
        mass_ratio = np.exp(delta_v / exhaust_velocity)
        return initial_mass / mass_ratio

    def propellant_mass_fraction(self, delta_v: float, specific_impulse: float) -> float:
        """
        Calculate propellant mass fraction for given delta-v

        Args:
            delta_v: Required velocity change in m/s
            specific_impulse: Specific impulse in seconds

        Returns:
            Propellant mass fraction (0-1)
        """
        exhaust_velocity = specific_impulse * self.gravity
        mass_ratio = np.exp(delta_v / exhaust_velocity)
        return 1 - 1/mass_ratio

    def thrust_to_weight(self, thrust: float, mass: float, altitude: float = 0) -> float:
        """
        Calculate thrust-to-weight ratio

        Args:
            thrust: Thrust in Newtons
            mass: Vehicle mass in kg
            altitude: Altitude in meters

        Returns:
            Thrust-to-weight ratio
        """
        # Account for gravity variation with altitude
        r = self.earth_radius + altitude
        g_local = self.gravity * (self.earth_radius / r)**2

        weight = mass * g_local
        return thrust / weight

    def jet_engine_thrust(self, mass_flow: float, exit_velocity: float,
                         exit_pressure: float, exit_area: float,
                         ambient_pressure: float) -> float:
        """
        Calculate jet engine thrust

        F = ṁ*Ve + (Pe - Pa)*Ae

        Args:
            mass_flow: Mass flow rate in kg/s
            exit_velocity: Exit velocity in m/s
            exit_pressure: Exit pressure in Pa
            exit_area: Nozzle exit area in m²
            ambient_pressure: Ambient pressure in Pa

        Returns:
            Thrust in Newtons
        """
        momentum_thrust = mass_flow * exit_velocity
        pressure_thrust = (exit_pressure - ambient_pressure) * exit_area
        return momentum_thrust + pressure_thrust

    def range_equation(self, velocity: float, lift_to_drag: float,
                      specific_fuel_consumption: float,
                      initial_weight: float, final_weight: float) -> float:
        """
        Calculate aircraft range using Breguet equation

        R = (V * L/D / SFC) * ln(W0/W1)

        Args:
            velocity: Cruise velocity in m/s
            lift_to_drag: L/D ratio
            specific_fuel_consumption: SFC in 1/s
            initial_weight: Initial weight in N
            final_weight: Final weight in N

        Returns:
            Range in meters
        """
        return (velocity * lift_to_drag / specific_fuel_consumption) * \
               np.log(initial_weight / final_weight)

    def endurance_equation(self, lift_to_drag: float,
                         specific_fuel_consumption: float,
                         initial_weight: float, final_weight: float) -> float:
        """
        Calculate aircraft endurance

        E = (L/D / SFC) * ln(W0/W1)

        Args:
            lift_to_drag: L/D ratio
            specific_fuel_consumption: SFC in 1/s
            initial_weight: Initial weight in N
            final_weight: Final weight in N

        Returns:
            Endurance in seconds
        """
        return (lift_to_drag / specific_fuel_consumption) * \
               np.log(initial_weight / final_weight)

    def static_stability_margin(self, cg_location: float, neutral_point: float,
                              mean_chord: float) -> float:
        """
        Calculate static stability margin

        SM = (x_np - x_cg) / c

        Args:
            cg_location: Center of gravity location in meters
            neutral_point: Neutral point location in meters
            mean_chord: Mean aerodynamic chord in meters

        Returns:
            Static margin (positive = stable)
        """
        return (neutral_point - cg_location) / mean_chord

    def dutch_roll_frequency(self, velocity: float, wingspan: float,
                            yaw_damping: float, directional_stability: float) -> float:
        """
        Calculate Dutch roll natural frequency

        Args:
            velocity: Airspeed in m/s
            wingspan: Wingspan in meters
            yaw_damping: Yaw damping derivative
            directional_stability: Directional stability derivative

        Returns:
            Dutch roll frequency in rad/s
        """
        return np.sqrt(directional_stability * velocity / wingspan)

    def phugoid_period(self, velocity: float) -> float:
        """
        Calculate phugoid oscillation period

        T = 0.138 * V (approximate formula in SI units)

        Args:
            velocity: Airspeed in m/s

        Returns:
            Period in seconds
        """
        return 0.138 * velocity

    def beam_stress(self, moment: float, distance: float, inertia: float) -> float:
        """
        Calculate bending stress in structural member

        σ = M*y/I

        Args:
            moment: Bending moment in N·m
            distance: Distance from neutral axis in meters
            inertia: Second moment of inertia in m⁴

        Returns:
            Stress in Pa
        """
        return moment * distance / inertia

    def wing_loading(self, weight: float, wing_area: float) -> float:
        """
        Calculate wing loading

        W/S = Weight / Wing_Area

        Args:
            weight: Aircraft weight in N
            wing_area: Wing area in m²

        Returns:
            Wing loading in N/m²
        """
        return weight / wing_area

    def stall_speed(self, weight: float, wing_area: float,
                   cl_max: float, altitude: float = 0) -> float:
        """
        Calculate stall speed

        V_stall = √(2W/(ρ*S*CL_max))

        Args:
            weight: Aircraft weight in N
            wing_area: Wing area in m²
            cl_max: Maximum lift coefficient
            altitude: Altitude in meters

        Returns:
            Stall speed in m/s
        """
        atm = self.standard_atmosphere(altitude)
        return np.sqrt(2 * weight / (atm['density'] * wing_area * cl_max))

    def turn_radius(self, velocity: float, bank_angle: float) -> float:
        """
        Calculate turning radius for coordinated turn

        r = V²/(g*tan(φ))

        Args:
            velocity: Airspeed in m/s
            bank_angle: Bank angle in radians

        Returns:
            Turn radius in meters
        """
        return velocity**2 / (self.gravity * np.tan(bank_angle))

    def load_factor(self, bank_angle: float) -> float:
        """
        Calculate load factor in turn

        n = 1/cos(φ)

        Args:
            bank_angle: Bank angle in radians

        Returns:
            Load factor (g's)
        """
        return 1 / np.cos(bank_angle)

    def reynolds_number(self, velocity: float, characteristic_length: float,
                       altitude: float = 0) -> float:
        """
        Calculate Reynolds number for aerodynamic analysis

        Re = ρVL/μ

        Args:
            velocity: Flow velocity in m/s
            characteristic_length: Characteristic length in meters
            altitude: Altitude in meters

        Returns:
            Reynolds number
        """
        atm = self.standard_atmosphere(altitude)

        # Dynamic viscosity using Sutherland's law
        t_ref = 291.15
        mu_ref = 1.827e-5
        s = 120

        mu = mu_ref * (atm['temperature'] / t_ref)**1.5 * \
             (t_ref + s) / (atm['temperature'] + s)

        return atm['density'] * velocity * characteristic_length / mu

    def heat_flux_reentry(self, velocity: float, altitude: float,
                        nose_radius: float) -> float:
        """
        Calculate stagnation point heat flux during atmospheric entry

        Simplified correlation: q = C * √(ρ/r) * V³

        Args:
            velocity: Entry velocity in m/s
            altitude: Altitude in meters
            nose_radius: Nose radius in meters

        Returns:
            Heat flux in W/m²
        """
        atm = self.standard_atmosphere(altitude)
        c = 1.83e-4  # Empirical constant for Earth atmosphere

        heat_flux = c * np.sqrt(atm['density'] / nose_radius) * velocity**3

        return heat_flux

    def trajectory_simulation(self, initial_conditions: Dict,
                            vehicle_params: Dict,
                            time_span: Tuple[float, float],
                            time_steps: int = 1000) -> Dict[str, np.ndarray]:
        """
        Simulate 2D trajectory with drag

        Args:
            initial_conditions: Dict with 'velocity', 'altitude', 'flight_path_angle'
            vehicle_params: Dict with 'mass', 'area', 'cd'
            time_span: Tuple of (start_time, end_time) in seconds
            time_steps: Number of time steps

        Returns:
            Dictionary with time, altitude, velocity, and range arrays
        """
        t = np.linspace(time_span[0], time_span[1], time_steps)
        dt = (time_span[1] - time_span[0]) / (time_steps - 1)

        # Initialize arrays
        altitude = np.zeros(time_steps)
        velocity = np.zeros(time_steps)
        range_distance = np.zeros(time_steps)
        flight_path_angle = np.zeros(time_steps)

        # Set initial conditions
        altitude[0] = initial_conditions['altitude']
        velocity[0] = initial_conditions['velocity']
        flight_path_angle[0] = initial_conditions['flight_path_angle']

        # Extract vehicle parameters
        mass = vehicle_params['mass']
        area = vehicle_params['area']
        cd = vehicle_params['cd']

        # Integrate trajectory
        for i in range(1, time_steps):
            # Current atmospheric properties
            atm = self.standard_atmosphere(altitude[i-1])

            # Drag acceleration
            drag_accel = 0.5 * atm['density'] * velocity[i-1]**2 * area * cd / mass

            # Gravity (accounting for altitude)
            r = self.earth_radius + altitude[i-1]
            g_local = self.gravity * (self.earth_radius / r)**2

            # Accelerations
            a_tangential = -drag_accel - g_local * np.sin(flight_path_angle[i-1])
            a_normal = -g_local * np.cos(flight_path_angle[i-1]) / velocity[i-1] \
                      if velocity[i-1] > 0 else 0

            # Update states
            velocity[i] = velocity[i-1] + a_tangential * dt
            flight_path_angle[i] = flight_path_angle[i-1] + a_normal * dt

            # Update position
            altitude[i] = altitude[i-1] + velocity[i-1] * np.sin(flight_path_angle[i-1]) * dt
            range_distance[i] = range_distance[i-1] + \
                               velocity[i-1] * np.cos(flight_path_angle[i-1]) * dt

            # Stop if hit ground
            if altitude[i] < 0:
                altitude[i:] = 0
                velocity[i:] = 0
                break

        return {
            'time': t,
            'altitude': altitude,
            'velocity': velocity,
            'range': range_distance,
            'flight_path_angle': np.degrees(flight_path_angle)
        }


def run_demo():
    """Demonstrate aerospace engineering calculations"""
    print("=" * 70)
    print("AEROSPACE ENGINEERING LAB - Production Demo")
    print("=" * 70)

    lab = AerospaceEngineeringLab()

    # Atmospheric Properties
    print("\n1. ATMOSPHERIC PROPERTIES")
    print("-" * 40)
    for alt in [0, 5000, 10000, 20000]:
        atm = lab.standard_atmosphere(alt)
        print(f"Altitude {alt}m:")
        print(f"  Temperature: {atm['temperature']:.1f} K")
        print(f"  Pressure: {atm['pressure']/1000:.1f} kPa")
        print(f"  Density: {atm['density']:.3f} kg/m³")

    # Aerodynamics
    print("\n2. AERODYNAMIC FORCES")
    print("-" * 40)
    cl = lab.lift_coefficient(np.radians(5), 8)
    cd = lab.drag_coefficient(cl, 8)
    forces = lab.lift_drag_forces(250, 10000, 120, cl, cd)
    print(f"At 250 m/s, 10km altitude:")
    print(f"  Lift coefficient: {cl:.3f}")
    print(f"  Drag coefficient: {cd:.4f}")
    print(f"  Lift: {forces['lift']/1000:.1f} kN")
    print(f"  L/D ratio: {forces['lift_to_drag']:.1f}")

    # Orbital Mechanics
    print("\n3. ORBITAL MECHANICS")
    print("-" * 40)
    leo_altitude = 400000  # 400 km
    geo_altitude = 35786000  # GEO
    v_leo = lab.orbital_velocity(lab.earth_radius + leo_altitude)
    period_geo = lab.orbital_period(lab.earth_radius + geo_altitude)
    print(f"LEO (400km): {v_leo:.0f} m/s")
    print(f"GEO period: {period_geo/3600:.1f} hours")

    hohmann = lab.hohmann_transfer(lab.earth_radius + leo_altitude,
                                   lab.earth_radius + geo_altitude)
    print(f"LEO to GEO Hohmann transfer:")
    print(f"  Total Δv: {hohmann['delta_v_total']:.0f} m/s")
    print(f"  Transfer time: {hohmann['transfer_time_hours']:.1f} hours")

    # Propulsion
    print("\n4. PROPULSION ANALYSIS")
    print("-" * 40)
    isp = 300  # seconds
    delta_v = 3000  # m/s
    initial_mass = 10000  # kg
    final_mass = lab.tsiolkovsky_equation(delta_v, isp, initial_mass)
    prop_fraction = lab.propellant_mass_fraction(delta_v, isp)
    print(f"Rocket equation (Δv=3000 m/s, Isp=300s):")
    print(f"  Final mass: {final_mass:.0f} kg")
    print(f"  Propellant fraction: {prop_fraction:.3f}")

    thrust = lab.jet_engine_thrust(100, 400, 50000, 1.0, 20000)
    print(f"Jet engine thrust: {thrust/1000:.1f} kN")

    # Stability and Control
    print("\n5. STABILITY & CONTROL")
    print("-" * 40)
    stall = lab.stall_speed(100000, 120, 1.8)
    print(f"Stall speed: {stall:.1f} m/s")

    turn_r = lab.turn_radius(100, np.radians(30))
    load = lab.load_factor(np.radians(30))
    print(f"Turn at 100 m/s, 30° bank:")
    print(f"  Radius: {turn_r:.0f} m")
    print(f"  Load factor: {load:.2f} g")

    margin = lab.static_stability_margin(2.5, 3.0, 2.0)
    print(f"Static margin: {margin:.1%}")

    # Trajectory Simulation
    print("\n6. TRAJECTORY SIMULATION")
    print("-" * 40)
    initial = {'velocity': 300, 'altitude': 10000, 'flight_path_angle': np.radians(-5)}
    vehicle = {'mass': 1000, 'area': 2, 'cd': 0.3}
    traj = lab.trajectory_simulation(initial, vehicle, (0, 60), 100)
    print(f"Ballistic trajectory (60s simulation):")
    print(f"  Final altitude: {traj['altitude'][-1]:.0f} m")
    print(f"  Final velocity: {traj['velocity'][-1]:.0f} m/s")
    print(f"  Range covered: {traj['range'][-1]/1000:.1f} km")

    print("\n" + "=" * 70)
    print("Demo completed successfully!")


if __name__ == '__main__':
    run_demo()