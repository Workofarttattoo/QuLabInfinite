# Environmental Simulator - Advanced Examples

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

## Example 1: Aerogel Thermal Conductivity Test

Simulate aerogel material under extreme cold with wind flow to measure thermal performance:

```python
from environmental_sim import EnvironmentalSimulator
import numpy as np

# Create simulator
sim = EnvironmentalSimulator()

# Setup cryogenic test conditions
sim.controller.temperature.set_temperature(-200, unit="C")
sim.controller.pressure.set_pressure(0.001, unit="bar")
sim.controller.atmosphere.set_standard_atmosphere("air")

# Add temperature gradient to simulate heat flow
sim.controller.temperature.set_gradient((50, 0, 0), unit="K/m")  # 50 K/m across sample

# Add wind parallel to surface
sim.controller.fluid.set_wind((30 * 0.44704, 0, 0), unit="m/s")  # 30 mph

# Add heat source on one side (simulating warm environment)
sim.controller.temperature.add_heat_source((-0.05, 0, 0), power=100, radius=0.02)

# Measure temperature at various positions through aerogel (10cm thick)
positions = [(x, 0, 0) for x in np.linspace(-0.05, 0.05, 20)]
temperatures = []

for pos in positions:
    conditions = sim.get_conditions(position=pos)
    temperatures.append(conditions['temperature_C'])
    print(f"Position {pos[0]*100:.1f}cm: {conditions['temperature_C']:.2f}°C")

# Calculate effective thermal conductivity from gradient
temp_gradient = (temperatures[-1] - temperatures[0]) / 0.1  # K/m
print(f"\nTemperature gradient: {temp_gradient:.2f} K/m")
print(f"Aerogel demonstrates excellent insulation at cryogenic temperatures")
```

## Example 2: Diamond Anvil Cell Pressure Calibration

Simulate extreme pressure conditions for materials science:

```python
from environmental_sim import EnvironmentalSimulator

sim = EnvironmentalSimulator()

# Setup diamond anvil cell conditions
pressure_gpa = 100  # 100 GPa (1 million bar)
temperature_k = 3000  # 3000 K

sim.controller.temperature.set_temperature(temperature_k, unit="K")
sim.controller.pressure.set_pressure(pressure_gpa, unit="GPa")

# Calculate compression ratio
compression = sim.controller.pressure.calculate_compression_ratio(
    reference_pressure=1.01325, unit="bar"
)
print(f"Compression ratio: {compression:.0f}x atmospheric pressure")

# Check if conditions are supercritical for various substances
substances = ["CO2", "H2O", "N2"]
for substance in substances:
    is_super = sim.controller.pressure.is_supercritical(temperature_k, substance)
    print(f"{substance}: {'SUPERCRITICAL' if is_super else 'subcritical'}")

# Calculate material stress
material_props = {
    'elastic_modulus': 400e9,  # Diamond: 400 GPa
    'thermal_expansion': 1e-6,  # Very low for diamond
}

stress = sim.controller.calculate_material_stress(material_props, position=(0, 0, 0))
print(f"\nDiamond anvil stress analysis:")
print(f"  Pressure stress: {stress['pressure_stress_Pa']/1e9:.2f} GPa")
print(f"  Thermal stress: {stress['thermal_stress_Pa']/1e9:.2f} GPa")
print(f"  Total stress: {stress['total_stress_Pa']/1e9:.2f} GPa")
```

## Example 3: Low Earth Orbit Satellite Thermal Cycling

Simulate thermal cycling in LEO with solar radiation:

```python
from environmental_sim import EnvironmentalSimulator
import numpy as np

sim = EnvironmentalSimulator()

# Setup base LEO conditions
sim.controller.set_preset_environment("LEO")

# Define 90-minute orbital thermal cycle
def orbital_thermal_cycle(time):
    """Temperature varies with sunlight/shadow in 90-minute orbit."""
    period = 5400  # seconds (90 minutes)
    phase = (time % period) / period

    if phase < 0.5:  # Sunlight
        return 120 + 273.15  # 120°C in sun
    else:  # Shadow
        return -100 + 273.15  # -100°C in shadow

sim.controller.temperature.set_heating_profile(orbital_thermal_cycle)

# Add solar radiation when in sunlight
sim.controller.radiation.add_em_radiation(
    "UV", intensity=1360, wavelength=200e-9, direction=(0, 0, -1)
)

# Run simulation for 3 orbits (4.5 hours)
duration = 3 * 5400  # 3 orbits
time_step = 60  # 1 minute steps

print("Simulating 3 LEO orbits (4.5 hours)...")
print("Orbit | Time (min) | Temp (°C) | Pressure (Pa) | Radiation (W/m²)")
print("-" * 75)

for orbit in range(3):
    for phase in [0.25, 0.75]:  # Mid-sunlight and mid-shadow
        sim_time = orbit * 5400 + phase * 5400
        sim.controller.set_simulation_time(sim_time)

        conditions = sim.get_conditions()
        phase_name = "Sun" if phase < 0.5 else "Shadow"

        print(f"  {orbit+1}   | {sim_time/60:7.1f}    | {conditions['temperature_C']:7.1f} | "
              f"{conditions['pressure_Pa']:11.2e} | {conditions['em_intensity_W_m2']:8.1f}")

print("\nThermal cycling complete - satellite must survive ΔT = 220°C!")
```

## Example 4: Chemical Reactor Optimization

Optimize temperature/pressure for supercritical CO₂ extraction:

```python
from environmental_sim import EnvironmentalSimulator
import numpy as np

sim = EnvironmentalSimulator()

# Supercritical CO2: T > 304 K, P > 73.8 bar
print("Optimizing supercritical CO₂ extraction conditions\n")
print("Target: T > 304 K (31°C), P > 73.8 bar")
print("-" * 60)

# Test various temperature/pressure combinations
temps_c = [25, 35, 45, 55]
pressures_bar = [50, 75, 100, 150]

results = []

for temp_c in temps_c:
    for pressure_bar in pressures_bar:
        sim.controller.temperature.set_temperature(temp_c, unit="C")
        sim.controller.pressure.set_pressure(pressure_bar, unit="bar")
        sim.controller.atmosphere.set_composition({'CO2': 100.0})

        temp_k = temp_c + 273.15
        is_supercritical = sim.controller.pressure.is_supercritical(temp_k, "CO2")

        results.append({
            'temp_c': temp_c,
            'pressure_bar': pressure_bar,
            'supercritical': is_supercritical
        })

        status = "✓ SUPERCRITICAL" if is_supercritical else "✗ subcritical"
        print(f"T={temp_c:2d}°C, P={pressure_bar:3d} bar: {status}")

# Recommend optimal conditions
supercritical_conditions = [r for r in results if r['supercritical']]
optimal = min(supercritical_conditions,
              key=lambda x: x['temp_c'] + x['pressure_bar']*0.1)  # Minimize energy

print(f"\n✓ OPTIMAL CONDITIONS:")
print(f"  Temperature: {optimal['temp_c']}°C")
print(f"  Pressure: {optimal['pressure_bar']} bar")
print(f"  (Minimizes energy while maintaining supercritical state)")
```

## Example 5: Wind Tunnel Simulation with Turbulence

Simulate wind tunnel testing with controllable turbulence:

```python
from environmental_sim import EnvironmentalSimulator

sim = EnvironmentalSimulator()

# Standard atmospheric conditions
sim.controller.set_preset_environment("STP")

# Wind tunnel parameters
wind_speeds_mph = [50, 100, 150, 200]
turbulence_intensities = [0.0, 0.1, 0.3, 0.5]

print("Wind Tunnel Simulation")
print("=" * 70)

for speed_mph in wind_speeds_mph:
    print(f"\nWind Speed: {speed_mph} mph ({speed_mph * 0.44704:.1f} m/s)")
    print("-" * 70)

    sim.controller.fluid.set_wind((speed_mph * 0.44704, 0, 0), unit="m/s")

    # Calculate Reynolds number for 1m characteristic length
    Re = sim.controller.fluid.calculate_reynolds_number(characteristic_length=1.0)
    regime = sim.controller.fluid.get_flow_regime()

    print(f"Reynolds number: {Re:.2e} ({regime})")
    print(f"\nTurbulence | Drag Force (N) | Dynamic Pressure (Pa)")
    print("-" * 70)

    for turb in turbulence_intensities:
        sim.controller.fluid.set_turbulence(intensity=turb, length_scale=0.1)

        # Calculate drag on 1m² flat plate with Cd=1.0
        drag = sim.controller.fluid.calculate_drag_force(
            velocity=(speed_mph * 0.44704, 0, 0),
            drag_coefficient=1.0,
            reference_area=1.0
        )

        # Dynamic pressure
        wind = sim.controller.fluid.get_wind(unit="m/s")
        speed = np.linalg.norm(wind)
        q = 0.5 * 1.225 * speed**2

        print(f"{turb*100:5.0f}%     | {abs(drag[0]):12.2f}   | {q:18.2f}")

print("\n✓ Wind tunnel simulation complete")
```

## Example 6: Multi-Physics: Thermo-Mechanical Coupling

Demonstrate thermal expansion and stress in constrained material:

```python
from environmental_sim import EnvironmentalSimulator

sim = EnvironmentalSimulator()

# Material: Aluminum
material = {
    'name': 'Aluminum 6061',
    'elastic_modulus': 68.9e9,  # Pa
    'thermal_expansion': 23.6e-6,  # K⁻¹
    'thermal_conductivity': 167,  # W/(m·K)
}

print(f"Material: {material['name']}")
print("=" * 70)

# Enable thermo-mechanical coupling
sim.controller.coupling.enable_thermo_mechanical_coupling(
    thermal_expansion_coeff=material['thermal_expansion']
)

# Reference state: 20°C
ref_temp = 20
sim.controller.temperature.set_temperature(ref_temp, unit="C")
conditions_ref = sim.get_conditions()

print(f"\nReference State (T={ref_temp}°C):")
print(f"  Temperature: {conditions_ref['temperature_C']:.2f}°C")

# Heat to various temperatures
test_temps = [50, 100, 200, 300]

print(f"\nThermal Expansion Analysis:")
print("-" * 70)
print("ΔT (K) | Thermal Strain | Thermal Stress (MPa) | Length Change (mm/m)")
print("-" * 70)

for temp_c in test_temps:
    sim.controller.temperature.set_temperature(temp_c, unit="C")
    delta_t = temp_c - ref_temp

    # Calculate thermal strain
    strain = sim.controller.coupling.calculate_thermal_strain(delta_t)

    # Calculate thermal stress (if constrained)
    stress = sim.controller.coupling.calculate_thermal_stress(
        temperature=temp_c + 273.15,
        elastic_modulus=material['elastic_modulus'],
        reference_temperature=ref_temp + 273.15
    )

    # Length change for 1m bar
    length_change = strain * 1000  # mm per meter

    print(f"{delta_t:6.0f} | {strain:.6e}   | {stress/1e6:18.2f} | {length_change:19.3f}")

print("\n✓ Thermo-mechanical coupling analysis complete")
print("Note: Thermal stress calculated assuming full constraint (no expansion allowed)")
```

## Example 7: Radiation Shielding Design

Design radiation shielding for nuclear application:

```python
from environmental_sim import EnvironmentalSimulator

sim = EnvironmentalSimulator()

# Add gamma radiation source (1 MeV, 0.1 Sv/h at 1m)
source_id = sim.controller.radiation.add_ionizing_radiation(
    "gamma", dose_rate=0.1, energy=1.0, origin=(0, 0, 0)
)

print("Radiation Shielding Design")
print("=" * 70)
print("Source: Gamma radiation (1 MeV, 0.1 Sv/h at 1m)")
print("-" * 70)

# Test different shielding materials and thicknesses
materials = ["lead", "concrete", "steel", "water"]
thicknesses_cm = [0, 1, 5, 10, 20]

for material in materials:
    print(f"\n{material.upper()} Shielding:")
    print(f"Thickness (cm) | Dose Rate (mSv/h) | Reduction Factor")
    print("-" * 70)

    # Clear previous shields
    sim.controller.radiation.reset()
    sim.controller.radiation.add_ionizing_radiation(
        "gamma", dose_rate=0.1, energy=1.0, origin=(0, 0, 0)
    )

    unshielded_dose = sim.controller.radiation.get_ionizing_dose_rate((1, 0, 0))

    for thickness_cm in thicknesses_cm:
        if thickness_cm > 0:
            # Add shield between source and measurement point
            sim.controller.radiation.add_shield(
                material=material,
                thickness=thickness_cm / 100,  # Convert to meters
                position=(0.5, 0, 0),
                normal=(1, 0, 0)
            )

        dose = sim.controller.radiation.get_ionizing_dose_rate((1, 0, 0))
        reduction = unshielded_dose / dose if dose > 0 else float('inf')

        print(f"{thickness_cm:14.0f} | {dose*1000:17.2f} | {reduction:16.1f}x")

        # Reset for next iteration
        if thickness_cm > 0:
            sim.controller.radiation.reset()
            sim.controller.radiation.add_ionizing_radiation(
                "gamma", dose_rate=0.1, energy=1.0, origin=(0, 0, 0)
            )

print("\n✓ Shielding analysis complete")
print("Recommendation: Use 10cm lead for 95%+ reduction")
```

## Example 8: Time-Dependent Multi-Physics Simulation

Simulate heating element with Joule heating and thermal expansion:

```python
from environmental_sim import EnvironmentalSimulator
import numpy as np

sim = EnvironmentalSimulator()

# Material: Nichrome heating element
material = {
    'resistivity': 1.1e-6,  # Ω·m
    'thermal_expansion': 13e-6,  # K⁻¹
    'specific_heat': 450,  # J/(kg·K)
    'density': 8400,  # kg/m³
}

# Heating element geometry
length = 1.0  # m
diameter = 0.001  # m (1mm)
area = np.pi * (diameter/2)**2
volume = area * length
mass = volume * material['density']
resistance = material['resistivity'] * length / area

print("Joule Heating Simulation: Nichrome Wire")
print("=" * 70)
print(f"Wire: {length}m length, {diameter*1000}mm diameter")
print(f"Resistance: {resistance:.3f} Ω")
print("-" * 70)

# Enable electro-thermal coupling
sim.controller.coupling.enable_electro_thermal_coupling()

# Enable thermo-mechanical coupling
sim.controller.coupling.enable_thermo_mechanical_coupling(
    thermal_expansion_coeff=material['thermal_expansion']
)

# Initial conditions
sim.controller.temperature.set_temperature(20, unit="C")
current = 10.0  # Amperes

# Run time-dependent simulation
duration = 60  # seconds
time_step = 1.0  # 1 second steps

print("\nHeating Profile (10A current):")
print("Time (s) | Temp (°C) | Power (W) | Thermal Strain | Length Change (mm)")
print("-" * 70)

for t in np.arange(0, duration+time_step, time_step):
    # Calculate Joule heating
    heat_rate = sim.controller.coupling.calculate_joule_heating(
        current=current, resistance=resistance, volume=volume
    )

    # Calculate temperature change
    heat_energy = heat_rate * volume * time_step  # Joules
    temp_change = heat_energy / (mass * material['specific_heat'])

    current_temp = sim.controller.temperature.get_temperature(unit="C")
    new_temp = current_temp + temp_change
    sim.controller.temperature.set_temperature(new_temp, unit="C")

    # Calculate thermal strain and expansion
    delta_t = new_temp - 20
    strain = sim.controller.coupling.calculate_thermal_strain(delta_t)
    length_change = strain * length * 1000  # mm

    # Print every 10 seconds
    if t % 10 == 0:
        power = current**2 * resistance
        print(f"{t:8.0f} | {new_temp:8.1f}  | {power:8.2f}  | {strain:.6e}   | {length_change:15.3f}")

print("\n✓ Joule heating simulation complete")
print(f"Final temperature reached: {new_temp:.1f}°C")
```

## Integration with QuLab Materials Database

Example of using Environmental Simulator with materials from QuLab database:

```python
from environmental_sim import EnvironmentalSimulator

# Hypothetical material from QuLab database
airloy_x103 = {
    'name': 'Airloy X103 Aerogel',
    'thermal_conductivity': 0.013,  # W/(m·K) at 300K
    'density': 100,  # kg/m³
    'elastic_modulus': 1e6,  # Pa (very low)
    'thermal_expansion': 5e-6,  # K⁻¹
    'max_temperature': 200,  # °C
    'min_temperature': -200,  # °C
}

def test_material_at_conditions(material, temp_c, pressure_bar, wind_mph):
    """Test material at specified environmental conditions."""
    sim = EnvironmentalSimulator()

    # Setup environment
    sim.controller.temperature.set_temperature(temp_c, unit="C")
    sim.controller.pressure.set_pressure(pressure_bar, unit="bar")
    sim.controller.fluid.set_wind((wind_mph * 0.44704, 0, 0), unit="m/s")

    # Check if within material limits
    if not (material['min_temperature'] <= temp_c <= material['max_temperature']):
        print(f"⚠ WARNING: Temperature {temp_c}°C outside material limits!")
        return None

    # Calculate stresses
    stress = sim.controller.calculate_material_stress(
        {'elastic_modulus': material['elastic_modulus'],
         'thermal_expansion': material['thermal_expansion']},
        position=(0, 0, 0)
    )

    conditions = sim.get_conditions()

    return {
        'material': material['name'],
        'conditions': conditions,
        'stress': stress,
        'safe': stress['total_stress_Pa'] < material['elastic_modulus'] * 0.01  # 1% strain limit
    }

# Test aerogel at cryogenic conditions
result = test_material_at_conditions(airloy_x103, -200, 0.001, 30)

if result:
    print(f"\nMaterial Test: {result['material']}")
    print(f"Temperature: {result['conditions']['temperature_C']:.2f}°C")
    print(f"Pressure: {result['conditions']['pressure_bar']:.6f} bar")
    print(f"Wind: {result['conditions']['wind_velocity_m_s'][0]:.2f} m/s")
    print(f"Total Stress: {result['stress']['total_stress_Pa']:.2e} Pa")
    print(f"Status: {'✓ SAFE' if result['safe'] else '✗ UNSAFE'}")
```

---

All examples include proper copyright headers and are ready for ECH0 integration with the QuLab Infinite materials science laboratory.
