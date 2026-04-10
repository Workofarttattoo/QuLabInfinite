# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Renewable Energy Lab Demo
Comprehensive demonstrations of all renewable energy simulations
"""

import numpy as np
import json
from renewable_energy_lab import (
    SolarCellSimulator,
    BatteryChemistry,
    FuelCellAnalysis,
    EnergyStorageOptimization
)


def run_all_demos():
    """Run all renewable energy demonstrations"""
    results = {
        'lab_name': 'Renewable Energy Laboratory',
        'demonstrations': {}
    }

    # 1. Solar Cell Simulation
    print("=" * 60)
    print("SOLAR CELL SIMULATION")
    print("=" * 60)

    solar = SolarCellSimulator()

    # Shockley diode equation
    voltage_V = np.linspace(0, 0.7, 100)
    diode_result = solar.shockley_diode_equation(
        voltage_V=voltage_V,
        photocurrent_A=0.035,
        saturation_current_A=1e-12,
        ideality_factor=1.2,
        series_resistance_ohm=0.5,
        shunt_resistance_ohm=1000,
        temperature_K=300
    )
    print(f"Solar Cell Performance:")
    print(f"Open-circuit voltage: {diode_result['V_oc']:.3f} V")
    print(f"Short-circuit current: {diode_result['I_sc']:.3f} A")
    print(f"Fill factor: {diode_result['fill_factor']:.3f}")
    print(f"Efficiency: {diode_result['efficiency_percent']:.2f}%")
    print(f"Max power: {diode_result['P_max']:.4f} W")

    results['demonstrations']['solar_cell_iv'] = diode_result

    # Shockley-Queisser limit
    bandgap_eV = np.linspace(0.5, 3.0, 50)
    sq_result = solar.shockley_queisser_limit(
        bandgap_eV=bandgap_eV,
        temperature_K=300,
        concentration=1.0
    )
    print(f"\nShockley-Queisser Limit:")
    print(f"Optimal bandgap: {sq_result['optimal_bandgap_eV']:.2f} eV")
    print(f"Maximum efficiency: {sq_result['max_efficiency_percent']:.2f}%")

    results['demonstrations']['shockley_queisser'] = sq_result

    # 2. Battery Chemistry
    print("\n" + "=" * 60)
    print("BATTERY CHEMISTRY")
    print("=" * 60)

    battery = BatteryChemistry()

    # Nernst voltage
    nernst_result = battery.nernst_voltage(
        standard_potential_V=3.7,  # Li-ion
        activity_oxidized=0.5,
        activity_reduced=1.0,
        n_electrons=1,
        temperature_K=298.15
    )
    print(f"Nernst Equation:")
    print(f"Cell voltage: {nernst_result['cell_voltage_V']:.3f} V")
    print(f"Standard potential: {nernst_result['standard_potential_V']:.1f} V")
    print(f"Overpotential: {nernst_result['overpotential_V']:.3f} V")

    results['demonstrations']['nernst_voltage'] = nernst_result

    # Li-ion discharge
    discharge_result = battery.lithium_ion_discharge_curve(
        capacity_Ah=3.0,
        nominal_voltage_V=3.7,
        cutoff_voltage_V=3.0,
        discharge_rate_C=1.0
    )
    print(f"\nLi-ion Battery Discharge:")
    print(f"Capacity: 3.0 Ah")
    print(f"Energy delivered: {discharge_result['energy_delivered_Wh']:.2f} Wh")
    print(f"Discharge time: {discharge_result['discharge_time_hours']:.1f} hours")
    print(f"Discharge current: {discharge_result['discharge_current_A']:.2f} A")

    results['demonstrations']['lithium_ion_discharge'] = discharge_result

    # Ragone plot
    energy_density = np.array([50, 100, 150, 200, 250])
    power_density = np.array([1000, 500, 300, 200, 150])
    ragone_result = battery.ragone_plot(
        energy_density_Wh_per_kg=energy_density,
        power_density_W_per_kg=power_density
    )
    print(f"\nRagone Plot Analysis:")
    print(f"Energy densities: {ragone_result['energy_density_Wh_per_kg']}")
    print(f"Power densities: {ragone_result['power_density_W_per_kg']}")

    results['demonstrations']['ragone_plot'] = ragone_result

    # 3. Fuel Cell Analysis
    print("\n" + "=" * 60)
    print("FUEL CELL ANALYSIS")
    print("=" * 60)

    fuel_cell = FuelCellAnalysis()

    # Polarization curve
    current_density = np.linspace(0, 1.5, 100)
    polar_result = fuel_cell.fuel_cell_voltage(
        current_density_A_per_cm2=current_density,
        temperature_K=353,
        pressure_atm=1.5
    )
    print(f"PEM Fuel Cell Performance:")
    print(f"Reversible voltage: {polar_result['reversible_voltage_V']:.3f} V")
    max_power_idx = np.argmax(polar_result['power_density_W_per_cm2'])
    print(f"Max power density: {polar_result['power_density_W_per_cm2'][max_power_idx]:.3f} W/cm²")
    print(f"At current density: {polar_result['current_density_A_per_cm2'][max_power_idx]:.2f} A/cm²")

    results['demonstrations']['fuel_cell_polarization'] = polar_result

    # Hydrogen consumption
    h2_result = fuel_cell.hydrogen_consumption_rate(
        current_A=100,
        number_of_cells=50,
        utilization=0.95
    )
    print(f"\nHydrogen Consumption:")
    print(f"Flow rate: {h2_result['hydrogen_flow_rate_kg_per_hour']:.4f} kg/hr")
    print(f"Flow rate: {h2_result['hydrogen_flow_rate_L_per_min_STP']:.2f} L/min (STP)")

    results['demonstrations']['hydrogen_consumption'] = h2_result

    # 4. Energy Storage Optimization
    print("\n" + "=" * 60)
    print("ENERGY STORAGE OPTIMIZATION")
    print("=" * 60)

    storage = EnergyStorageOptimization()

    # LCOS
    lcos_result = storage.levelized_cost_of_storage(
        capital_cost_USD=50000,
        capacity_kWh=10,
        power_rating_kW=5,
        cycle_life=5000,
        round_trip_efficiency=0.9,
        discount_rate=0.05,
        years=10
    )
    print(f"Levelized Cost of Storage:")
    print(f"LCOS: ${lcos_result['LCOS_USD_per_kWh']:.2f}/kWh")
    print(f"Capital cost per kWh: ${lcos_result['capital_cost_per_kWh']:.0f}/kWh")
    print(f"Capital cost per kW: ${lcos_result['capital_cost_per_kW']:.0f}/kW")
    print(f"Expected cycles: {lcos_result['expected_cycles']:.0f}")

    results['demonstrations']['lcos'] = lcos_result

    # Charge/discharge optimization
    # Simulate hourly prices (higher during day)
    hours = 24
    prices = 0.10 + 0.05 * np.sin(np.linspace(0, 2*np.pi, hours))
    optim_result = storage.charge_discharge_optimization(
        electricity_price_USD_per_kWh=prices,
        capacity_kWh=10,
        power_rating_kW=5,
        efficiency=0.9
    )
    print(f"\nPrice Arbitrage Optimization:")
    print(f"Total revenue: ${optim_result['total_revenue_USD']:.2f}")
    print(f"Threshold price: ${optim_result['threshold_price']:.3f}/kWh")

    results['demonstrations']['charge_discharge_optimization'] = optim_result

    # Battery degradation
    cycles_array = np.linspace(0, 5000, 50)
    degrad_result = storage.battery_degradation_model(
        cycles=cycles_array,
        temperature_C=25,
        depth_of_discharge=0.8
    )
    print(f"\nBattery Degradation:")
    print(f"End of life cycles: {degrad_result['end_of_life_cycles']:.0f}")
    print(f"Capacity at 2500 cycles: {degrad_result['capacity_retention_percent'][25]:.1f}%")

    results['demonstrations']['battery_degradation'] = degrad_result

    print("\n" + "=" * 60)
    print("RENEWABLE ENERGY LAB DEMO COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = run_all_demos()

    # Save results to JSON
    with open('/Users/noone/QuLabInfinite/renewable_energy_lab_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to: /Users/noone/QuLabInfinite/renewable_energy_lab_results.json")
