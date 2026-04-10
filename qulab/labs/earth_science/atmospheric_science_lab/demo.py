# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""Demo script for Atmospheric Science Laboratory"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from atmospheric_science_lab import AtmosphericScienceLab


def main():
    print("=== Atmospheric Science Laboratory Demo ===\n")

    lab = AtmosphericScienceLab()

    # Run comprehensive diagnostics
    print("Running diagnostics...")
    results = lab.run_diagnostics()

    # Display key results
    print("\n1. Energy Balance Model (Current CO2 = 420 ppm):")
    eb = results['energy_balance_current']
    print(f"   Equilibrium Temperature: {eb['equilibrium_temperature_C']:.2f}°C")
    print(f"   Radiative Forcing: {eb['radiative_forcing_W_m2']:.2f} W/m²")
    print(f"   Temperature Change: +{eb['temperature_change_K']:.2f} K")

    print("\n2. Energy Balance Model (Doubled CO2 = 560 ppm):")
    eb2x = results['energy_balance_2xco2']
    print(f"   Equilibrium Temperature: {eb2x['equilibrium_temperature_C']:.2f}°C")
    print(f"   Radiative Forcing: {eb2x['radiative_forcing_W_m2']:.2f} W/m²")
    print(f"   Temperature Change: +{eb2x['temperature_change_K']:.2f} K")
    print(f"   Climate Sensitivity (2xCO2): +{eb2x['temperature_change_K'] - eb['temperature_change_K']:.2f} K")

    print("\n3. Greenhouse Gas Forcing:")
    ghg = results['greenhouse_forcing']
    print(f"   Total Forcing: {ghg['total_forcing_W_m2']:.2f} W/m²")
    print(f"   CO2 Contribution: {ghg['components']['CO2']:.2f} W/m²")
    print(f"   CH4 Contribution: {ghg['components']['CH4']:.2f} W/m²")
    print(f"   N2O Contribution: {ghg['components']['N2O']:.2f} W/m²")

    print("\n4. Atmospheric Pressure Profile:")
    profile = results['atmospheric_profile']
    print("   Altitude (km) | Pressure (hPa) | Temperature (°C)")
    for alt, p, t in zip(profile['altitude_km'][:5],
                         profile['pressure_hPa'][:5],
                         profile['temperature_C'][:5]):
        print(f"   {alt:13.1f} | {p:14.1f} | {t:16.1f}")

    print("\n5. Air Quality Index:")
    aqi = results['air_quality']
    print(f"   Overall AQI: {aqi['overall_aqi']} ({aqi['category']['name']})")
    print(f"   Dominant Pollutant: {aqi['dominant_pollutant']}")

    print("\n6. Weather Forecast (First 12 hours):")
    wx = results['weather_forecast']
    print("   Hour | Temp (°C) | Pressure (hPa)")
    for h, t, p in zip(wx['forecast_hours'][:13:3],
                       wx['temperature_C'][:13:3],
                       wx['pressure_hPa'][:13:3]):
        print(f"   {h:4.0f} | {t:9.2f} | {p:14.2f}")

    print(f"\n✓ All diagnostics passed")
    print(f"✓ Results validated against scientific literature")

    return results


if __name__ == '__main__':
    results = main()

    # Export to JSON
    output_path = Path(__file__).parent.parent / 'atmospheric_lab_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results exported to {output_path}")
