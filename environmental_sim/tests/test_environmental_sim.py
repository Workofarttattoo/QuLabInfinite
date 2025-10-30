# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Comprehensive tests for Environmental Simulator
Tests for <0.1% error on controlled parameters
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from environmental_sim import (
    EnvironmentalSimulator,
    TemperatureControl,
    PressureControl,
    AtmosphereControl,
    MechanicalForces,
    FluidFlow,
    RadiationEnvironment,
    MultiPhysicsCoupling,
    EnvironmentController,
)


class TestTemperatureControl(unittest.TestCase):
    """Test temperature control system."""

    def setUp(self):
        self.temp = TemperatureControl(precision=0.001)

    def test_temperature_setting_celsius(self):
        """Test setting temperature in Celsius."""
        self.temp.set_temperature(25, unit="C")
        temp_c = self.temp.get_temperature(unit="C")
        self.assertAlmostEqual(temp_c, 25.0, delta=0.001)

    def test_temperature_setting_kelvin(self):
        """Test setting temperature in Kelvin."""
        self.temp.set_temperature(300, unit="K")
        temp_k = self.temp.get_temperature(unit="K")
        self.assertAlmostEqual(temp_k, 300.0, delta=0.001)

    def test_temperature_conversion(self):
        """Test temperature unit conversions."""
        self.temp.set_temperature(0, unit="C")
        temp_k = self.temp.get_temperature(unit="K")
        self.assertAlmostEqual(temp_k, 273.15, delta=0.001)

    def test_cryogenic_temperature(self):
        """Test cryogenic temperature range."""
        self.temp.set_temperature(-200, unit="C")
        temp_c = self.temp.get_temperature(unit="C")
        self.assertAlmostEqual(temp_c, -200.0, delta=0.001)

    def test_high_temperature(self):
        """Test high temperature range (up to 10,000°C)."""
        self.temp.set_temperature(5000, unit="C")
        temp_c = self.temp.get_temperature(unit="C")
        self.assertAlmostEqual(temp_c, 5000.0, delta=0.001)

    def test_absolute_zero_limit(self):
        """Test that absolute zero is the lower limit."""
        with self.assertRaises(ValueError):
            self.temp.set_temperature(-300, unit="C")

    def test_temperature_gradient(self):
        """Test temperature gradient."""
        self.temp.set_temperature(25, unit="C")
        self.temp.set_gradient((10, 0, 0), unit="K/m")  # 10 K/m in x-direction

        temp_origin = self.temp.get_temperature((0, 0, 0), unit="K")
        temp_1m = self.temp.get_temperature((1, 0, 0), unit="K")

        self.assertAlmostEqual(temp_1m - temp_origin, 10.0, delta=0.001)

    def test_heat_source(self):
        """Test heat source addition."""
        self.temp.set_temperature(25, unit="C")
        source_id = self.temp.add_heat_source((0, 0, 0), power=1000, radius=0.1)
        self.assertIsInstance(source_id, int)

        # Temperature should be higher near heat source
        temp_at_source = self.temp.get_temperature((0, 0, 0), unit="C")
        temp_far = self.temp.get_temperature((10, 10, 10), unit="C")
        self.assertGreater(temp_at_source, temp_far)

    def test_precision(self):
        """Test temperature precision (±0.001 K)."""
        self.temp.set_temperature(25.0001234, unit="C")
        temp = self.temp.get_temperature(unit="C")
        # Should be quantized to precision
        self.assertAlmostEqual(temp, 25.0, delta=0.001)


class TestPressureControl(unittest.TestCase):
    """Test pressure control system."""

    def setUp(self):
        self.pressure = PressureControl(precision_percent=0.01)

    def test_pressure_setting(self):
        """Test setting pressure in various units."""
        self.pressure.set_pressure(1.0, unit="bar")
        p_bar = self.pressure.get_pressure(unit="bar")
        self.assertAlmostEqual(p_bar, 1.0, delta=1e-4)

    def test_vacuum_conditions(self):
        """Test vacuum pressure levels."""
        self.pressure.set_vacuum_level(1e-6, unit="torr")
        p_torr = self.pressure.get_pressure(unit="torr")
        self.assertAlmostEqual(p_torr, 1e-6, delta=1e-8)

    def test_high_pressure(self):
        """Test high pressure (up to 1,000,000 bar)."""
        self.pressure.set_pressure(100000, unit="bar")
        p_bar = self.pressure.get_pressure(unit="bar")
        self.assertAlmostEqual(p_bar, 100000, delta=100)  # 0.1% precision

    def test_pressure_unit_conversion(self):
        """Test pressure unit conversions."""
        self.pressure.set_pressure(1.01325, unit="bar")  # 1 atm
        p_atm = self.pressure.get_pressure(unit="atm")
        self.assertAlmostEqual(p_atm, 1.0, delta=0.001)

        p_pa = self.pressure.get_pressure(unit="Pa")
        self.assertAlmostEqual(p_pa, 101325, delta=100)

    def test_supercritical_co2(self):
        """Test supercritical CO2 detection."""
        self.pressure.set_pressure(100, unit="bar")  # Above critical pressure
        is_supercritical = self.pressure.is_supercritical(320, substance="CO2")  # Above critical temp
        self.assertTrue(is_supercritical)

    def test_hydrostatic_pressure(self):
        """Test hydrostatic pressure calculation."""
        depth = 1000  # meters
        p_total = self.pressure.calculate_hydrostatic_pressure(depth, fluid_density=1000)
        # P = P_atm + ρgh ≈ 1 + 98 bar = 99 bar
        self.assertGreater(p_total, 90)  # Should be ~99 bar

    def test_pressure_precision(self):
        """Test pressure precision (±0.01%)."""
        self.pressure.set_pressure(100.0, unit="bar")
        p = self.pressure.get_pressure(unit="bar")
        # 0.01% of 100 bar = 0.01 bar
        self.assertAlmostEqual(p, 100.0, delta=0.01)


class TestAtmosphereControl(unittest.TestCase):
    """Test atmosphere control system."""

    def setUp(self):
        self.atmo = AtmosphereControl()

    def test_standard_air(self):
        """Test standard air composition."""
        self.atmo.set_standard_atmosphere("air")
        n2 = self.atmo.get_composition("N2")
        o2 = self.atmo.get_composition("O2")

        self.assertAlmostEqual(n2, 78.084, delta=0.1)
        self.assertAlmostEqual(o2, 20.946, delta=0.1)

    def test_inert_atmosphere(self):
        """Test inert atmosphere (nitrogen)."""
        self.atmo.set_standard_atmosphere("nitrogen")
        n2 = self.atmo.get_composition("N2")
        self.assertAlmostEqual(n2, 100.0, delta=0.01)

    def test_oxidizing_atmosphere(self):
        """Test oxidizing atmosphere."""
        self.atmo.set_oxidizing_atmosphere(oxygen_percent=100)
        o2 = self.atmo.get_composition("O2")
        self.assertAlmostEqual(o2, 100.0, delta=0.01)

    def test_reducing_atmosphere(self):
        """Test reducing atmosphere."""
        self.atmo.set_reducing_atmosphere(hydrogen_percent=5.0)
        h2 = self.atmo.get_composition("H2")
        self.assertAlmostEqual(h2, 5.0, delta=0.01)

    def test_partial_pressure_calculation(self):
        """Test partial pressure calculation."""
        self.atmo.set_standard_atmosphere("air")
        partial_pressures = self.atmo.calculate_partial_pressures(1.0, unit="bar")

        # O2 partial pressure should be ~0.21 bar
        self.assertAlmostEqual(partial_pressures['O2'], 0.21, delta=0.01)

    def test_humidity_control(self):
        """Test humidity control (0-100% RH)."""
        self.atmo.set_humidity(50.0)
        rh = self.atmo.get_humidity()
        self.assertAlmostEqual(rh, 50.0, delta=0.1)

    def test_contaminant_tracking(self):
        """Test contaminant tracking (ppm level)."""
        self.atmo.add_contaminant("CO", 100.0)  # 100 ppm CO
        co_ppm = self.atmo.get_contaminant("CO")
        self.assertAlmostEqual(co_ppm, 100.0, delta=0.1)

    def test_contaminant_decay(self):
        """Contaminant concentrations should decay according to half-life."""
        self.atmo.add_contaminant("NO2", 120.0, half_life_hours=1.0, accumulate=False)
        self.atmo.update_contaminants(3600.0)  # 1 hour
        remaining = self.atmo.get_contaminant("NO2")
        self.assertAlmostEqual(remaining, 60.0, delta=1.0)

    def test_contaminant_removal_efficiency(self):
        """Removal efficiency should reduce concentration over exposure."""
        self.atmo.add_contaminant("VOC", 80.0, removal_efficiency=0.5, accumulate=False)
        self.atmo.update_contaminants(3600.0)
        remaining = self.atmo.get_contaminant("VOC")
        self.assertAlmostEqual(remaining, 40.0, delta=1.0)

    def test_contaminant_profile_export(self):
        """Detailed contaminant profiles should include metadata."""
        self.atmo.add_contaminant("SO2", 25.0, half_life_hours=2.0, removal_efficiency=0.3, accumulate=False)
        profile = self.atmo.get_contaminant_profile("SO2")
        self.assertIn("half_life_hours", profile)
        self.assertEqual(profile["name"], "SO2")
        self.assertIn("total_removed_ppm", profile)

    def test_breathability(self):
        """Test breathability check."""
        self.atmo.set_standard_atmosphere("air")
        is_breathable = self.atmo.is_breathable(partial_pressure_o2=0.21)
        self.assertTrue(is_breathable)

        # Pure nitrogen is not breathable
        self.atmo.set_standard_atmosphere("nitrogen")
        is_breathable = self.atmo.is_breathable(partial_pressure_o2=0.0)
        self.assertFalse(is_breathable)


class TestMechanicalForces(unittest.TestCase):
    """Test mechanical forces system."""

    def setUp(self):
        self.mech = MechanicalForces()

    def test_earth_gravity(self):
        """Test Earth gravity (1g)."""
        self.mech.set_gravity(g_factor=1.0)
        g = self.mech.get_gravity()
        self.assertAlmostEqual(np.linalg.norm(g), 9.80665, delta=0.001)

    def test_microgravity(self):
        """Test microgravity (0g)."""
        self.mech.set_gravity(g_factor=0.0)
        g = self.mech.get_gravity()
        self.assertAlmostEqual(np.linalg.norm(g), 0.0, delta=0.001)

    def test_high_gravity(self):
        """Test high gravity (up to 100g)."""
        self.mech.set_gravity(g_factor=10.0)
        g = self.mech.get_gravity()
        self.assertAlmostEqual(np.linalg.norm(g), 98.0665, delta=0.01)

    def test_centrifugal_force(self):
        """Test centrifugal force calculation."""
        # Rotating at 1 rad/s around z-axis
        self.mech.set_rotation(angular_velocity=1.0, axis=(0, 0, 1), center=(0, 0, 0))
        force = self.mech.get_centrifugal_force((1, 0, 0), mass=1.0)

        # F = m*ω²*r = 1 * 1² * 1 = 1 N (radial)
        self.assertAlmostEqual(np.linalg.norm(force), 1.0, delta=0.01)

    def test_sinusoidal_vibration(self):
        """Test sinusoidal vibration."""
        vib_id = self.mech.add_vibration("sinusoidal", frequency=10, amplitude=0.001,
                                        direction=(0, 0, 1))
        displacement = self.mech.get_vibration_displacement(0.0, vibration_id=vib_id)
        self.assertIsInstance(displacement, np.ndarray)

    def test_random_vibration(self):
        """Test random vibration."""
        vib_id = self.mech.add_vibration("random", frequency=0, amplitude=0.001,
                                        direction=(0, 0, 1))
        displacement = self.mech.get_vibration_displacement(0.0, vibration_id=vib_id)
        self.assertIsInstance(displacement, np.ndarray)

    def test_acoustic_wave(self):
        """Test acoustic wave."""
        source_id = self.mech.add_acoustic_wave(frequency=1000, amplitude=100,
                                               origin=(0, 0, 0))
        pressure = self.mech.get_acoustic_pressure((1, 0, 0), time=0.0)
        self.assertIsInstance(pressure, float)


class TestFluidFlow(unittest.TestCase):
    """Test fluid flow system."""

    def setUp(self):
        self.fluid = FluidFlow()

    def test_wind_setting(self):
        """Test wind velocity setting."""
        self.fluid.set_wind((10, 0, 0), unit="m/s")
        wind = self.fluid.get_wind(unit="m/s")
        self.assertAlmostEqual(wind[0], 10.0, delta=0.01)

    def test_wind_unit_conversion(self):
        """Test wind unit conversions."""
        self.fluid.set_wind((10, 0, 0), unit="m/s")
        wind_mph = self.fluid.get_wind(unit="mph")
        # 10 m/s ≈ 22.37 mph
        self.assertAlmostEqual(wind_mph[0], 22.37, delta=0.1)

    def test_high_wind_speed(self):
        """Test high wind speed (up to 500 mph)."""
        self.fluid.set_wind((200, 0, 0), unit="mph")
        wind_mph = self.fluid.get_wind(unit="mph")
        self.assertAlmostEqual(wind_mph[0], 200.0, delta=0.1)

    def test_turbulence(self):
        """Test turbulence intensity."""
        self.fluid.set_wind((10, 0, 0), unit="m/s")
        self.fluid.set_turbulence(intensity=0.5, length_scale=1.0)

        # Turbulent wind should vary
        wind1 = self.fluid.get_wind(unit="m/s")
        wind2 = self.fluid.get_wind(unit="m/s")
        # Due to randomness, they should differ
        self.assertIsInstance(wind1, np.ndarray)

    def test_reynolds_number(self):
        """Test Reynolds number calculation."""
        self.fluid.set_wind((10, 0, 0), unit="m/s")
        Re = self.fluid.calculate_reynolds_number(characteristic_length=1.0)
        self.assertGreater(Re, 0)

        # Should be turbulent flow (Re > 4000)
        regime = self.fluid.get_flow_regime()
        self.assertEqual(regime, "turbulent")

    def test_vortex(self):
        """Test vortex structure."""
        vortex_id = self.fluid.add_vortex(center=(0, 0, 0), axis=(0, 0, 1),
                                         circulation=10.0, core_radius=0.1)
        velocity = self.fluid.get_vortex_velocity((0.2, 0, 0), vortex_id=vortex_id)
        self.assertIsInstance(velocity, np.ndarray)
        self.assertGreater(np.linalg.norm(velocity), 0)

    def test_drag_force(self):
        """Test drag force calculation."""
        force = self.fluid.calculate_drag_force(velocity=(10, 0, 0),
                                               drag_coefficient=0.5,
                                               reference_area=1.0)
        self.assertIsInstance(force, np.ndarray)
        self.assertLess(force[0], 0)  # Opposing motion


class TestRadiationEnvironment(unittest.TestCase):
    """Test radiation environment system."""

    def setUp(self):
        self.rad = RadiationEnvironment()

    def test_em_radiation(self):
        """Test electromagnetic radiation."""
        source_id = self.rad.add_em_radiation("UV", intensity=50, wavelength=300e-9)
        intensity = self.rad.get_em_intensity()
        self.assertAlmostEqual(intensity, 50, delta=0.1)

    def test_ionizing_radiation(self):
        """Test ionizing radiation."""
        source_id = self.rad.add_ionizing_radiation("gamma", dose_rate=0.1, energy=1.0,
                                                    origin=(0, 0, 0))
        dose_rate = self.rad.get_ionizing_dose_rate((1, 0, 0))
        self.assertGreater(dose_rate, 0)

    def test_dose_accumulation(self):
        """Test radiation dose accumulation."""
        self.rad.add_ionizing_radiation("gamma", dose_rate=0.1, energy=1.0,
                                       origin=(0, 0, 0))
        dose = self.rad.accumulate_dose((1, 0, 0), duration=10.0)  # 10 hours
        self.assertGreater(dose, 0)

    def test_shielding(self):
        """Test radiation shielding."""
        self.rad.add_em_radiation("gamma", intensity=100, frequency=1e18)
        shield_id = self.rad.add_shield("lead", thickness=0.01,
                                       position=(0, 0, 0), normal=(0, 0, 1))

        # Intensity should be reduced by shielding
        intensity_unshielded = 100
        intensity_shielded = self.rad.get_em_intensity((0, 0, 1))
        self.assertLess(intensity_shielded, intensity_unshielded)


class TestMultiPhysicsCoupling(unittest.TestCase):
    """Test multi-physics coupling system."""

    def setUp(self):
        self.coupling = MultiPhysicsCoupling()

    def test_thermal_expansion(self):
        """Test thermal expansion."""
        alpha = 1e-5  # K⁻¹ (typical for metals)
        self.coupling.enable_thermo_mechanical_coupling(alpha)

        strain = self.coupling.calculate_thermal_strain(temperature_change=100)
        # ε = α * ΔT = 1e-5 * 100 = 1e-3
        self.assertAlmostEqual(strain, 1e-3, delta=1e-6)

    def test_thermal_stress(self):
        """Test thermal stress calculation."""
        alpha = 1e-5  # K⁻¹
        E = 200e9  # Pa (Young's modulus)
        self.coupling.enable_thermo_mechanical_coupling(alpha)

        stress = self.coupling.calculate_thermal_stress(temperature=400, elastic_modulus=E,
                                                       reference_temperature=300)
        # σ = E * α * ΔT = 200e9 * 1e-5 * 100 = 200 MPa
        self.assertAlmostEqual(stress, 200e6, delta=1e6)

    def test_joule_heating(self):
        """Test Joule heating calculation."""
        self.coupling.enable_electro_thermal_coupling()
        heat_rate = self.coupling.calculate_joule_heating(current=10, resistance=1.0,
                                                         volume=0.001)
        # P = I²R = 10² * 1 = 100 W, Q̇ = 100/0.001 = 100,000 W/m³
        self.assertAlmostEqual(heat_rate, 100000, delta=100)


class TestEnvironmentController(unittest.TestCase):
    """Test environment controller integration."""

    def setUp(self):
        self.controller = EnvironmentController(update_rate=100)

    def test_get_conditions_at_position(self):
        """Test getting all conditions at a position."""
        conditions = self.controller.get_conditions_at_position((0, 0, 0))

        self.assertIn('temperature_C', conditions)
        self.assertIn('pressure_bar', conditions)
        self.assertIn('wind_velocity_m_s', conditions)
        self.assertIn('gravity_m_s2', conditions)

    def test_preset_stp(self):
        """Test STP preset."""
        self.controller.set_preset_environment("STP")
        conditions = self.controller.get_conditions_at_position((0, 0, 0))

        self.assertAlmostEqual(conditions['temperature_C'], 25, delta=0.1)
        self.assertAlmostEqual(conditions['pressure_bar'], 1.01325, delta=0.01)

    def test_preset_vacuum(self):
        """Test vacuum preset."""
        self.controller.set_preset_environment("vacuum")
        conditions = self.controller.get_conditions_at_position((0, 0, 0))

        self.assertLess(conditions['pressure_Pa'], 1.0)  # High vacuum

    def test_preset_leo(self):
        """Test LEO preset."""
        self.controller.set_preset_environment("LEO")
        conditions = self.controller.get_conditions_at_position((0, 0, 0))

        self.assertLess(conditions['temperature_C'], 0)  # Cold
        self.assertEqual(conditions['gravity_m_s2'], [0, 0, 0])  # Microgravity

    def test_update_cycle(self):
        """Test environment update cycle."""
        initial_time = self.controller.get_simulation_time()
        self.controller.update(dt=0.01)
        final_time = self.controller.get_simulation_time()

        self.assertGreater(final_time, initial_time)

    def test_material_stress_calculation(self):
        """Test integrated material stress calculation."""
        material_props = {
            'elastic_modulus': 200e9,  # Pa
            'thermal_expansion': 1e-5,  # K⁻¹
        }

        self.controller.temperature.set_temperature(100, unit="C")
        self.controller.pressure.set_pressure(10, unit="bar")

        stress = self.controller.calculate_material_stress(material_props, (0, 0, 0))

        self.assertIn('thermal_stress_Pa', stress)
        self.assertIn('pressure_stress_Pa', stress)
        self.assertIn('total_stress_Pa', stress)

    def test_contaminant_decay_during_update(self):
        """Controller update should decay contaminants via atmosphere control."""
        self.controller.atmosphere.add_contaminant("CO2", 100.0, half_life_hours=1.0, accumulate=False)
        self.controller.update(dt=3600.0)
        concentration = self.controller.atmosphere.get_contaminant("CO2")
        self.assertLess(concentration, 100.0)
        self.assertAlmostEqual(concentration, 50.0, delta=1.5)

    def test_corrosion_record_and_state(self):
        """Corrosion multipliers should accumulate and be exposed via state."""
        self.controller.set_corrosion_baseline("Carbon Steel", 1.0)
        self.controller.record_corrosion_effect(
            "Carbon Steel",
            multiplier=1.2,
            exposure_hours=4.0,
            source={"name": "acidic_vapor"}
        )
        self.controller.update(dt=7200.0)  # 2 hours
        state = self.controller.get_corrosion_state("Carbon Steel")
        self.assertAlmostEqual(state["adjusted_rate_mm_per_year"], 1.2, delta=1e-6)
        self.assertGreater(state["cumulative_loss_mm"], 0.0)
        self.assertGreaterEqual(state["total_exposure_hours"], 4.0)
        self.assertTrue(state["sources"])

        full_state = self.controller.get_full_state()
        self.assertIn("corrosion", full_state)


class TestEnvironmentalSimulator(unittest.TestCase):
    """Test high-level environmental simulator API."""

    def test_aerogel_simulation(self):
        """Test aerogel simulation setup."""
        from environmental_sim import create_aerogel_simulation

        sim = create_aerogel_simulation(temp_c=-200, pressure_bar=0.001, wind_mph=30)
        conditions = sim.get_conditions()

        self.assertAlmostEqual(conditions['temperature_C'], -200, delta=0.1)
        self.assertAlmostEqual(conditions['pressure_bar'], 0.001, delta=0.0001)

    def test_diamond_anvil_simulation(self):
        """Test diamond anvil cell simulation."""
        from environmental_sim import create_diamond_anvil_simulation

        sim = create_diamond_anvil_simulation(pressure_gpa=100, temp_k=3000)
        conditions = sim.get_conditions()

        self.assertAlmostEqual(conditions['temperature_K'], 3000, delta=1)
        self.assertGreater(conditions['pressure_Pa'], 1e10)  # 100 GPa

    def test_leo_simulation(self):
        """Test LEO simulation."""
        from environmental_sim import create_leo_simulation

        sim = create_leo_simulation(altitude_km=400)
        conditions = sim.get_conditions()

        self.assertLess(conditions['pressure_Pa'], 1e-3)  # Vacuum


class TestPrecisionRequirements(unittest.TestCase):
    """Test <0.1% error requirement on controlled parameters."""

    def test_temperature_precision_requirement(self):
        """Verify temperature control meets <0.1% error."""
        temp = TemperatureControl(precision=0.001)

        test_temps = [25, 100, 500, -100, 3000]
        for target in test_temps:
            temp.set_temperature(target, unit="C")
            measured = temp.get_temperature(unit="C")

            if target != 0:
                error_percent = abs((measured - target) / target) * 100
                self.assertLess(error_percent, 0.1, f"Temperature error {error_percent}% exceeds 0.1%")

    def test_pressure_precision_requirement(self):
        """Verify pressure control meets <0.1% error."""
        pressure = PressureControl(precision_percent=0.01)

        test_pressures = [1, 10, 100, 1000, 10000]
        for target in test_pressures:
            pressure.set_pressure(target, unit="bar")
            measured = pressure.get_pressure(unit="bar")

            error_percent = abs((measured - target) / target) * 100
            self.assertLess(error_percent, 0.1, f"Pressure error {error_percent}% exceeds 0.1%")


def run_all_tests():
    """Run all environmental simulator tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTemperatureControl))
    suite.addTests(loader.loadTestsFromTestCase(TestPressureControl))
    suite.addTests(loader.loadTestsFromTestCase(TestAtmosphereControl))
    suite.addTests(loader.loadTestsFromTestCase(TestMechanicalForces))
    suite.addTests(loader.loadTestsFromTestCase(TestFluidFlow))
    suite.addTests(loader.loadTestsFromTestCase(TestRadiationEnvironment))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiPhysicsCoupling))
    suite.addTests(loader.loadTestsFromTestCase(TestEnvironmentController))
    suite.addTests(loader.loadTestsFromTestCase(TestEnvironmentalSimulator))
    suite.addTests(loader.loadTestsFromTestCase(TestPrecisionRequirements))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    print("="*80)
    print("Environmental Simulator - Comprehensive Test Suite")
    print("="*80)
    print()

    result = run_all_tests()

    print()
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    print("="*80)
