#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Comprehensive Materials Lab Tests
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import time
from materials_lab import MaterialsLab
from validation.results_validator import ValidationStatus


class TestMaterialsDatabase(unittest.TestCase):
    """Test materials database"""

    @classmethod
    def setUpClass(cls):
        cls.lab = MaterialsLab()

    def test_database_loaded(self):
        """Test database is loaded with 1000+ materials"""
        count = self.lab.db.get_count()
        self.assertGreaterEqual(count, 1000, "Database should have 1000+ materials")

    def test_airloy_x103_exists(self):
        """Test Airloy X103 aerogel exists with complete properties"""
        airloy = self.lab.get_material("Airloy X103")
        self.assertIsNotNone(airloy, "Airloy X103 must exist")
        self.assertEqual(airloy.name, "Airloy X103")
        self.assertEqual(airloy.category, "nanomaterial")
        self.assertEqual(airloy.subcategory, "aerogel")

        # Check critical properties
        self.assertEqual(airloy.density, 144, "Density should be 144 kg/m³")
        self.assertAlmostEqual(airloy.thermal_conductivity, 0.014, places=3,
                              msg="Thermal conductivity should be 14 mW/(m·K)")
        self.assertGreater(airloy.tensile_strength, 0, "Must have tensile strength")
        self.assertGreater(airloy.compressive_strength, 0, "Must have compressive strength")

    def test_lookup_speed(self):
        """Test lookup is <10ms"""
        materials = ["Airloy X103", "Ti-6Al-4V", "SS 304", "PEEK", "Carbon Fiber Epoxy"]

        for mat_name in materials:
            start = time.time()
            mat = self.lab.get_material(mat_name)
            end = time.time()

            lookup_time = (end - start) * 1000
            self.assertLess(lookup_time, 10, f"Lookup should be <10ms, got {lookup_time:.2f}ms")
            self.assertIsNotNone(mat, f"Material {mat_name} not found")

    def test_search_materials(self):
        """Test material search"""
        # Find lightweight strong materials
        results = self.lab.search_materials(
            min_density=100,
            max_density=2000,
            min_strength=100
        )
        self.assertGreater(len(results), 0, "Should find lightweight strong materials")

    def test_categories(self):
        """Test categories"""
        categories = self.lab.list_categories()
        expected = ["metal", "ceramic", "polymer", "composite", "nanomaterial"]

        for cat in expected:
            self.assertIn(cat, categories, f"Category {cat} should exist")

    def test_inconel_superalloy_profile(self):
        """Supplemental superalloy entry should expose high-temperature capability."""
        inconel = self.lab.get_material("Inconel 718")
        self.assertIsNotNone(inconel, "Inconel 718 must be present in the database")
        self.assertEqual(inconel.subcategory, "nickel_superalloy")
        self.assertGreater(inconel.tensile_strength, 1000)
        self.assertGreater(inconel.max_service_temp, 1100)
        self.assertEqual(inconel.corrosion_resistance, "excellent")

    def test_polyimide_temperature_window(self):
        """Kapton should retain data for extreme temperature swings."""
        kapton = self.lab.get_material("Kapton HN Polyimide")
        self.assertIsNotNone(kapton)
        self.assertEqual(kapton.category, "polymer")
        self.assertGreater(kapton.glass_transition_temp, 600)
        self.assertLess(kapton.thermal_conductivity, 0.2)
        self.assertGreater(kapton.dielectric_constant, 3.0)

    def test_adhesive_dielectric_properties(self):
        """Structural adhesive entry should expose electrical and mechanical properties."""
        adhesive = self.lab.get_material("Loctite EA 9396")
        self.assertIsNotNone(adhesive)
        self.assertEqual(adhesive.subcategory, "epoxy_adhesive")
        self.assertGreater(adhesive.dielectric_strength, 10.0)
        self.assertGreater(adhesive.glass_transition_temp, 400)
        self.assertGreater(adhesive.tensile_strength, 50)

    def test_carbon_carbon_high_temperature(self):
        """Carbon-carbon composite should report ultra-high service temperatures."""
        ccarb = self.lab.get_material("Carbon-Carbon Composite (2D)")
        self.assertIsNotNone(ccarb)
        self.assertEqual(ccarb.category, "composite")
        self.assertGreater(ccarb.max_service_temp, 3000)
        self.assertLess(ccarb.thermal_expansion, 5e-6)
        self.assertGreater(ccarb.fracture_toughness, 10)

    def test_tungsten_carbide_properties(self):
        """WC-Co cermet should expose extreme hardness and conductivity."""
        wc = self.lab.get_material("Tungsten Carbide WC-Co K20")
        self.assertIsNotNone(wc)
        self.assertGreater(wc.hardness_vickers, 1500)
        self.assertGreater(wc.thermal_conductivity, 50)
        self.assertLess(wc.thermal_expansion, 6e-6)

    def test_graphene_aerogel_density(self):
        """Graphene aerogel should be ultralight with low conductivity."""
        ga = self.lab.get_material("Graphene Aerogel")
        self.assertIsNotNone(ga)
        self.assertLess(ga.density, 20)
        self.assertLess(ga.thermal_conductivity, 0.02)
        self.assertLess(ga.electrical_conductivity, 1.0)

    def test_material_safety_lookup(self):
        """Safety database should provide MSDS style guidance."""
        safety = self.lab.get_material_safety("Hastelloy X")
        self.assertIsNotNone(safety)
        self.assertIn("signal_word", safety)
        self.assertIn("ppe", safety)
        profile = self.lab.get_material_profile("Hastelloy X")
        self.assertIn("safety", profile)

    def test_hastelloy_high_temp(self):
        """Hastelloy X should retain strength at high temperature."""
        hx = self.lab.get_material("Hastelloy X")
        self.assertIsNotNone(hx)
        self.assertGreater(hx.yield_strength, 600)
        self.assertGreater(hx.max_service_temp, 1300)
        self.assertEqual(hx.corrosion_resistance, "excellent")

    def test_magnesium_lightweight(self):
        """Magnesium AZ31B should be lighter than aluminum alloys."""
        az = self.lab.get_material("Magnesium AZ31B")
        self.assertIsNotNone(az)
        self.assertLess(az.density, 2000)
        self.assertGreater(az.tensile_strength, 250)
        self.assertLess(az.thermal_conductivity, 100)

    def test_peek_polymer_window(self):
        """PEEK 450G should have high glass transition temperature."""
        peek = self.lab.get_material("PEEK 450G")
        self.assertIsNotNone(peek)
        self.assertGreater(peek.glass_transition_temp, 400)
        self.assertGreater(peek.tensile_strength, 90)
        self.assertLess(peek.thermal_conductivity, 0.3)

    def test_ybco_superconductor_properties(self):
        """YBCO superconductor should require cryogenic temperatures."""
        ybco = self.lab.get_material("YBCO Superconductor")
        self.assertIsNotNone(ybco)
        self.assertLess(ybco.max_service_temp, 150)
        self.assertEqual(ybco.electrical_resistivity, 0.0)
        self.assertGreater(ybco.dielectric_constant, 30)

    def test_lithium_electrolyte_conductivity(self):
        """Battery electrolyte should report ionic conductivity."""
        electrolyte = self.lab.get_material("Lithium Hexafluorophosphate Electrolyte")
        self.assertIsNotNone(electrolyte)
        self.assertGreater(electrolyte.electrical_conductivity, 40)
        self.assertGreater(electrolyte.dielectric_constant, 15)
        self.assertLess(electrolyte.max_service_temp, 360)


class TestMaterialTesting(unittest.TestCase):
    """Test material testing modules"""

    @classmethod
    def setUpClass(cls):
        cls.lab = MaterialsLab()

    def test_tensile_test_steel(self):
        """Test tensile test on steel"""
        result = self.lab.tensile_test("SS 304", max_strain=0.3)

        self.assertTrue(result.success)
        self.assertEqual(result.test_type, "tensile")
        self.assertIn("youngs_modulus", result.data)
        self.assertIn("yield_strength", result.data)
        self.assertIn("ultimate_strength", result.data)
        self.assertIn("uncertainty", result.data)
        self.assertIn("yield_strength", result.data["uncertainty"])

        # Check values are reasonable
        E = result.data["youngs_modulus"]
        self.assertGreater(E, 150000, "Steel modulus should be >150 GPa")
        self.assertLess(E, 250000, "Steel modulus should be <250 GPa")

    def test_compression_test(self):
        """Test compression test"""
        result = self.lab.compression_test("Alumina 99.5%", max_strain=0.1)

        self.assertTrue(result.success)
        self.assertIn("compressive_strength", result.data)

    def test_fatigue_test(self):
        """Test fatigue test"""
        result = self.lab.fatigue_test("Al 7075-T6")

        self.assertTrue(result.success)
        self.assertIn("stress_amplitude", result.data)
        self.assertIn("cycles_to_failure", result.data)
        self.assertIn("fatigue_limit", result.data)

    def test_impact_test(self):
        """Test impact test"""
        result = self.lab.impact_test("Ti-6Al-4V", temperature=298.15)

        self.assertTrue(result.success)
        self.assertIn("impact_energy", result.data)
        self.assertGreater(result.data["impact_energy"], 0)

    def test_hardness_test(self):
        """Test hardness test"""
        result = self.lab.hardness_test("Tool Steel D2")

        self.assertTrue(result.success)
        self.assertIn("vickers", result.data)
        self.assertIn("rockwell_c", result.data)

    def test_thermal_conductivity(self):
        """Test thermal conductivity measurement"""
        result = self.lab.thermal_test("Cu C11000", test_type="conductivity")

        self.assertTrue(result.success)
        self.assertIn("thermal_conductivity", result.data)
        # Copper has very high conductivity
        self.assertGreater(result.data["thermal_conductivity"], 300)

    def test_corrosion_test(self):
        """Test corrosion test"""
        result = self.lab.corrosion_test("SS 316", test_type="salt_spray", duration_hours=1000)

        self.assertTrue(result.success)
        self.assertIn("corrosion_rate_mm_per_year", result.data)
        # Stainless should have low corrosion rate
        self.assertLess(result.data["corrosion_rate_mm_per_year"], 50)


class TestEnvironmentalTesting(unittest.TestCase):
    """Test environmental testing - CRITICAL FOR AIRLOY X103"""

    @classmethod
    def setUpClass(cls):
        cls.lab = MaterialsLab()

    def test_airloy_extreme_cold_30mph(self):
        """
        Test Airloy X103 at -200°C with 30 mph wind
        THIS IS THE KEY TEST FROM USER REQUEST
        """
        result = self.lab.environmental_test(
            "Airloy X103",
            temperature=73,  # -200°C in Kelvin
            wind_speed=13.4,  # 30 mph = 13.4 m/s
            duration_hours=24
        )

        # Print detailed results
        print("\n" + "="*70)
        print("AIRLOY X103 EXTREME COLD TEST RESULTS")
        print("="*70)
        print(f"Material: {result.material_name}")
        print(f"Status: {result.data['status']}")
        print(f"\nConditions:")
        print(f"  Temperature: {result.data['temperature_celsius']:.0f}°C ({result.data['temperature']:.0f} K)")
        print(f"  Wind Speed: {result.data['wind_speed_mph']:.1f} mph ({result.data['wind_speed_m_s']:.1f} m/s)")
        print(f"  Duration: {result.data['duration_hours']:.0f} hours")
        print(f"\nIn Service Range: {result.data['in_service_range']}")
        print(f"Performance Factor: {result.data['performance_factor']:.3f}")
        print(f"Strength Retention: {result.data['strength_retention_percent']:.1f}%")
        print(f"\nThermal Analysis:")
        print(f"  Heat Loss Rate: {result.data['heat_loss_rate_W_m2']:.1f} W/m²")
        print(f"  Convection Coefficient: {result.data['convection_coefficient']:.1f} W/(m²·K)")
        print(f"  Effective Thermal Conductivity: {result.data['effective_thermal_conductivity']*1000:.2f} mW/(m·K)")
        print(f"\nStructural Analysis:")
        print(f"  Thermal Stress: {result.data['thermal_stress_MPa']:.3f} MPa")
        print(f"  Adjusted Tensile Strength: {result.data['adjusted_tensile_strength']:.3f} MPa")
        print(f"  Adjusted Modulus: {result.data['adjusted_modulus']*1000:.2f} MPa")
        print(f"\nMin Service Temp: {result.data['min_service_temp']:.0f} K ({result.data['min_service_temp']-273:.0f}°C)")
        print(f"\nTest Result: {'✓ PASS' if result.success else '✗ FAIL'}")
        print("="*70)

        # Assertions
        self.assertTrue(result.success, "Airloy X103 should PASS extreme cold test")
        self.assertEqual(result.material_name, "Airloy X103")
        self.assertTrue(result.data['in_service_range'], "Should be in service range")
        self.assertGreater(result.data['strength_retention_percent'], 80,
                          "Should retain >80% strength")

        # Check thermal performance
        k_effective = result.data['effective_thermal_conductivity']
        self.assertLess(k_effective, 0.020, "Thermal conductivity should stay low")

        # Check structural integrity
        adjusted_strength = result.data['adjusted_tensile_strength']
        self.assertGreater(adjusted_strength, 0.20, "Should maintain reasonable strength")

    def test_ice_growth_simulation(self):
        """Direct ice growth simulation should return nucleation metrics."""
        metrics = self.lab.simulate_ice_growth(
            "SS 304",
            temperature_k=258.15,
            relative_humidity=0.75,
            duration_hours=2.0,
        )
        self.assertIn("nucleation_rate_m3s", metrics)
        self.assertIn("growth_velocity_m_s", metrics)
        self.assertGreaterEqual(metrics["supercooling_K"], 0)

    def test_airloy_room_temp(self):
        """Test Airloy X103 at room temperature for comparison"""
        result = self.lab.environmental_test(
            "Airloy X103",
            temperature=298.15,  # 25°C
            wind_speed=0,
            duration_hours=1
        )

        self.assertTrue(result.success)
        self.assertAlmostEqual(result.data['performance_factor'], 1.0, places=1,
                              msg="Room temp should have performance factor ~1.0")


class TestCalibrationFramework(unittest.TestCase):
    """Ensure calibration and uncertainty hooks behave."""

    @classmethod
    def setUpClass(cls):
        cls.lab = MaterialsLab()

    def test_calibration_adjusts_tensile(self):
        base = self.lab.tensile_test("SS 304")
        original = base.data["yield_strength"]

        # Register a calibration that nudges yield strength upward by 5 MPa
        self.lab.register_calibration(
            "SS 304",
            "tensile",
            "yield_strength",
            reference_value=original + 5,
            measured_value=original,
        )

        corrected = self.lab.tensile_test("SS 304")
        self.assertGreater(corrected.data["yield_strength"], original)
        self.assertIn("calibration", corrected.data)

    def test_steel_extreme_cold(self):
        """Test steel at extreme cold (should also pass)"""
        result = self.lab.environmental_test(
            "SS 304",
            temperature=73,
            wind_speed=13.4,
            duration_hours=24
        )

        # Stainless steel should handle -200°C
        self.assertTrue(result.success)


class TestMaterialDesign(unittest.TestCase):
    """Test material design tools"""

    @classmethod
    def setUpClass(cls):
        cls.lab = MaterialsLab()

    def test_composite_design(self):
        """Test composite design"""
        result = self.lab.design_composite(
            "Carbon Fiber Epoxy",
            "Epoxy Resin",
            fiber_volume_fraction=0.60,
            layup=[0, 90, 0, 90]
        )

        self.assertIsNotNone(result.optimized_properties)
        # Composite should be lighter than steel but strong
        self.assertLess(result.optimized_properties.density, 3000)
        self.assertGreater(result.optimized_properties.tensile_strength, 500)

    def test_nanoparticle_enhancement(self):
        """Test nanoparticle enhancement"""
        result = self.lab.add_nanoparticles(
            "PEEK",
            nanoparticle_type="CNT",
            loading_percent=2.0
        )

        # Should increase strength
        peek = self.lab.get_material("PEEK")
        enhanced = result.optimized_properties

        self.assertGreater(enhanced.tensile_strength, peek.tensile_strength,
                          "Nanoparticles should enhance strength")

    def test_lattice_structure(self):
        """Test lattice structure design"""
        result = self.lab.design_lattice(
            "Ti-6Al-4V",
            relative_density=0.30,
            cell_type="octet"
        )

        ti = self.lab.get_material("Ti-6Al-4V")
        lattice = result.optimized_properties

        # Should be much lighter
        self.assertLess(lattice.density, ti.density * 0.4)
        # But weaker (scaled by density)
        self.assertLess(lattice.tensile_strength, ti.tensile_strength)


class TestPropertyPrediction(unittest.TestCase):
    """Test property prediction"""

    @classmethod
    def setUpClass(cls):
        cls.lab = MaterialsLab()

    def test_predict_from_composition(self):
        """Test prediction from composition"""
        composition = {"Fe": 98, "C": 2}
        predictions = self.lab.predict_from_composition(
            composition,
            ["density", "tensile_strength"]
        )

        self.assertEqual(len(predictions), 2)
        for pred in predictions:
            self.assertGreater(pred.predicted_value, 0)
            self.assertGreaterEqual(pred.confidence, 0)
            self.assertLessEqual(pred.confidence, 1)

    def test_predict_by_similarity(self):
        """Test prediction by similarity"""
        pred = self.lab.predict_by_similarity("Al 7075-T6", "fracture_toughness")

        self.assertIsNotNone(pred)
        self.assertGreaterEqual(pred.confidence, 0)


class TestMaterialComparison(unittest.TestCase):
    """Test material comparison features"""

    @classmethod
    def setUpClass(cls):
        cls.lab = MaterialsLab()

    def test_compare_materials(self):
        """Test material comparison"""
        comparison = self.lab.compare_materials(
            ["Al 7075-T6", "Ti-6Al-4V", "SS 304"],
            ["density", "tensile_strength", "cost_per_kg"]
        )

        self.assertEqual(len(comparison), 3)
        for mat_name, props in comparison.items():
            self.assertIn("density", props)
            self.assertIn("tensile_strength", props)
            self.assertIn("cost_per_kg", props)

    def test_find_best_material(self):
        """Test finding best material"""
        best = self.lab.find_best_material(
            category="metal",
            optimize_for="tensile_strength",
            constraints={"density": (0, 5000)}
        )

        self.assertIsNotNone(best)
        self.assertEqual(best.category, "metal")
        self.assertLessEqual(best.density, 5000)


class TestAccuracy(unittest.TestCase):
    """Test real-world accuracy (<1% error for well-characterized materials)"""

    @classmethod
    def setUpClass(cls):
        cls.lab = MaterialsLab()

    def _assert_material_accuracy(self, material_name, properties):
        results = self.lab.validate_material_properties(material_name, properties)
        for prop in properties:
            self.assertIn(prop, results, f"{material_name} missing validation result for '{prop}'")
            result = results[prop]
            self.assertEqual(
                result.status,
                ValidationStatus.PASS,
                f"{material_name} {prop} failed validation: {result.message}",
            )

    def test_steel_304_accuracy(self):
        """Test SS 304 properties against known values"""
        self._assert_material_accuracy(
            "SS 304",
            ["density", "youngs_modulus", "yield_strength", "thermal_conductivity"],
        )

    def test_aluminum_6061_accuracy(self):
        """Test Al 6061-T6 properties"""
        self._assert_material_accuracy(
            "Al 6061-T6",
            ["density", "youngs_modulus", "yield_strength", "thermal_conductivity"],
        )

    def test_titanium_6al4v_accuracy(self):
        """Test Ti-6Al-4V properties"""
        self._assert_material_accuracy(
            "Ti-6Al-4V",
            ["density", "youngs_modulus", "yield_strength", "thermal_conductivity"],
        )

    def test_accuracy_suite_report(self):
        """Validate aggregated accuracy suite across mapped materials."""
        report = self.lab.validate_accuracy_suite()
        self.assertGreaterEqual(len(report), 3, "Expected at least three materials in accuracy suite")

        for material, results in report.items():
            for prop, result in results.items():
                self.assertEqual(
                    result.status,
                    ValidationStatus.PASS,
                    f"{material} {prop} failed validation: {result.message}",
                )


if __name__ == "__main__":
    # Run tests
    print("="*70)
    print("MATERIALS LAB - COMPREHENSIVE TEST SUITE")
    print("="*70)

    # Run tests with verbose output
    unittest.main(verbosity=2)
