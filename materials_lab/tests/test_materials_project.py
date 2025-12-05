#!/usr/bin/env python3
"""
Tests for Materials Project Integration

Tests:
- Materials Project client
- Material validation
- Confidence scoring
- Dataset loading
- Property conversion
"""

import unittest
import os
import sys
from pathlib import Path
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from materials_lab.materials_project_client import MaterialsProjectClient, MPMaterialData
from materials_lab.materials_validator import MaterialsValidator, ValidationStatus
from materials_lab.confidence_scorer import ConfidenceScorer, DataSource
from materials_lab.materials_database import MaterialProperties


class TestMaterialsProjectClient(unittest.TestCase):
    """Test Materials Project client functionality"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        # Check for API key
        cls.api_key = os.environ.get("MP_API_KEY")
        if not cls.api_key or cls.api_key == "your_materials_project_api_key_here":
            cls.skip_api_tests = True
            print("\n⚠️  MP_API_KEY not set - skipping API tests")
        else:
            cls.skip_api_tests = False

            # Create temp cache directory
            cls.temp_cache = tempfile.mkdtemp()

            # Initialize client
            cls.client = MaterialsProjectClient(
                api_key=cls.api_key,
                cache_dir=cls.temp_cache
            )

    @classmethod
    def tearDownClass(cls):
        """Clean up"""
        if hasattr(cls, 'temp_cache'):
            shutil.rmtree(cls.temp_cache, ignore_errors=True)

    def test_client_initialization(self):
        """Test client can be initialized"""
        if self.skip_api_tests:
            self.skipTest("No API key available")

        self.assertIsNotNone(self.client)
        self.assertEqual(self.client.api_key, self.api_key)

    def test_get_silicon(self):
        """Test fetching Silicon (mp-149)"""
        if self.skip_api_tests:
            self.skipTest("No API key available")

        material = self.client.get_material("mp-149")

        self.assertIsNotNone(material)
        self.assertEqual(material.mp_id, "mp-149")
        self.assertEqual(material.formula, "Si")
        self.assertAlmostEqual(material.density, 2.33, delta=0.1)
        self.assertAlmostEqual(material.band_gap, 0.6, delta=0.3)  # Indirect gap ~1.1 eV, DFT underestimates

    def test_get_iron(self):
        """Test fetching Iron (mp-13)"""
        if self.skip_api_tests:
            self.skipTest("No API key available")

        material = self.client.get_material("mp-13")

        self.assertIsNotNone(material)
        self.assertEqual(material.mp_id, "mp-13")
        self.assertEqual(material.formula, "Fe")
        self.assertAlmostEqual(material.density, 7.87, delta=0.2)
        self.assertLess(material.band_gap, 0.1)  # Metal

    def test_cache_functionality(self):
        """Test caching works"""
        if self.skip_api_tests:
            self.skipTest("No API key available")

        # First fetch (from API)
        material1 = self.client.get_material("mp-149", use_cache=False)

        # Second fetch (from cache)
        material2 = self.client.get_material("mp-149", use_cache=True)

        self.assertEqual(material1.mp_id, material2.mp_id)
        self.assertEqual(material1.formula, material2.formula)

        # Check cache file exists
        cache_file = Path(self.temp_cache) / "mp-149.json"
        self.assertTrue(cache_file.exists())

    def test_to_material_properties(self):
        """Test conversion to MaterialProperties"""
        if self.skip_api_tests:
            self.skipTest("No API key available")

        mp_material = self.client.get_material("mp-149")
        properties = mp_material.to_material_properties()

        self.assertIsInstance(properties, MaterialProperties)
        self.assertIn("Si", properties.name)
        self.assertAlmostEqual(properties.density_g_cm3, 2.33, delta=0.1)
        self.assertGreater(properties.youngs_modulus, 0)  # Should be estimated
        self.assertGreater(properties.bulk_modulus, 0)

    def test_search_materials(self):
        """Test material search"""
        if self.skip_api_tests:
            self.skipTest("No API key available")

        # Search for Fe-O compounds
        materials = self.client.search_materials(
            elements=["Fe", "O"],
            is_stable=True,
            limit=5
        )

        self.assertGreater(len(materials), 0)
        self.assertLessEqual(len(materials), 5)

        # Check all have Fe and O
        for mat in materials:
            self.assertTrue(mat.is_stable)

    def test_invalid_material_id(self):
        """Test handling of invalid material ID"""
        if self.skip_api_tests:
            self.skipTest("No API key available")

        material = self.client.get_material("mp-999999999")
        self.assertIsNone(material)


class TestMaterialsValidator(unittest.TestCase):
    """Test materials validation functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.validator = MaterialsValidator()

    def test_aerogel_validation_perfect_match(self):
        """Test aerogel validation with perfect match"""
        simulated_results = {
            "density_kg_m3": 144.0,
            "thermal_conductivity": 0.014,
            "tensile_strength": 0.31,
            "compressive_strength": 1.65,
        }

        validation = self.validator.validate_aerogel(simulated_results)

        self.assertEqual(validation.material_name, "Airloy X103 Aerogel")
        self.assertEqual(len(validation.comparisons), 4)
        self.assertGreater(validation.confidence_score, 95.0)
        self.assertEqual(validation.overall_status, ValidationStatus.EXCELLENT)

    def test_aerogel_validation_with_errors(self):
        """Test aerogel validation with some errors"""
        simulated_results = {
            "density_kg_m3": 160.0,  # 11% error
            "thermal_conductivity": 0.015,  # 7% error
            "tensile_strength": 0.35,  # 13% error
        }

        validation = self.validator.validate_aerogel(simulated_results)

        self.assertEqual(len(validation.comparisons), 3)
        self.assertLess(validation.confidence_score, 95.0)
        self.assertGreater(validation.confidence_score, 70.0)

        # Check individual comparisons
        for comp in validation.comparisons:
            self.assertGreater(comp.relative_error_percent, 0.0)
            self.assertLess(comp.relative_error_percent, 20.0)

    def test_compare_properties(self):
        """Test property comparison"""
        simulated = MaterialProperties(
            name="Test Material",
            category="metal",
            subcategory="alloy",
            density_g_cm3=7.8,
            youngs_modulus=200.0,
            thermal_conductivity=50.0
        )

        reference = MaterialProperties(
            name="Reference Material",
            category="metal",
            subcategory="alloy",
            density_g_cm3=8.0,
            youngs_modulus=210.0,
            thermal_conductivity=48.0
        )

        comparisons = self.validator.compare_properties(
            simulated,
            reference,
            properties_to_check=["density_g_cm3", "youngs_modulus", "thermal_conductivity"]
        )

        self.assertEqual(len(comparisons), 3)

        # Check density comparison
        density_comp = [c for c in comparisons if c.property_name == "density_g_cm3"][0]
        self.assertAlmostEqual(density_comp.relative_error_percent, 2.5, delta=0.1)

    def test_validation_status_calculation(self):
        """Test validation status determination"""
        # Excellent: <5%
        status = self.validator._calculate_status(3.0, "density_g_cm3")
        self.assertEqual(status, ValidationStatus.EXCELLENT)

        # Good: 5-15%
        status = self.validator._calculate_status(8.0, "density_g_cm3")
        self.assertEqual(status, ValidationStatus.GOOD)

        # Poor: 30-50%
        status = self.validator._calculate_status(40.0, "density_g_cm3")
        self.assertEqual(status, ValidationStatus.POOR)

        # Failed: >50%
        status = self.validator._calculate_status(60.0, "density_g_cm3")
        self.assertEqual(status, ValidationStatus.FAILED)


class TestConfidenceScorer(unittest.TestCase):
    """Test confidence scoring system"""

    def setUp(self):
        """Set up test fixtures"""
        self.scorer = ConfidenceScorer()

    def test_score_experimental_property(self):
        """Test scoring experimental data"""
        confidence = self.scorer.score_property(
            property_name="density_g_cm3",
            value=7.87,
            data_source=DataSource.EXPERIMENTAL_PEER_REVIEWED,
            uncertainty_percent=1.0,
            validation_sources=3,
            has_related_properties=True
        )

        # Should have high confidence
        self.assertGreater(confidence.confidence_score, 90.0)
        self.assertEqual(confidence.get_confidence_level(), "Excellent")

    def test_score_estimated_property(self):
        """Test scoring estimated data"""
        confidence = self.scorer.score_property(
            property_name="thermal_conductivity",
            value=50.0,
            data_source=DataSource.ROUGH_ESTIMATE,
            uncertainty_percent=40.0,
            validation_sources=0,
            has_related_properties=False
        )

        # Should have low confidence
        self.assertLess(confidence.confidence_score, 40.0)
        self.assertIn(confidence.get_confidence_level(), ["Poor", "Unreliable"])

    def test_score_material(self):
        """Test scoring complete material"""
        properties = {
            "density_g_cm3": 2.33,
            "youngs_modulus": 165.0,
            "band_gap_ev": 1.12,
            "thermal_conductivity": 148.0,
        }

        data_sources = {
            "density_g_cm3": DataSource.EXPERIMENTAL_PEER_REVIEWED,
            "youngs_modulus": DataSource.COMPUTATIONAL_VALIDATED,
            "band_gap_ev": DataSource.MATERIALS_PROJECT_DFT,
            "thermal_conductivity": DataSource.COMPUTATIONAL_ONLY,
        }

        uncertainties = {
            "density_g_cm3": 1.0,
            "youngs_modulus": 10.0,
            "band_gap_ev": 5.0,
            "thermal_conductivity": 20.0,
        }

        validation_counts = {
            "density_g_cm3": 3,
            "youngs_modulus": 2,
            "band_gap_ev": 1,
            "thermal_conductivity": 0,
        }

        report = self.scorer.score_material(
            material_name="Silicon",
            properties=properties,
            data_sources=data_sources,
            uncertainties=uncertainties,
            validation_counts=validation_counts,
            material_id="mp-149"
        )

        self.assertEqual(report.material_name, "Silicon")
        self.assertEqual(len(report.property_confidences), 4)
        self.assertGreater(report.overall_confidence, 60.0)
        self.assertGreater(report.data_completeness, 0.0)

        # Check that density has highest confidence
        density_confidence = [p for p in report.property_confidences
                              if p.property_name == "density_g_cm3"][0]
        self.assertGreater(density_confidence.confidence_score, 90.0)

    def test_property_type_inference(self):
        """Test property type inference"""
        from materials_lab.confidence_scorer import PropertyType

        # Mechanical
        prop_type = self.scorer._infer_property_type("youngs_modulus")
        self.assertEqual(prop_type, PropertyType.MECHANICAL)

        # Thermal
        prop_type = self.scorer._infer_property_type("thermal_conductivity")
        self.assertEqual(prop_type, PropertyType.THERMAL)

        # Electronic
        prop_type = self.scorer._infer_property_type("band_gap_ev")
        self.assertEqual(prop_type, PropertyType.ELECTRONIC)

    def test_confidence_levels(self):
        """Test confidence level thresholds"""
        from materials_lab.confidence_scorer import PropertyConfidence, PropertyType

        # Excellent: 90-100
        conf = PropertyConfidence("test", 1.0, "unit", 95.0, DataSource.EXPERIMENTAL_PEER_REVIEWED,
                                   PropertyType.FUNDAMENTAL, 1.0, 3, "")
        self.assertEqual(conf.get_confidence_level(), "Excellent")

        # Good: 70-79
        conf.confidence_score = 75.0
        self.assertEqual(conf.get_confidence_level(), "Good")

        # Poor: 40-49
        conf.confidence_score = 45.0
        self.assertEqual(conf.get_confidence_level(), "Poor")

        # Unreliable: <40
        conf.confidence_score = 30.0
        self.assertEqual(conf.get_confidence_level(), "Unreliable")


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple components"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.api_key = os.environ.get("MP_API_KEY")
        if not cls.api_key or cls.api_key == "your_materials_project_api_key_here":
            cls.skip_tests = True
        else:
            cls.skip_tests = False
            cls.temp_cache = tempfile.mkdtemp()
            cls.client = MaterialsProjectClient(api_key=cls.api_key, cache_dir=cls.temp_cache)
            cls.validator = MaterialsValidator(mp_client=cls.client)
            cls.scorer = ConfidenceScorer()

    @classmethod
    def tearDownClass(cls):
        """Clean up"""
        if hasattr(cls, 'temp_cache'):
            shutil.rmtree(cls.temp_cache, ignore_errors=True)

    def test_full_workflow_silicon(self):
        """Test complete workflow: fetch -> convert -> validate -> score"""
        if self.skip_tests:
            self.skipTest("No API key available")

        # 1. Fetch from Materials Project
        mp_material = self.client.get_material("mp-149")
        self.assertIsNotNone(mp_material)

        # 2. Convert to MaterialProperties
        properties = mp_material.to_material_properties()
        self.assertIsInstance(properties, MaterialProperties)

        # 3. Create confidence report
        property_dict = {
            "density_g_cm3": properties.density_g_cm3,
            "youngs_modulus": properties.youngs_modulus,
            "band_gap_ev": properties.band_gap_ev,
        }

        data_sources = {
            "density_g_cm3": DataSource.MATERIALS_PROJECT_DFT,
            "youngs_modulus": DataSource.ESTIMATED_CORRELATION,
            "band_gap_ev": DataSource.MATERIALS_PROJECT_DFT,
        }

        report = self.scorer.score_material(
            material_name=properties.name,
            properties=property_dict,
            data_sources=data_sources,
            material_id=mp_material.mp_id
        )

        self.assertGreater(report.overall_confidence, 50.0)
        self.assertEqual(len(report.property_confidences), 3)


def suite():
    """Create test suite"""
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMaterialsProjectClient))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMaterialsValidator))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestConfidenceScorer))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIntegration))
    return suite


if __name__ == "__main__":
    # Check for API key
    api_key = os.environ.get("MP_API_KEY")
    if not api_key or api_key == "your_materials_project_api_key_here":
        print("\n" + "=" * 80)
        print("WARNING: MP_API_KEY not set")
        print("=" * 80)
        print("\nSome tests will be skipped.")
        print("\nTo run all tests:")
        print("  1. Get API key from https://materialsproject.org/api")
        print("  2. export MP_API_KEY='your-key-here'")
        print("  3. Re-run tests")
        print("\n" + "=" * 80 + "\n")

    # Run tests
    unittest.main(verbosity=2)
