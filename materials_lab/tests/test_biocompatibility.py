#!/usr/bin/env python3
"""
Test Biocompatibility Predictor
"""

import unittest
import sys
import os
import json
from unittest.mock import patch, mock_open

# Ensure we can import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fast_biocompatibility_predictor import FastBiocompatibilityPredictor, BiomaterialProperties, ContactType, ContactDuration, BiocompatibilityGrade

class TestFastBiocompatibilityPredictor(unittest.TestCase):

    def setUp(self):
        """Set up test environment"""
        self.predictor = FastBiocompatibilityPredictor()

    def test_database_loading(self):
        """Test that the database is loaded correctly"""
        # Check for key materials
        self.assertIn("Ti-6Al-4V", self.predictor.materials)
        self.assertIn("SS 316L", self.predictor.materials)
        self.assertIn("PEEK", self.predictor.materials)

        # Check properties of a specific material
        ti = self.predictor.materials["Ti-6Al-4V"]
        self.assertIsInstance(ti, BiomaterialProperties)
        self.assertEqual(ti.name, "Ti-6Al-4V (Titanium alloy)")
        self.assertEqual(ti.polymer_type, "metal")
        self.assertFalse(ti.degradable)
        self.assertEqual(ti.surface_energy, 45.0)

    def test_prediction_logic(self):
        """Test prediction logic for a known material"""
        prediction = self.predictor.predict_biocompatibility("Ti-6Al-4V")

        self.assertEqual(prediction.material, "Ti-6Al-4V")
        self.assertEqual(prediction.overall_grade, BiocompatibilityGrade.EXCELLENT)
        self.assertEqual(prediction.cytotoxicity_risk, "Low")
        self.assertEqual(prediction.immune_response_risk, "Low")

    def test_unknown_material(self):
        """Test handling of unknown material"""
        prediction = self.predictor.predict_biocompatibility("Unobtainium")
        self.assertEqual(prediction.material, "Unobtainium")
        self.assertEqual(prediction.overall_grade, BiocompatibilityGrade.MARGINAL)
        self.assertEqual(prediction.confidence, 0.0)

    def test_fallback_mechanism(self):
        """Test that the system falls back to a minimal set if the file is missing"""

        # We need to simulate the file being missing during initialization
        # Since FastBiocompatibilityPredictor loads in __init__, we need to patch BEFORE instantiation

        # We'll use a subclass or mock to override the behavior, but refactoring isn't done yet.
        # So for now, we will assume the refactoring will use open() to read the file.

        # Note: This test will fail UNTIL the refactoring is implemented,
        # because the current implementation hardcodes the data.
        # But once refactored, we want to ensure it handles FileNotFoundError.

        with patch('builtins.open', side_effect=FileNotFoundError):
             # This should NOT raise an exception, but print a warning and load fallback
            predictor = FastBiocompatibilityPredictor()

            # Should have at least one fallback material (e.g. Ti-6Al-4V)
            self.assertIn("Ti-6Al-4V", predictor.materials)
            # Should not be empty
            self.assertGreater(len(predictor.materials), 0)

if __name__ == "__main__":
    unittest.main()
