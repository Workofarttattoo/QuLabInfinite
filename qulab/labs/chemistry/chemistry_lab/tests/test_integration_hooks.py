"""
Unit tests for chemistry_lab.integration quantitative hooks.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from chemistry_lab.integration import apply_environmental_adjustments, apply_material_updates  # noqa: E402
from materials_lab.materials_lab import MaterialsLab  # noqa: E402
from environmental_sim.environmental_sim import EnvironmentalSimulator  # noqa: E402


class TestIntegrationHooks(unittest.TestCase):
    """Validate that integration hooks perform quantitative updates."""

    def setUp(self):
        self.materials_lab = MaterialsLab()
        self.environment = EnvironmentalSimulator()

    def test_material_updates_adjust_properties(self):
        """Corrosion hazards should trigger quantitative property adjustments."""
        material = self.materials_lab.get_material("SS 304")
        self.assertIsNotNone(material)
        original_tensile = material.tensile_strength

        payload = {
            "reaction_name": "propene_hydroformylation",
            "links": {"product_ids": ["SS 304"]},
            "kinetics": {
                "rate_constant": 1.2e-6,
                "activation_energy": 21.5,
            },
            "thermodynamics": {
                "delta_h_kcal_per_mol": -36.0,
                "delta_g_kcal_per_mol": -23.0,
            },
            "hazards": ["Corrosive", "Oxidizer"],
            "effects": {
                "SS 304": {
                    "tensile_strength": {"multiplier": 0.97},
                    "notes": "Exposure to syngas condensate",
                }
            },
        }

        apply_material_updates(self.materials_lab, payload)

        updated = self.materials_lab.get_material("SS 304")
        self.assertIsNotNone(updated)
        self.assertTrue(hasattr(updated, "chemistry_corrosion_rate_mm_per_year"))
        self.assertGreater(updated.chemistry_corrosion_rate_mm_per_year, 0.0)
        self.assertEqual(updated.chemistry_last_reaction, "propene_hydroformylation")
        self.assertIsNotNone(updated.notes)
        self.assertIn("Reaction: propene_hydroformylation", updated.notes)
        self.assertLess(updated.tensile_strength, original_tensile)

    def test_environment_adjustments_update_controller(self):
        """Environmental payload should create contaminant decay and corrosion records."""
        payload = [
            {
                "material_id": "Hydrogen Iodide Vapor",
                "name": "hydrogen_iodide",
                "phase": "gas",
                "estimated_release_rate": 0.02,
                "hazards": ["corrosive"],
                "disposal": "scrubber",
                "corrosion_rate_multiplier": 1.6,
                "target_material": "SS 304",
                "baseline_corrosion_rate_mm_per_year": 0.25,
                "exposure_hours": 2.5,
                "decay_half_life_hours": 1.5,
                "removal_efficiency": 0.5,
            }
        ]

        apply_environmental_adjustments(self.environment, payload)

        controller = self.environment.controller
        state = controller.get_corrosion_state("SS 304")
        self.assertTrue(state)
        self.assertGreater(state.get("adjusted_rate_mm_per_year", 0.0), 0.0)
        self.assertIn("sources", state)
        contaminant_ppm = controller.atmosphere.get_contaminant("Hydrogen Iodide Vapor")
        self.assertGreater(contaminant_ppm, 0.0)
        self.assertTrue(hasattr(controller, "chemistry_emission_profiles"))
        self.assertTrue(controller.chemistry_emission_profiles)  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()
