import unittest
import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ech0_market_domination_displays import create_market_domination_inventions
from ech0_invention_poc_pipeline import InventionConcept

class TestEch0MarketDominationDisplays(unittest.TestCase):
    def setUp(self):
        self.inventions = create_market_domination_inventions()

    def test_returns_correct_number_of_inventions(self):
        """Test that exactly 25 inventions are created as documented."""
        self.assertEqual(len(self.inventions), 25, "Should create exactly 25 inventions")

    def test_all_items_are_invention_concepts(self):
        """Test that all returned items are instances of InventionConcept."""
        for i, invention in enumerate(self.inventions):
            self.assertIsInstance(
                invention,
                InventionConcept,
                f"Item at index {i} is not an InventionConcept"
            )

    def test_inventions_have_required_fields(self):
        """Test that each invention has a valid name and description."""
        for i, invention in enumerate(self.inventions):
            self.assertTrue(hasattr(invention, 'name'), f"Invention {i} missing name attribute")
            self.assertTrue(hasattr(invention, 'description'), f"Invention {i} missing description attribute")

            self.assertIsInstance(invention.name, str)
            self.assertIsInstance(invention.description, str)

            self.assertTrue(len(invention.name.strip()) > 0, f"Invention {i} has empty name")
            self.assertTrue(len(invention.description.strip()) > 0, f"Invention {i} has empty description")

    def test_specific_inventions_exist(self):
        """Test that a few specific expected inventions exist in the list."""
        names = [inv.name for inv in self.inventions]

        # Check for one from each major category
        expected_names = [
            "Retro-Reflective Pepper's Ghost Hologram", # Daylight Holograms
            "Electrochromic Pigment Display",           # New Display Tech
            "DIY Transparent LCD Panel"                 # Affordable Screen Innovations
        ]

        for name in expected_names:
            self.assertIn(name, names, f"Expected invention '{name}' not found")

if __name__ == '__main__':
    unittest.main()
