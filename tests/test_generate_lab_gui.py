import unittest
import sys
import os

# Add the root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generate_lab_gui import generate_lab_gui

class TestGenerateLabGui(unittest.TestCase):

    def test_generate_lab_gui_all_inputs(self):
        """Test that all input types are correctly generated into HTML."""
        config = {
            "title": "Test Lab",
            "tagline": "Test Tagline",
            "color": "#123456",
            "inputs": [
                {"name": "test_select", "label": "Test Select", "type": "select", "options": ["opt1", "opt_2"]},
                {"name": "test_number", "label": "Test Number", "type": "number", "default": 42},
                {"name": "test_text", "label": "Test Text", "type": "text", "placeholder": "Enter text"},
                {"name": "test_checkbox", "label": "Test Checkbox", "type": "checkbox"}
            ],
            "demo": {"test_select": "opt1", "test_number": 42, "test_text": "hello", "test_checkbox": True}
        }

        html = generate_lab_gui("test_lab", config)

        # Check basic structure
        self.assertIn("Test Lab - QuLab", html)
        self.assertIn("🔬 Test Lab", html)
        self.assertIn("Test Tagline", html)
        self.assertIn("#123456", html)

        # Check select input
        self.assertIn('<select id="test_select"', html)
        self.assertIn('<option value="opt1">Opt1</option>', html)
        self.assertIn('<option value="opt_2">Opt 2</option>', html)

        # Check number input
        self.assertIn('<input type="number" id="test_number"', html)
        self.assertIn('value="42"', html)

        # Check text input
        self.assertIn('<input type="text" id="test_text"', html)
        self.assertIn('placeholder="Enter text"', html)

        # Check checkbox input
        self.assertIn('<input type="checkbox" id="test_checkbox"', html)

        # Check JS generation
        self.assertIn('"test_select", "test_number", "test_text", "test_checkbox"', html)
        self.assertIn('const demoData = {"test_select": "opt1", "test_number": 42, "test_text": "hello", "test_checkbox": true};', html)

    def test_generate_lab_gui_no_demo(self):
        """Test generation works when no demo data is provided."""
        config = {
            "title": "Minimal Lab",
            "tagline": "No demo",
            "color": "#000000",
            "inputs": [
                {"name": "simple", "label": "Simple", "type": "text"}
            ]
            # Missing 'demo' key
        }

        html = generate_lab_gui("minimal_lab", config)
        self.assertIn("Minimal Lab", html)
        self.assertIn("const demoData = {};", html)

if __name__ == '__main__':
    unittest.main()
