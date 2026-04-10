import unittest
from unittest.mock import patch, mock_open, MagicMock
import json
import sys
import os

# Add parent directory to path so we can import the script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# We need to mock the external dependencies before importing if they don't exist
try:
    import demo_ech0_poc_analysis
except ImportError:
    pass # Handled in tests if needed

class TestDemoEch0PocAnalysis(unittest.TestCase):

    def setUp(self):
        # Sample data for tests
        self.sample_materials_list = {
            "chemicals": [{"name": "Chemical A", "cost": "$10", "category": "precursor"}],
            "equipment": [{"name": "Equipment B", "cost": "$100"}],
            "elements": [{"symbol": "Si", "properties": {"mass": 28.085}}],
            "validation_status": {},
            "total_estimated_cost": 110.0,
            "experiments": [
                {
                    "name": "Test Experiment",
                    "objective": "Test objective",
                    "method": "Test method",
                    "materials_needed": ["Material X"],
                    "expected_result": "Success",
                    "duration": "1 hour"
                }
            ]
        }

        self.sample_analysis_results = [
            {
                "invention": {
                    "id": "INV-001",
                    "name": "Test Invention",
                    "description": "Test description",
                    "materials": {}
                },
                "ech0_analysis": "Test analysis JSON",
                "timestamp": "2023-10-27T10:00:00.000000"
            }
        ]

    @patch('demo_ech0_poc_analysis.open', new_callable=mock_open)
    @patch('demo_ech0_poc_analysis.json.dump')
    def test_create_lab_demo_package_structure(self, mock_json_dump, mock_file_open):
        """Test that the returned dictionary has the expected structure and contents."""

        from demo_ech0_poc_analysis import create_lab_demo_package

        # Call the function
        result = create_lab_demo_package(self.sample_materials_list, self.sample_analysis_results)

        # Assertions on the result structure
        self.assertEqual(result["title"], "ECH0 + QuLab: POC Materials & Test Plan")
        self.assertIn("created", result)
        self.assertEqual(result["created_by"], "ECH0 14B + QuLabInfinite")
        self.assertEqual(result["inventions_analyzed"], 1)
        self.assertEqual(result["materials_list"], self.sample_materials_list)
        self.assertEqual(result["ech0_analyses"], self.sample_analysis_results)

        # Verify demo_instructions are present
        self.assertIn("demo_instructions", result)
        self.assertIn("preparation", result["demo_instructions"])
        self.assertIn("demo_flow", result["demo_instructions"])
        self.assertIn("key_talking_points", result["demo_instructions"])

    @patch('demo_ech0_poc_analysis.open', new_callable=mock_open)
    @patch('demo_ech0_poc_analysis.json.dump')
    def test_create_lab_demo_package_file_writes(self, mock_json_dump, mock_file_open):
        """Test that the function attempts to write the JSON and MD files correctly."""

        from demo_ech0_poc_analysis import create_lab_demo_package

        # Call the function
        create_lab_demo_package(self.sample_materials_list, self.sample_analysis_results)

        # Verify that builtins.open was called twice (once for JSON, once for MD)
        self.assertEqual(mock_file_open.call_count, 2)

        # Check the paths used
        expected_json_path = "/Users/noone/QuLabInfinite/data/ech0_poc_demo_package.json"
        expected_md_path = "/Users/noone/QuLabInfinite/data/POC_MATERIALS_CHECKLIST.md"

        calls = mock_file_open.call_args_list
        self.assertEqual(calls[0][0][0], expected_json_path)
        self.assertEqual(calls[0][0][1], 'w')
        self.assertEqual(calls[1][0][0], expected_md_path)
        self.assertEqual(calls[1][0][1], 'w')

        # Verify json.dump was called
        mock_json_dump.assert_called_once()

        # Verify markdown content writing
        handle = mock_file_open()
        write_calls = handle.write.call_args_list

        # Check if some expected text is in the write calls
        written_text = "".join([call[0][0] for call in write_calls])
        self.assertIn("# POC Materials & Experiment Checklist", written_text)
        self.assertIn("Chemical A", written_text)
        self.assertIn("Equipment B", written_text)
        self.assertIn("Test Experiment", written_text)
        self.assertIn("$110.0", written_text)

    @patch('demo_ech0_poc_analysis.open', new_callable=mock_open)
    @patch('demo_ech0_poc_analysis.json.dump')
    def test_create_lab_demo_package_empty_inputs(self, mock_json_dump, mock_file_open):
        """Test the function behavior with empty input lists."""

        from demo_ech0_poc_analysis import create_lab_demo_package

        empty_materials = {
            "chemicals": [],
            "equipment": [],
            "elements": [],
            "validation_status": {},
            "total_estimated_cost": 0,
            "experiments": []
        }
        empty_analysis = []

        result = create_lab_demo_package(empty_materials, empty_analysis)

        self.assertEqual(result["inventions_analyzed"], 0)
        self.assertEqual(result["materials_list"], empty_materials)
        self.assertEqual(result["ech0_analyses"], empty_analysis)

        # Verify it still tries to write files
        self.assertEqual(mock_file_open.call_count, 2)

if __name__ == '__main__':
    unittest.main()
