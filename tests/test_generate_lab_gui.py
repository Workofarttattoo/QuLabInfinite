import unittest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Adjust sys.path to ensure we can import the module from the parent directory
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import generate_lab_gui

class TestGenerateLabGui(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    @patch('generate_lab_gui.Path')
    def test_generate_all_guis(self, mock_path_class):
        # Configure the mock to return our temp directory when parent is accessed
        mock_path_instance = MagicMock()
        mock_path_class.return_value = mock_path_instance
        mock_path_instance.parent = self.test_path

        # Execute the function
        generate_lab_gui.generate_all_guis()

        # Verify lab_guis directory was created
        output_dir = self.test_path / "lab_guis"
        self.assertTrue(output_dir.exists())
        self.assertTrue(output_dir.is_dir())

        # Verify index.html was created
        index_file = output_dir / "index.html"
        self.assertTrue(index_file.exists())

        # Read index.html content
        with open(index_file, 'r', encoding='utf-8') as f:
            index_content = f.read()

        self.assertIn("QuLab GUI Index", index_content)

        # Verify each lab GUI file was created and is linked in index
        for lab_id, config in generate_lab_gui.LAB_CONFIGS.items():
            lab_file = output_dir / f"{lab_id}.html"
            self.assertTrue(lab_file.exists(), f"Missing file for lab: {lab_id}")

            # Check content of the lab file
            with open(lab_file, 'r', encoding='utf-8') as f:
                content = f.read()

            self.assertIn(config["title"], content)
            self.assertIn(config["tagline"], content)

            # Verify it's linked in the index
            self.assertIn(f"{lab_id}.html", index_content)

if __name__ == '__main__':
    unittest.main()
