import pytest
from pathlib import Path
from unittest.mock import patch
import sys
import os

# Add the parent directory to sys.path to import generate_lab_gui
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import generate_lab_gui

def test_generate_all_guis(tmp_path, monkeypatch):
    """
    Test that generate_all_guis correctly creates HTML files for each lab
    config and generates an index.html file.
    """
    # Mock the __file__ attribute of the module to point to our tmp_path
    # so that the output_dir becomes tmp_path / "lab_guis"
    monkeypatch.setattr(generate_lab_gui, "__file__", str(tmp_path / "generate_lab_gui.py"))

    # Run the function
    generate_lab_gui.generate_all_guis()

    # The expected output directory
    expected_out_dir = tmp_path / "lab_guis"

    # Assert directory was created
    assert expected_out_dir.exists()
    assert expected_out_dir.is_dir()

    # Check that individual lab HTML files were created
    for lab_id in generate_lab_gui.LAB_CONFIGS.keys():
        expected_file = expected_out_dir / f"{lab_id}.html"
        assert expected_file.exists()
        assert expected_file.is_file()

        # Optionally, check that the file has some content
        content = expected_file.read_text()
        assert "<!DOCTYPE html>" in content
        assert lab_id.replace("_", " ").title() in content or lab_id in content or generate_lab_gui.LAB_CONFIGS[lab_id]["title"] in content

    # Check that index.html was created
    index_file = expected_out_dir / "index.html"
    assert index_file.exists()
    assert index_file.is_file()

    # Check index content
    index_content = index_file.read_text()
    assert "<!DOCTYPE html>" in index_content
    assert "QuLab GUI Index" in index_content

    # Verify the total number of files created (labs + index.html)
    created_files = list(expected_out_dir.glob("*.html"))
    assert len(created_files) == len(generate_lab_gui.LAB_CONFIGS) + 1
