import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# Ensure the parent directory is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Bypass demo.py's problematic imports completely using patch.dict
import importlib.util

spec = importlib.util.spec_from_file_location("demo", "quantum_lab/demo.py")
demo_module = importlib.util.module_from_spec(spec)
sys.modules["quantum_lab.demo"] = demo_module

# Before executing the module, set up the mocked dependencies in sys.modules
with patch.dict('sys.modules', {
    'quantum_lab': MagicMock(),
    'quantum_chemistry': MagicMock(),
    'quantum_validation': MagicMock(),
}):
    spec.loader.exec_module(demo_module)

print_section = demo_module.print_section

def test_print_section(capsys):
    """Test that print_section formats the output correctly."""
    title = "TEST TITLE"
    print_section(title)

    captured = capsys.readouterr()

    expected_output = "\n" + "="*70 + "\n" + f"  {title}\n" + "="*70 + "\n\n"

    assert captured.out == expected_output
    assert captured.err == ""

def test_print_section_empty_title(capsys):
    """Test print_section with an empty string."""
    title = ""
    print_section(title)

    captured = capsys.readouterr()

    expected_output = "\n" + "="*70 + "\n" + f"  {title}\n" + "="*70 + "\n\n"

    assert captured.out == expected_output
    assert captured.err == ""

def test_print_section_long_title(capsys):
    """Test print_section with a long string."""
    title = "A" * 100
    print_section(title)

    captured = capsys.readouterr()

    expected_output = "\n" + "="*70 + "\n" + f"  {title}\n" + "="*70 + "\n\n"

    assert captured.out == expected_output
    assert captured.err == ""
