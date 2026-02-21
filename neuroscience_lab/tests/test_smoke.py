import importlib.util
from pathlib import Path

LAB_PACKAGE = "neuroscience_lab"


def test_neuroscience_lab_package_stub_exists():
    spec = importlib.util.find_spec(LAB_PACKAGE)
    assert spec is not None, f"Package neuroscience_lab not discoverable"


def test_neuroscience_lab_readme_present():
    lab_root = Path(__file__).resolve().parents[1]
    readme = lab_root / "README.md"
    assert readme.exists(), "README.md missing"
