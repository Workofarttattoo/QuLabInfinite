import importlib.util
from pathlib import Path

LAB_PACKAGE = "pharmacokinetics_lab"


def test_pharmacokinetics_lab_package_stub_exists():
    spec = importlib.util.find_spec(LAB_PACKAGE)
    assert spec is not None, f"Package pharmacokinetics_lab not discoverable"


def test_pharmacokinetics_lab_readme_present():
    lab_root = Path(__file__).resolve().parents[1]
    readme = lab_root / "README.md"
    assert readme.exists(), "README.md missing"
