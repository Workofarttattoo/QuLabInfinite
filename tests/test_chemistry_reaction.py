"""
Unit test for chemistry reaction experiment runner.
"""

import sys
from pathlib import Path

import numpy as np

# Ensure project root import resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chemistry_lab.chemistry_lab import ChemistryLaboratory  # noqa: E402


def test_reaction_experiment_happy_path():
    """Reaction experiments should build molecules, run simulation, and return kinetics + profiles."""
    lab = ChemistryLaboratory()

    experiment = {
        "experiment_type": "reaction_simulation",
        "reactants": [
            {"formula": "A", "smiles": "A", "energy": 0.0, "enthalpy": 0.0, "entropy": 50.0},
            {"formula": "B", "smiles": "B", "energy": 0.0, "enthalpy": 0.0, "entropy": 60.0},
        ],
        "products": [
            {"formula": "AB", "smiles": "AB", "energy": -10.0, "enthalpy": -10.0, "entropy": 80.0},
        ],
        "conditions": {
            "temperature": 298.15,
            "pressure": 1.0,
            "solvent": "water",
        },
        "reaction_name": "test_reaction",
    }

    result = lab.run_experiment(experiment)

    assert result["status"] == "completed"
    assert "kinetics" in result and result["kinetics"]
    assert result["kinetics"]["rate_constant"] > 0
    assert result["profiles"]["time"][0] > 0

    reactant_curve = np.array(result["profiles"]["reactant_concentration"])
    product_curve = np.array(result["profiles"]["product_concentration"])

    assert reactant_curve[0] > reactant_curve[-1]
    assert product_curve[-1] > product_curve[0]
    assert "total" in result["profiles"]["product_distribution"]
