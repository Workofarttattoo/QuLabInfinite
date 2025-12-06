"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

reference_data - Part of Chemistry Lab
Reference experimental data for calibration routines.

These small datasets provide authoritative values that are used to benchmark
the deterministic chemistry models (spectroscopy predictor and synthesis
planner). The numbers come from widely documented textbook values for the
selected molecules; they are intentionally light-weight so that the test suite
can compute calibration metrics without pulling external resources.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# Spectroscopy reference data -------------------------------------------------

SPECTROSCOPY_REFERENCE: Dict[str, Dict] = {
    "ethanol": {
        "molecule": {
            "name": "ethanol",
            "smiles": "CCO",
            "molecular_weight": 46.07,
            "functional_groups": ["alkane_CH3", "alkane_CH2", "alcohol"],
        },
        # 1H NMR chemical shifts (ppm)
        "nmr_1h": [0.90, 1.25, 3.65],
        # Representative IR absorption peaks (cm^-1)
        "ir": [1050.0, 2980.0, 3350.0],
        # Dominant mass-spec fragments (m/z, relative intensity)
        "mass_spec": [(46.0, 1.0), (31.0, 0.7), (45.0, 0.45)],
    },
    "acetone": {
        "molecule": {
            "name": "acetone",
            "smiles": "CC(=O)C",
            "molecular_weight": 58.08,
            "functional_groups": ["ketone", "alkane_CH3"],
        },
        "nmr_1h": [2.09],  # singlet methyls
        "ir": [1715.0, 1365.0, 1220.0],
        "mass_spec": [(58.0, 1.0), (43.0, 0.8)],
    },
    "benzene": {
        "molecule": {
            "name": "benzene",
            "smiles": "c1ccccc1",
            "molecular_weight": 78.11,
            "functional_groups": ["aromatic"],
        },
        "nmr_1h": [7.27],
        "ir": [3040.0, 1600.0, 690.0],
        "mass_spec": [(78.0, 1.0), (77.0, 0.85)],
    },
    "acetic_acid": {
        "molecule": {
            "name": "acetic_acid",
            "smiles": "CC(=O)O",
            "molecular_weight": 60.05,
            "functional_groups": ["carboxylic_acid", "alkane_CH3"],
        },
        "nmr_1h": [2.10, 11.50],
        "ir": [1710.0, 2500.0, 1240.0],
        "mass_spec": [(60.0, 1.0), (45.0, 0.6)],
    },
}

# Synthesis reference data ----------------------------------------------------

SYNTHESIS_REFERENCE: Dict[str, Dict] = {
    "aspirin": {
        "target": {
            "name": "aspirin",
            "smiles": "CC(=O)Oc1ccccc1C(=O)O",
            "molecular_weight": 180.16,
            "functional_groups": ["ester", "aromatic", "carboxylic_acid"],
            "complexity": 30.0,
            "cost_per_gram": 1.10,
            "availability": "synthesis_required",
        },
        "expected": {
            "total_steps": 1,
            "overall_yield": 0.85,
        },
    },
    "ethyl_acetate": {
        "target": {
            "name": "ethyl_acetate",
            "smiles": "CCOC(=O)C",
            "molecular_weight": 88.11,
            "functional_groups": ["ester"],
            "complexity": 18.0,
            "cost_per_gram": 1.50,
            "availability": "synthesis_required",
        },
        "expected": {
            "total_steps": 1,
            "overall_yield": 0.82,
        },
    },
}


def list_known_targets() -> List[str]:
    """Return names of synthesis targets with curated routes."""
    return sorted(SYNTHESIS_REFERENCE.keys())


def list_known_molecules() -> List[str]:
    """Return molecules with spectroscopy reference data."""
    return sorted(SPECTROSCOPY_REFERENCE.keys())
