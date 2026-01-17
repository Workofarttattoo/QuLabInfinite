#!/usr/bin/env python3
"""Run golden-path validations using the Simple QuLab API internals.

This script executes deterministic, representative experiments for each
Simple QuLab lab plus a generic chemistry lab call, then writes proofs
(JSON + Markdown) under reports/golden_runs.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
from fastapi import HTTPException

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from simple_qulab_api import (
    ExperimentRequest,
    list_labs,
    quantum_simulate,
    oncology_simulate,
    pharmacology_simulate,
    climate_simulate,
    ml_train,
    generic_lab_simulate,
)

OUTPUT_DIR = Path("reports/golden_runs")
SEED = 42


def run_and_save(name: str, payload: dict, output_dir: Path) -> dict:
    output_path = output_dir / f"{name}.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return {
        "name": name,
        "path": str(output_path),
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "labs": list_labs(),
    }

    proofs: list[dict] = []

    try:
        quantum_payload = quantum_simulate(
            ExperimentRequest(
                experiment_type="teleportation_fidelity",
                parameters={
                    "distance_km": 1,
                    "protocol": "GHZ",
                    "depolarizing_rate": 0.0001,
                },
            )
        )
        proofs.append(run_and_save("quantum_mechanics_teleportation", quantum_payload, OUTPUT_DIR))

        oncology_payload = oncology_simulate(
            ExperimentRequest(
                experiment_type="tumor_evolution",
                parameters={
                    "tumor_type": "NSCLC",
                    "initial_size_mm": 15,
                    "simulation_days": 90,
                    "treatment": "pembrolizumab",
                },
            )
        )
        proofs.append(run_and_save("oncology_tumor_evolution", oncology_payload, OUTPUT_DIR))

        pharmacology_payload = pharmacology_simulate(
            ExperimentRequest(
                experiment_type="pk_model",
                parameters={
                    "dose_mg": 100,
                    "clearance_L_hr": 5,
                    "volume_L": 70,
                },
            )
        )
        proofs.append(run_and_save("pharmacology_pk_model", pharmacology_payload, OUTPUT_DIR))

        climate_payload = climate_simulate(
            ExperimentRequest(
                experiment_type="temperature_projection",
                parameters={
                    "scenario": "RCP4.5",
                    "years": 100,
                },
            )
        )
        proofs.append(run_and_save("climate_temperature_projection", climate_payload, OUTPUT_DIR))

        ml_payload = ml_train(
            ExperimentRequest(
                experiment_type="model_comparison",
                parameters={
                    "models": ["random_forest", "gradient_boost", "svm"],
                    "dataset_size": 1000,
                },
            )
        )
        proofs.append(run_and_save("machine_learning_model_comparison", ml_payload, OUTPUT_DIR))

        chemistry_payload = generic_lab_simulate(
            "chemistry_lab",
            ExperimentRequest(
                experiment_type="reaction_kinetics",
                parameters={
                    "temperature_K": 298.15,
                    "pressure_bar": 1.0,
                    "activation_energy_kJ_mol": 52.0,
                    "rate_constant_s": 0.002,
                },
            ),
        )
        proofs.append(run_and_save("chemistry_reaction_kinetics", chemistry_payload, OUTPUT_DIR))

    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail)
        raise SystemExit(f"Validation failed: {detail}") from exc

    index = {
        "metadata": metadata,
        "proofs": proofs,
    }
    (OUTPUT_DIR / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True))

    readme_lines = [
        "# Golden Run Proofs",
        "",
        "This directory contains deterministic validation outputs from the Simple QuLab API.",
        "",
        "## Index",
    ]
    for proof in proofs:
        readme_lines.append(f"- `{proof['name']}` → `{proof['path']}`")

    (OUTPUT_DIR / "README.md").write_text("\n".join(readme_lines) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
