#!/usr/bin/env python3
"""
Run a compact comparison between ECH0 (QuLab) and other labs using novel experiments.
Outputs a JSON and Markdown summary in reports/.
"""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from complete_realistic_lab import RealisticTumor
from clinical_trials_simulation_lab import (
    ClinicalTrialsSimulationLab,
    TrialDesign,
    TrialPhase,
    EndpointType,
    TreatmentArm,
)


def load_oncology_lab_module():
    module_path = Path(__file__).with_name("oncology_lab.py")
    spec = importlib.util.spec_from_file_location("oncology_lab_file", module_path)
    module = importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load oncology_lab.py")
    spec.loader.exec_module(module)
    return module


def run_ech0_protocol() -> dict:
    """Compare baseline chemo vs ECH0 field-boosted protocol."""
    baseline = RealisticTumor(initial_cells=200, tumor_type="ovarian", seed=101)
    for _ in range(3):
        baseline.administer_drug("cisplatin")
        baseline.grow(14)
    baseline_stats = baseline.get_stats()

    ech0 = RealisticTumor(initial_cells=200, tumor_type="ovarian", seed=102)
    for cycle in range(3):
        fields = ["glucose", "oxygen", "temperature"]
        if cycle == 1:
            fields.append("lactate")
        ech0.apply_field_interventions(fields)
        ech0.administer_drug("cisplatin")
        ech0.grow(14)
    ech0_stats = ech0.get_stats()

    improvement = ech0_stats["shrinkage_percent"] - baseline_stats["shrinkage_percent"]

    return {
        "baseline": {
            "shrinkage_percent": baseline_stats["shrinkage_percent"],
            "alive_cells": baseline_stats["alive_cells"],
        },
        "ech0_protocol": {
            "shrinkage_percent": ech0_stats["shrinkage_percent"],
            "alive_cells": ech0_stats["alive_cells"],
            "field_schedule": ["glucose", "oxygen", "temperature", "lactate (cycle 2)"]
        },
        "improvement_points": improvement,
    }


def run_oncology_lab() -> dict:
    """Run a compact oncology lab experiment with growth + drug response + synergy."""
    oncology = load_oncology_lab_module()

    patient = oncology.PatientProfile(
        age=62,
        weight=72,
        height=168,
        bsa=1.78,
        gender="female",
        performance_status=1,
        creatinine_clearance=90,
        liver_function=0.95,
        albumin=4.0,
        prior_therapies=["carboplatin"],
        genetic_mutations={"EGFR": True, "KRAS": False},
    )

    tumor = oncology.TumorCharacteristics(
        initial_volume=3.8,
        doubling_time=80,
        carrying_capacity=450,
        hypoxic_fraction=0.12,
        proliferation_index=0.32,
        mutation_burden=9.0,
        angiogenic_switch=True,
        grade=3,
        tnm_stage="T2N0M0",
        biomarkers={"PD-L1": 35, "Ki-67": 30},
    )

    lab = oncology.OncologyLab(patient, tumor)

    time_points = np.array([0, 90, 180])
    gompertz = lab.gompertz_growth(time_points)

    drug = oncology.DrugProperties(
        name="Erlotinib",
        drug_class=oncology.DrugClass.TARGETED_EGFR,
        ic50=2.0,
        hill_coefficient=1.4,
        max_efficacy=0.85,
        half_life=36,
        clearance=4.5,
        volume_distribution=3.5,
        protein_binding=0.95,
        bioavailability=0.6,
        myelosuppression_risk=0.2,
    )

    response_at_5 = lab.hill_drug_response(5.0, drug)

    bliss = lab.bliss_independence(0.55, 0.45)
    ci = lab.loewe_additivity(1.2, 1.6, 2.0, 2.4)

    return {
        "tumor_growth_cm3": {
            "day_0": float(gompertz[0]),
            "day_90": float(gompertz[1]),
            "day_180": float(gompertz[2]),
        },
        "drug_response": {
            "drug": drug.name,
            "response_at_5uM": response_at_5,
        },
        "combination_synergy": {
            "bliss_expected": bliss,
            "loewe_ci": ci,
        },
    }


def run_clinical_trial(ech0_response_rate: float, baseline_response_rate: float) -> dict:
    """Simulate a compact clinical trial to compare response rates."""
    arms = [
        TreatmentArm(
            arm_id="A",
            arm_name="Standard Chemo",
            drug_name="Cisplatin",
            dose=75,
            schedule="q3w",
            route="IV",
            expected_response_rate=baseline_response_rate,
            expected_median_survival=12.0,
            toxicity_rate=0.35,
            dropout_rate=0.12,
        ),
        TreatmentArm(
            arm_id="B",
            arm_name="ECH0 Protocol",
            drug_name="Cisplatin + Fields",
            dose=75,
            schedule="q3w",
            route="IV",
            expected_response_rate=ech0_response_rate,
            expected_median_survival=16.0,
            toxicity_rate=0.30,
            dropout_rate=0.10,
        ),
    ]

    design = TrialDesign(
        phase=TrialPhase.PHASE_2,
        primary_endpoint=EndpointType.RESPONSE_RATE,
        sample_size=60,
        arms=arms,
        randomization_ratio=[1, 1],
        interim_analyses=[0.5],
    )

    lab = ClinicalTrialsSimulationLab(design, seed=202)
    lab.generate_patient_population()
    assignments = lab.simple_randomization()
    outcomes = lab.simulate_trial_outcomes(assignments)

    response_rates = {"A": 0, "B": 0}
    counts = {"A": 0, "B": 0}
    for outcome in outcomes:
        counts[outcome.arm_id] += 1
        response_rates[outcome.arm_id] += int(outcome.response)

    for arm_id in response_rates:
        if counts[arm_id]:
            response_rates[arm_id] /= counts[arm_id]

    return {
        "sample_size": design.sample_size,
        "response_rates": response_rates,
        "assignments": counts,
    }


def write_report(payload: dict) -> Path:
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    json_path = reports_dir / "echo_lab_comparison.json"
    json_path.write_text(json.dumps(payload, indent=2))

    md_path = reports_dir / "echo_lab_comparison.md"
    md_lines = [
        "# ECH0 vs Other Labs – Comparison Report",
        "",
        f"Generated: {payload['metadata']['generated_at']}",
        "",
        "## ECH0 (QuLab) Tumor Protocol",
        f"- Baseline shrinkage: {payload['ech0']['baseline']['shrinkage_percent']:.2f}%",
        f"- ECH0 shrinkage: {payload['ech0']['ech0_protocol']['shrinkage_percent']:.2f}%",
        f"- Improvement: {payload['ech0']['improvement_points']:.2f} points",
        "",
        "## Oncology Lab",
        f"- Tumor volume (cm³): day 0 {payload['oncology']['tumor_growth_cm3']['day_0']:.2f} → day 180 {payload['oncology']['tumor_growth_cm3']['day_180']:.2f}",
        f"- Erlotinib response at 5 μM: {payload['oncology']['drug_response']['response_at_5uM']:.2%}",
        f"- Bliss expected combo effect: {payload['oncology']['combination_synergy']['bliss_expected']:.2%}",
        f"- Loewe CI: {payload['oncology']['combination_synergy']['loewe_ci']:.2f}",
        "",
        "## Clinical Trials Simulation",
        f"- Sample size: {payload['clinical_trial']['sample_size']}",
        f"- Response rate (Standard): {payload['clinical_trial']['response_rates']['A']:.2%}",
        f"- Response rate (ECH0): {payload['clinical_trial']['response_rates']['B']:.2%}",
    ]
    md_path.write_text("\n".join(md_lines) + "\n")

    return md_path


def main() -> None:
    ech0_results = run_ech0_protocol()
    oncology_results = run_oncology_lab()

    baseline_rate = max(0.1, min(0.8, ech0_results["baseline"]["shrinkage_percent"] / 100))
    ech0_rate = max(0.1, min(0.9, ech0_results["ech0_protocol"]["shrinkage_percent"] / 100))
    clinical_results = run_clinical_trial(ech0_rate, baseline_rate)

    payload = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "ech0": ech0_results,
        "oncology": oncology_results,
        "clinical_trial": clinical_results,
    }

    report_path = write_report(payload)
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
