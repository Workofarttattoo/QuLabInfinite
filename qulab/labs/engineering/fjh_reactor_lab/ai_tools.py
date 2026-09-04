"""
AI Scientist interface tools for FJH reactor digital twin.

Distinguishes KNOWN INPUT, MEASURED RESULT, SIMULATED RESULT,
LITERATURE-DERIVED ASSUMPTION, HYPOTHESIS, UNKNOWN.
"""

from __future__ import annotations

from typing import Any

from .config import ReactorConfiguration
from .ledger import ExperimentLedger
from .types import DataProvenance, HypothesisScores


class FJHAIScientistTools:
    """Tools for QuLab AI layer to query FJH reactor simulations."""

    def __init__(self, ledger: ExperimentLedger | None = None):
        self.ledger = ledger or ExperimentLedger()

    def what_assumption_dominates_uncertainty(
        self, uncertainty_result: dict[str, Any]
    ) -> dict[str, Any]:
        dominant = uncertainty_result.get("dominant_uncertain_parameters", [])
        sensitivity = uncertainty_result.get("sensitivity", {})
        return {
            "question": "What assumption dominates uncertainty?",
            "answer": (
                f"Dominant uncertain parameters: {dominant}. "
                f"Sensitivity map: {sensitivity}"
            ),
            "provenance": DataProvenance.SIMULATED_RESULT.value,
            "dominant_parameters": dominant,
            "sensitivity": sensitivity,
        }

    def which_measurement_would_improve_model(
        self, config: ReactorConfiguration, uncertainty_result: dict[str, Any]
    ) -> dict[str, Any]:
        dominant = uncertainty_result.get("dominant_uncertain_parameters", [])
        recommendations = []
        mapping = {
            "sample_resistance_ohm": "Four-point probe sample resistance vs temperature",
            "contact_resistance_ohm": "Contact resistance measurement before/after flash",
            "ESR_ohm": "ESR measurement of capacitor bank at discharge frequency",
            "specific_heat_J_kg_K": "DSC measurement of carbon support",
            "precursor_uniformity": "HAADF-STEM or EDX mapping of precursor distribution",
            "residual_oxygen_fraction": "Residual gas analysis (RGA) of chamber",
            "thermal_contact": "IR thermography during discharge",
            "igbt_current_rating_A": "Read Infineon part number and Ic / Icm from the package, do not fire to discover it",
            "side_electrolytic_esr": "Do not pulse-test the JCCON cans; they are not flash-rated",
        }
        for param in dominant[:3]:
            if param in mapping:
                recommendations.append({"parameter": param, "measurement": mapping[param]})

        unknowns = config.unknown_parameters()
        return {
            "question": "Which measurement would improve this model most?",
            "answer": recommendations,
            "unknown_parameters": unknowns,
            "provenance": DataProvenance.HYPOTHESIS.value,
        }

    def why_simulations_differ(
        self, experiment_id_a: str, experiment_id_b: str
    ) -> dict[str, Any]:
        comparison = self.ledger.compare(experiment_id_a, experiment_id_b)
        return {
            "question": "Why did simulation A differ from simulation B?",
            "comparison": comparison,
            "provenance": DataProvenance.SIMULATED_RESULT.value,
        }

    def variables_correlating_with_au_aggregation(
        self, experiment_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Identify parameter correlations with Au aggregation risk."""
        correlations = []
        for r in experiment_results:
            params = r.get("parameters", {})
            scores = r.get("hypothesis_scores", {})
            if scores:
                correlations.append({
                    "parameters": params,
                    "au_cluster_risk": scores.get("au_cluster_risk"),
                    "au_nanoparticle_risk": scores.get("au_nanoparticle_risk"),
                })
        return {
            "question": "Which variables correlate with Au aggregation risk?",
            "data": correlations,
            "provenance": DataProvenance.HYPOTHESIS.value,
            "note": "Correlations are HYPOTHESIS-level until calibrated against TEM/STEM",
        }

    def evidence_to_falsify_hypothesis(
        self, hypothesis: str = "isolated_Au_atoms"
    ) -> dict[str, Any]:
        falsification = {
            "isolated_Au_atoms": [
                "HAADF-STEM showing particles > 1 nm",
                "XANES showing bulk metallic Au signature only",
                "ICP-MS showing Au loss > predicted",
            ],
            "graphene_conversion": [
                "Raman D/G ratio inconsistent with turbostratic/graphene",
                "XRD showing only amorphous carbon",
            ],
        }
        return {
            "question": "What experimental evidence would falsify the current hypothesis?",
            "hypothesis": hypothesis,
            "falsification_criteria": falsification.get(hypothesis, []),
            "provenance": DataProvenance.HYPOTHESIS.value,
        }

    def characterization_for_isolated_vs_nanoparticle(
        self,
    ) -> dict[str, Any]:
        return {
            "question": "Which characterization method distinguishes isolated Au from nanoparticles?",
            "methods": [
                {
                    "method": "HAADF-STEM",
                    "distinguishes": "Single atoms vs clusters/nanoparticles by Z-contrast",
                    "provenance": DataProvenance.LITERATURE_DERIVED.value,
                },
                {
                    "method": "XANES/EXAFS",
                    "distinguishes": "Oxidation state and coordination environment",
                    "provenance": DataProvenance.LITERATURE_DERIVED.value,
                },
                {
                    "method": "CO-DRIFTS",
                    "distinguishes": "Single-atom vs cluster CO binding modes",
                    "provenance": DataProvenance.LITERATURE_DERIVED.value,
                },
            ],
            "note": "Simulation cannot replace these measurements",
        }

    @staticmethod
    def classify_provenance(value_type: str) -> str:
        """Map value type to provenance classification."""
        mapping = {
            "input": DataProvenance.KNOWN_INPUT.value,
            "measured": DataProvenance.MEASURED_RESULT.value,
            "simulated": DataProvenance.SIMULATED_RESULT.value,
            "literature": DataProvenance.LITERATURE_DERIVED.value,
            "hypothesis": DataProvenance.HYPOTHESIS.value,
            "unknown": DataProvenance.UNKNOWN.value,
        }
        return mapping.get(value_type, DataProvenance.UNKNOWN.value)

    def format_hypothesis_score(self, scores: HypothesisScores) -> str:
        """Format scores without implying validated composition percentages."""
        return (
            f"{scores.label}\n"
            f"  graphene_conversion_score: {scores.graphene_conversion_score:.3f}\n"
            f"  au_single_atom_retention_score: {scores.au_single_atom_retention_score:.3f}\n"
            f"  au_cluster_risk: {scores.au_cluster_risk:.3f}\n"
            f"  au_nanoparticle_risk: {scores.au_nanoparticle_risk:.3f}\n"
            f"  au_loss_risk: {scores.au_loss_risk:.3f}\n"
            f"  carbon_damage_risk: {scores.carbon_damage_risk:.3f}"
        )
