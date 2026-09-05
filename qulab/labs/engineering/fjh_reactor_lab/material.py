"""
Material and precursor configuration for FJH carbon/Au hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import UNKNOWN, AuOutcome, CarbonOutcome, UnknownValue, is_unknown


@dataclass
class CarbonSupport:
    """Carbon support material specification."""

    name: str = "Vulcan XC-72R"
    alternative: str = "Vulcan XC-72"
    surface_area_m2_g: float | UnknownValue = UNKNOWN
    defect_density: float | UnknownValue = UNKNOWN
    surface_chemistry: str | UnknownValue = UNKNOWN


@dataclass
class GoldPrecursor:
    """Gold precursor specification."""

    name: str = "aqueous HAuCl4"
    alternative: str = "gold chloride precursor"
    loading_wt_percent: float | UnknownValue = UNKNOWN
    distribution_uniformity: float | UnknownValue = UNKNOWN  # 0-1 score
    precursor_state: str = "ionic_aqueous"


@dataclass
class MaterialHypothesis:
    """
    Competing material outcome hypotheses — NOT established facts.
    """

    carbon_support: CarbonSupport = field(default_factory=CarbonSupport)
    gold_precursor: GoldPrecursor = field(default_factory=GoldPrecursor)

    possible_carbon_outcomes: list[CarbonOutcome] = field(default_factory=lambda: [
        CarbonOutcome.UNCONVERTED,
        CarbonOutcome.PARTIAL_GRAPHITIZATION,
        CarbonOutcome.TURBOSTRATIC,
        CarbonOutcome.GRAPHITIC,
    ])

    possible_au_outcomes: list[AuOutcome] = field(default_factory=lambda: [
        AuOutcome.IONIC_RESIDUE,
        AuOutcome.ISOLATED_ATOMS,
        AuOutcome.FEW_ATOM_CLUSTERS,
        AuOutcome.NANOPARTICLES,
        AuOutcome.AGGLOMERATES,
        AuOutcome.VOLATILIZATION,
    ])

    label: str = (
        "These are competing hypotheses/outputs, NOT established facts. "
        "Do not assume atomic Au dispersion."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "carbon_support": {
                "name": self.carbon_support.name,
                "surface_area_m2_g": (
                    float(self.carbon_support.surface_area_m2_g)
                    if not is_unknown(self.carbon_support.surface_area_m2_g)
                    else "UNKNOWN"
                ),
            },
            "gold_precursor": {
                "name": self.gold_precursor.name,
                "loading_wt_percent": (
                    float(self.gold_precursor.loading_wt_percent)
                    if not is_unknown(self.gold_precursor.loading_wt_percent)
                    else "UNKNOWN"
                ),
            },
            "possible_carbon_outcomes": [o.value for o in self.possible_carbon_outcomes],
            "possible_au_outcomes": [o.value for o in self.possible_au_outcomes],
        }


def default_fjh_material_hypothesis() -> MaterialHypothesis:
    """Default hypothesis for Vulcan XC-72 / HAuCl4 FJH research."""
    return MaterialHypothesis()
