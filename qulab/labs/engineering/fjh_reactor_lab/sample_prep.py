"""
Planned sample-preparation protocol: Vulcan + aqueous gold, premix and dry.

This is a planned workflow, not a completed physical run.
Premix-and-dry does NOT imply atomic Au dispersion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .material import CarbonSupport, GoldPrecursor, MaterialHypothesis
from .types import UNKNOWN, DataProvenance, UnknownValue, is_unknown


@dataclass
class SamplePrepProtocol:
    """
    Planned precursor loading sequence.

    Vulcan carbon + liquid gold (aqueous HAuCl4) → premix → dry → load reactor.
    """

    carbon_support: CarbonSupport = field(default_factory=CarbonSupport)
    gold_precursor: GoldPrecursor = field(
        default_factory=lambda: GoldPrecursor(
            name="aqueous HAuCl4",
            alternative="liquid gold / gold chloride precursor",
            precursor_state="ionic_aqueous",
        )
    )
    steps: tuple[str, ...] = (
        "premix_vulcan_with_aqueous_gold",
        "dry",
        "load_reactor",
    )
    status: str = "planned"
    drying_completeness: float | UnknownValue = UNKNOWN
    residual_solvent: float | UnknownValue = UNKNOWN
    residual_chloride: float | UnknownValue = UNKNOWN
    precursor_loading_wt_percent: float | UnknownValue = UNKNOWN
    precursor_uniformity: float | UnknownValue = UNKNOWN
    note: str = (
        "Premix-and-dry is a precursor-distribution hypothesis, not evidence of "
        "atomic Au. Residual water/chloride and loading remain UNKNOWN until measured."
    )

    def to_dict(self) -> dict[str, Any]:
        def _u(v: Any) -> Any:
            return "UNKNOWN" if is_unknown(v) else v

        return {
            "status": self.status,
            "steps": list(self.steps),
            "carbon_support": self.carbon_support.name,
            "gold_precursor": self.gold_precursor.name,
            "precursor_state": self.gold_precursor.precursor_state,
            "drying_completeness": _u(self.drying_completeness),
            "residual_solvent": _u(self.residual_solvent),
            "residual_chloride": _u(self.residual_chloride),
            "precursor_loading_wt_percent": _u(self.precursor_loading_wt_percent),
            "precursor_uniformity": _u(self.precursor_uniformity),
            "note": self.note,
            "does_not_imply_atomic_Au": True,
            "provenance": DataProvenance.HYPOTHESIS.value,
        }


def planned_vulcan_gold_premix() -> SamplePrepProtocol:
    return SamplePrepProtocol()


def hypothesis_with_planned_prep() -> MaterialHypothesis:
    """Material hypothesis annotated with the planned premix/dry workflow."""
    hypo = MaterialHypothesis()
    hypo.gold_precursor.precursor_state = "ionic_aqueous_premix_then_dry_planned"
    hypo.label = (
        "Planned Vulcan + aqueous HAuCl4 premix/dry/load. "
        "Competing outcomes are hypotheses, NOT established facts. "
        "Do not assume atomic Au dispersion after drying."
    )
    return hypo
