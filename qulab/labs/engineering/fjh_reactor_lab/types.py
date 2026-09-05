"""
FJH Reactor Digital Twin - Common types and enums.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class UnknownValue:
    """Sentinel for explicitly unknown physical parameters."""

    __slots__ = ("reason",)

    def __init__(self, reason: str = "not measured"):
        self.reason = reason

    def __repr__(self) -> str:
        return f"UNKNOWN({self.reason})"

    def __bool__(self) -> bool:
        return False


UNKNOWN = UnknownValue()


def is_unknown(value: Any) -> bool:
    return isinstance(value, UnknownValue)


def require_known(value: Any, name: str) -> float:
    """Raise if value is UNKNOWN; otherwise return as float."""
    if is_unknown(value):
        raise ValueError(f"Parameter '{name}' is UNKNOWN: {value.reason}")
    return float(value)


class ModelLevel(Enum):
    """Electrical/thermal model fidelity levels."""

    LEVEL_0 = 0  # Idealized RC sanity model
    LEVEL_1 = 1  # RLC lumped circuit
    LEVEL_2 = 2  # Temperature-dependent resistance
    LEVEL_3 = 3  # Coupled electrical/thermal
    LEVEL_4 = 4  # Future spatial/multiphysics (placeholder)


class SanityStatus(Enum):
    """Physics sanity check outcomes."""

    VALID = "VALID"
    QUESTIONABLE = "QUESTIONABLE"
    PHYSICALLY_INVALID = "PHYSICALLY_INVALID"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class DataProvenance(Enum):
    """Provenance classification for AI interface."""

    KNOWN_INPUT = "KNOWN_INPUT"
    MEASURED_RESULT = "MEASURED_RESULT"
    SIMULATED_RESULT = "SIMULATED_RESULT"
    LITERATURE_DERIVED = "LITERATURE_DERIVED_ASSUMPTION"
    HYPOTHESIS = "HYPOTHESIS"
    UNKNOWN = "UNKNOWN"


class CapacitorConnection(Enum):
    PARALLEL = "parallel"
    SERIES = "series"


class AtmosphereType(Enum):
    VACUUM = "vacuum"
    ARGON = "argon"
    INERT_OTHER = "inert_other"
    USER_DEFINED = "user_defined"


class UltrasoundStage(Enum):
    PRECURSOR_SONICATION = "precursor_sonication_before_loading"
    DURING_FLASH = "acoustic_excitation_during_flash"
    DISABLED = "disabled"


class CarbonOutcome(Enum):
    UNCONVERTED = "unconverted_carbon"
    PARTIAL_GRAPHITIZATION = "partially_graphitized_carbon"
    TURBOSTRATIC = "turbostratic_graphene_like_carbon"
    GRAPHITIC = "graphitic_carbon"


class AuOutcome(Enum):
    IONIC_RESIDUE = "Au_ionic_precursor_residue"
    ISOLATED_ATOMS = "isolated_Au_atoms"
    FEW_ATOM_CLUSTERS = "few_atom_Au_clusters"
    NANOPARTICLES = "Au_nanoparticles"
    AGGLOMERATES = "larger_Au_agglomerates"
    VOLATILIZATION = "Au_loss_volatilization"


@dataclass
class TimeSeries:
    """Generic time-series result container."""

    time_s: list[float]
    values: list[float]
    unit: str
    label: str
    provenance: DataProvenance = DataProvenance.SIMULATED_RESULT


@dataclass
class EnergyAccounting:
    """Energy balance for a simulated pulse."""

    initial_capacitor_energy_J: float
    remaining_capacitor_energy_J: float
    sample_energy_J: float
    busbar_losses_J: float
    switch_losses_J: float
    contact_losses_J: float
    other_losses_J: float
    balance_error_J: float
    balance_error_fraction: float
    is_conserved: bool
    model_level: ModelLevel


@dataclass
class HypothesisScores:
    """Material outcome hypothesis scores — NOT validated predictions."""

    graphene_conversion_score: float
    au_single_atom_retention_score: float
    au_cluster_risk: float
    au_nanoparticle_risk: float
    au_loss_risk: float
    carbon_damage_risk: float
    label: str = "HYPOTHESIS SCORES — not validated material composition predictions"

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "provenance": DataProvenance.HYPOTHESIS.value,
            "graphene_conversion_score": self.graphene_conversion_score,
            "au_single_atom_retention_score": self.au_single_atom_retention_score,
            "au_cluster_risk": self.au_cluster_risk,
            "au_nanoparticle_risk": self.au_nanoparticle_risk,
            "au_loss_risk": self.au_loss_risk,
            "carbon_damage_risk": self.carbon_damage_risk,
        }


@dataclass
class SimulationResult:
    """Complete simulation output."""

    experiment_id: str
    model_level: ModelLevel
    sanity_status: SanityStatus
    sanity_messages: list[str]
    energy: EnergyAccounting
    V_cap: TimeSeries
    V_sample: TimeSeries
    current: TimeSeries
    P_sample: TimeSeries
    T_sample: TimeSeries | None
    hypothesis_scores: HypothesisScores | None
    uncertainty: dict[str, Any] | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        def _ts(ts: TimeSeries | None) -> dict | None:
            if ts is None:
                return None
            return {
                "time_s": ts.time_s,
                "values": ts.values,
                "unit": ts.unit,
                "label": ts.label,
                "provenance": ts.provenance.value,
            }

        return {
            "experiment_id": self.experiment_id,
            "model_level": self.model_level.name,
            "sanity_status": self.sanity_status.value,
            "sanity_messages": self.sanity_messages,
            "energy": {
                "initial_capacitor_energy_J": self.energy.initial_capacitor_energy_J,
                "remaining_capacitor_energy_J": self.energy.remaining_capacitor_energy_J,
                "sample_energy_J": self.energy.sample_energy_J,
                "busbar_losses_J": self.energy.busbar_losses_J,
                "switch_losses_J": self.energy.switch_losses_J,
                "contact_losses_J": self.energy.contact_losses_J,
                "other_losses_J": self.energy.other_losses_J,
                "balance_error_J": self.energy.balance_error_J,
                "balance_error_fraction": self.energy.balance_error_fraction,
                "is_conserved": bool(self.energy.is_conserved),
            },
            "V_cap": _ts(self.V_cap),
            "V_sample": _ts(self.V_sample),
            "current": _ts(self.current),
            "P_sample": _ts(self.P_sample),
            "T_sample": _ts(self.T_sample),
            "hypothesis_scores": (
                self.hypothesis_scores.to_dict() if self.hypothesis_scores else None
            ),
            "uncertainty": self.uncertainty,
            "metadata": self.metadata,
        }
