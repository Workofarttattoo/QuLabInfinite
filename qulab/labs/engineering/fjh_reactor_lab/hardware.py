"""
Physical hardware description for the operator's FJH reactor.

Known operator-stated facts are marked KNOWN_INPUT.
Literature-derived wire properties are LITERATURE_DERIVED.
Unmeasured lengths, resistances, and geometries remain UNKNOWN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .sample_prep import SamplePrepProtocol, planned_vulcan_gold_premix
from .types import (
    UNKNOWN,
    DataProvenance,
    UnknownValue,
    is_unknown,
)

# NEC / standard annealed copper at 20 C. Not a measured cable.
AWG4_COPPER_RESISTANCE_OHM_PER_M_20C = 0.0008152
AWG4_DIAMETER_MM = 5.189
AWG4_AREA_MM2 = 21.15


@dataclass
class HVWiring:
    """4 AWG welding wire on all HV contacts."""

    awg: int = 4
    conductor: str = "copper_welding_wire"
    used_on: str = "all_HV_contacts"
    resistance_ohm_per_m_20C: float = AWG4_COPPER_RESISTANCE_OHM_PER_M_20C
    diameter_mm: float = AWG4_DIAMETER_MM
    area_mm2: float = AWG4_AREA_MM2
    length_m: float | UnknownValue = UNKNOWN
    measured_resistance_ohm: float | UnknownValue = UNKNOWN
    resistivity_provenance: str = DataProvenance.LITERATURE_DERIVED.value

    def resistance_ohm(self) -> float | UnknownValue:
        """Total HV wiring resistance. UNKNOWN until length or measurement exists."""
        if not is_unknown(self.measured_resistance_ohm):
            return float(self.measured_resistance_ohm)
        if not is_unknown(self.length_m):
            return self.resistance_ohm_per_m_20C * float(self.length_m)
        return UnknownValue("HV wire length not measured")

    def estimate_for_length_m(self, length_m: float) -> float:
        return self.resistance_ohm_per_m_20C * length_m

    def to_dict(self) -> dict[str, Any]:
        R = self.resistance_ohm()
        return {
            "awg": self.awg,
            "conductor": self.conductor,
            "used_on": self.used_on,
            "resistance_ohm_per_m_20C": self.resistance_ohm_per_m_20C,
            "diameter_mm": self.diameter_mm,
            "area_mm2": self.area_mm2,
            "length_m": "UNKNOWN" if is_unknown(self.length_m) else float(self.length_m),
            "total_resistance_ohm": "UNKNOWN" if is_unknown(R) else float(R),
            "example_0.5m_ohm": self.estimate_for_length_m(0.5),
            "example_2.0m_ohm": self.estimate_for_length_m(2.0),
            "note": (
                "4 AWG copper is electrically small versus typical sample/contact "
                "resistance. Length remains UNKNOWN; do not treat example lengths "
                "as measured."
            ),
            "provenance": {
                "awg": DataProvenance.KNOWN_INPUT.value,
                "resistivity": self.resistivity_provenance,
                "length": DataProvenance.UNKNOWN.value,
            },
        }


@dataclass
class GraphiteRodElectrodes:
    """Two long, thin graphite rods sandwiching the sample."""

    count: int = 2
    material: str = "graphite"
    shape: str = "long_thin_rods"
    arrangement: str = "sample_between_two_rods"
    length_mm: float | UnknownValue = UNKNOWN
    diameter_mm: float | UnknownValue = UNKNOWN
    contact_area_mm2: float | UnknownValue = UNKNOWN
    gap_mm: float | UnknownValue = UNKNOWN
    resistivity_ohm_m: float | UnknownValue = UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "material": self.material,
            "shape": self.shape,
            "arrangement": self.arrangement,
            "length_mm": "UNKNOWN" if is_unknown(self.length_mm) else float(self.length_mm),
            "diameter_mm": "UNKNOWN" if is_unknown(self.diameter_mm) else float(self.diameter_mm),
            "contact_area_mm2": (
                "UNKNOWN" if is_unknown(self.contact_area_mm2) else float(self.contact_area_mm2)
            ),
            "gap_mm": "UNKNOWN" if is_unknown(self.gap_mm) else float(self.gap_mm),
            "note": (
                "Graphite rods are a large thermal sink relative to 1 g powder. "
                "Rod dimensions and contact area are UNKNOWN, so electrode heat "
                "capacity and contact resistance are not invented."
            ),
            "provenance": {
                "material": DataProvenance.KNOWN_INPUT.value,
                "shape": DataProvenance.KNOWN_INPUT.value,
                "dimensions": DataProvenance.UNKNOWN.value,
            },
        }


@dataclass
class BleedResistor:
    """
    Long brown resistor used to bleed the capacitor bank empty after charge.

    Parallel to the bank. Not a pulse-current path unless its resistance is
    low enough to steal energy during the flash. Resistance is UNKNOWN.
    """

    present: bool = True
    description: str = "long brown resistor across capacitor bank"
    purpose: str = "bleed_caps_empty_after_charge"
    resistance_ohm: float | UnknownValue = UNKNOWN
    in_pulse_current_path: bool = False

    def energy_stolen_during_pulse_J(
        self,
        voltage_V: float,
        pulse_duration_s: float,
        assumed_R_ohm: float,
    ) -> float:
        """Hypothetical bleed energy if resistance were assumed_R_ohm: V^2/R * t."""
        if assumed_R_ohm <= 0:
            return float("inf")
        return (voltage_V ** 2 / assumed_R_ohm) * pulse_duration_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "present": self.present,
            "description": self.description,
            "purpose": self.purpose,
            "resistance_ohm": (
                "UNKNOWN" if is_unknown(self.resistance_ohm) else float(self.resistance_ohm)
            ),
            "in_pulse_current_path": self.in_pulse_current_path,
            "hypothetical_energy_stolen_J_at_450V_5ms": {
                "if_1_kohm": self.energy_stolen_during_pulse_J(450, 0.005, 1e3),
                "if_10_kohm": self.energy_stolen_during_pulse_J(450, 0.005, 1e4),
                "if_100_kohm": self.energy_stolen_during_pulse_J(450, 0.005, 1e5),
                "note": (
                    "These are hypothetical illustrations, not measured values. "
                    "Typical HV bleeders are high-R and steal negligible flash energy."
                ),
            },
            "provenance": {
                "presence": DataProvenance.KNOWN_INPUT.value,
                "resistance": DataProvenance.UNKNOWN.value,
            },
        }


@dataclass
class PhysicalLabHardware:
    """Operator-described reactor hardware for this phase."""

    hv_wiring: HVWiring = field(default_factory=HVWiring)
    electrodes: GraphiteRodElectrodes = field(default_factory=GraphiteRodElectrodes)
    bleed_resistor: BleedResistor = field(default_factory=BleedResistor)
    sample_mass_g: float = 1.0
    sample_mass_provenance: str = DataProvenance.KNOWN_INPUT.value
    sample_prep: SamplePrepProtocol = field(default_factory=planned_vulcan_gold_premix)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_mass_g": self.sample_mass_g,
            "sample_mass_provenance": self.sample_mass_provenance,
            "hv_wiring": self.hv_wiring.to_dict(),
            "electrodes": self.electrodes.to_dict(),
            "bleed_resistor": self.bleed_resistor.to_dict(),
            "sample_prep": self.sample_prep.to_dict(),
            "hardware_control_enabled": False,
        }


def default_physical_hardware() -> PhysicalLabHardware:
    return PhysicalLabHardware()
