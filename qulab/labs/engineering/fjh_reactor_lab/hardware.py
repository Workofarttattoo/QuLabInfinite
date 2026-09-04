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
class InfineonIGBT:
    """
    Operator switch: Infineon IGBT rated 600 V.

    Voltage rating is KNOWN. Exact part number and current ratings are UNKNOWN.
    600 V vs a 450 V bank is voltage-legal with ~150 V label margin.
    Pulse current capability is the missing constraint.
    """

    manufacturer: str = "Infineon"
    voltage_rating_V: float = 600.0
    part_number: str | UnknownValue = UNKNOWN
    current_rating_A: float | UnknownValue = UNKNOWN
    pulse_current_rating_A: float | UnknownValue = UNKNOWN
    package: str | UnknownValue = UNKNOWN

    def voltage_headroom_V(self, bank_voltage_V: float = 450.0) -> float:
        return self.voltage_rating_V - bank_voltage_V

    def voltage_ok_for_bank(self, bank_voltage_V: float = 450.0) -> bool:
        return bank_voltage_V <= self.voltage_rating_V

    def to_dict(self) -> dict[str, Any]:
        def _u(v: Any) -> Any:
            return "UNKNOWN" if is_unknown(v) else v

        return {
            "manufacturer": self.manufacturer,
            "voltage_rating_V": self.voltage_rating_V,
            "part_number": _u(self.part_number),
            "current_rating_A": _u(self.current_rating_A),
            "pulse_current_rating_A": _u(self.pulse_current_rating_A),
            "package": _u(self.package),
            "voltage_headroom_vs_450V_bank_V": self.voltage_headroom_V(450.0),
            "voltage_ok_for_450V_bank": self.voltage_ok_for_bank(450.0),
            "note": (
                "600 V Infineon rating covers the 450 V capacitor label. "
                "Inductive turn-off spikes can still exceed 600 V. "
                "Continuous/pulse current and the exact Infineon part remain UNKNOWN. "
                "A 600 V discrete is often tens of amps; FJH peaks can be kiloamps."
            ),
            "provenance": {
                "manufacturer": DataProvenance.KNOWN_INPUT.value,
                "voltage_rating_V": DataProvenance.KNOWN_INPUT.value,
                "part_number": DataProvenance.UNKNOWN.value,
                "current_rating": DataProvenance.UNKNOWN.value,
            },
        }


@dataclass
class NonFlashElectrolyticBank:
    """
    Operator inventory: 10 electrolytic capacitors, explicitly NOT flash-rated.

    Can markings stated by the operator: JCCON, CE10H / 105 °C, 450 V,
    4700 µF, CD136. These are general-purpose aluminum electrolytics
    (power-supply / inverter filter class), not pulse-dump parts.

    Energy numbers below are VIRTUAL ONLY. Do not use this bank as the
    FJH capacitor dump path.
    """

    count: int = 10
    manufacturer: str = "JCCON"
    series_marking: str = "CE10H"
    temperature_rating_C: float = 105.0
    voltage_rating_V: float = 450.0
    capacitance_each_uF: float = 4700.0
    form_factor: str = "CD136"
    flash_rated: bool = False
    operator_stated_not_flash_rated: bool = True
    usable_as_fjh_dump_bank: bool = False
    capacitance_tolerance_fraction: float = 0.20
    esr_ohm_each: float | UnknownValue = UNKNOWN
    ripple_current_A: float | UnknownValue = UNKNOWN
    pulse_current_A: float | UnknownValue = UNKNOWN
    can_diameter_mm: float | UnknownValue = UNKNOWN
    can_height_mm: float | UnknownValue = UNKNOWN

    def energy_each_J(self, voltage_V: float | None = None) -> float:
        v = self.voltage_rating_V if voltage_V is None else voltage_V
        return 0.5 * (self.capacitance_each_uF * 1e-6) * v ** 2

    def energy_bank_J(self, voltage_V: float | None = None) -> float:
        return self.count * self.energy_each_J(voltage_V)

    def total_capacitance_uF(self) -> float:
        return self.count * self.capacitance_each_uF

    def to_dict(self) -> dict[str, Any]:
        def _u(v: Any) -> Any:
            return "UNKNOWN" if is_unknown(v) else v

        return {
            "count": self.count,
            "manufacturer": self.manufacturer,
            "series_marking": self.series_marking,
            "temperature_rating_C": self.temperature_rating_C,
            "voltage_rating_V": self.voltage_rating_V,
            "capacitance_each_uF": self.capacitance_each_uF,
            "total_capacitance_uF": self.total_capacitance_uF(),
            "form_factor": self.form_factor,
            "flash_rated": self.flash_rated,
            "operator_stated_not_flash_rated": self.operator_stated_not_flash_rated,
            "usable_as_fjh_dump_bank": self.usable_as_fjh_dump_bank,
            "energy_each_J_at_450V": self.energy_each_J(450.0),
            "energy_bank_J_at_450V": self.energy_bank_J(450.0),
            "capacitance_tolerance_fraction": self.capacitance_tolerance_fraction,
            "esr_ohm_each": _u(self.esr_ohm_each),
            "ripple_current_A": _u(self.ripple_current_A),
            "pulse_current_A": _u(self.pulse_current_A),
            "can_diameter_mm": _u(self.can_diameter_mm),
            "can_height_mm": _u(self.can_height_mm),
            "note": (
                "Operator said these are not flash-rated. JCCON CD136 4700 µF / 450 V "
                "is a general-purpose 105 °C aluminum electrolytic (filter / PSU class). "
                "A millisecond dump is outside that rating. Energy figures are virtual "
                "½CV² only. Do not fire these into the sample path."
            ),
            "provenance": {
                "count": DataProvenance.KNOWN_INPUT.value,
                "manufacturer": DataProvenance.KNOWN_INPUT.value,
                "voltage_capacitance": DataProvenance.KNOWN_INPUT.value,
                "not_flash_rated": DataProvenance.KNOWN_INPUT.value,
                "esr_ripple_pulse": DataProvenance.UNKNOWN.value,
                "can_size": DataProvenance.UNKNOWN.value,
            },
        }


def evaluate_nonflash_side_bank(
    flash_bank_energy_J: float = 1093.5,
    sample_mass_g: float = 1.0,
    specific_heat_J_kg_K: float = 710.0,
    t0_K: float = 298.15,
    bank: NonFlashElectrolyticBank | None = None,
) -> dict[str, Any]:
    """
    Virtual energy comparison of the side JCCON bank vs the 12×900 µF flash path.

    Does not authorize using the side bank. Returns do_not_use=True.
    """
    side = bank or NonFlashElectrolyticBank()
    e_side = side.energy_bank_J(450.0)
    e_each = side.energy_each_J(450.0)
    e_flash = flash_bank_energy_J
    mass_kg = sample_mass_g / 1000.0
    adiabatic_T = t0_K + e_side / (mass_kg * specific_heat_J_kg_K)
    combined_E = e_flash + e_side
    combined_T = t0_K + combined_E / (mass_kg * specific_heat_J_kg_K)
    return {
        "do_not_use": True,
        "usable_as_fjh_dump_bank": False,
        "this_is_not_a_firing_recommendation": True,
        "side_bank": side.to_dict(),
        "flash_bank_energy_J": e_flash,
        "side_bank_energy_J": e_side,
        "energy_each_J": e_each,
        "energy_ratio_vs_flash_bank": e_side / e_flash if e_flash else None,
        "combined_energy_if_paralleled_J": combined_E,
        "virtual_adiabatic_1g_side_only_K": adiabatic_T,
        "virtual_adiabatic_1g_combined_K": combined_T,
        "virtual_adiabatic_exceeds_carbon_model_domain": adiabatic_T > 3500,
        "voltage_legal_vs_450V_label": side.voltage_rating_V >= 450.0,
        "voltage_legal_is_not_pulse_rating": True,
        "why_not": [
            "Operator stated these electrolytics are not flash-rated.",
            "JCCON CD136 / CE 105 °C 4700 µF 450 V is a general-purpose filter electrolytic, not a pulse-dump part.",
            "A millisecond capacitor dump looks like a near-short to a ripple-rated can: venting, rupture, electrolyte spray, fire.",
            f"Ten cans store ~{e_side:.1f} J — about {e_side / e_flash:.2f}× the 12×900 µF flash bank. That is a larger stored-energy hazard, not a safer substitute.",
            "ESR, ripple current, and pulse current of these cans are UNKNOWN. Filter electrolytics typically dump energy into their own ESR, not into the sample.",
            "The Infineon IGBT is 600 V with UNKNOWN current. Dumping ~4.8 kJ through it is worse than the 1.1 kJ flash bank, not better.",
            "Paralleling them with the 12×900 µF bank mixes unknown ESR networks and still leaves 10 cans outside their rating.",
        ],
        "catalog_class_note": (
            "Matching JCCON CD136 4700 µF / 450 V listings are sold as general-purpose "
            "105 °C aluminum electrolytics with bolt terminals. Jianghai's same CD136 "
            "family name is a power-supply / inverter ripple series, not a flash series. "
            "Can size and ripple numbers from catalogs are not measurements of these 10 cans."
        ),
        "safety": {
            "hardware_control_enabled": False,
            "do_not_fire": True,
            "do_not_parallel_onto_flash_bank": True,
            "do_not_substitute_for_900uF_flash_caps": True,
        },
    }


@dataclass
class PhysicalLabHardware:
    """Operator-described reactor hardware for this phase."""

    hv_wiring: HVWiring = field(default_factory=HVWiring)
    electrodes: GraphiteRodElectrodes = field(default_factory=GraphiteRodElectrodes)
    bleed_resistor: BleedResistor = field(default_factory=BleedResistor)
    igbt: InfineonIGBT = field(default_factory=InfineonIGBT)
    side_electrolytic_bank: NonFlashElectrolyticBank = field(
        default_factory=NonFlashElectrolyticBank
    )
    sample_mass_g: float = 1.0
    sample_mass_provenance: str = DataProvenance.KNOWN_INPUT.value
    planned_test_mass_g: float = 0.5
    planned_test_mass_note: str = (
        "0.5 g is the planned virtual test load on the 12×900 µF bank. "
        "Not a firing authorization."
    )
    sample_prep: SamplePrepProtocol = field(default_factory=planned_vulcan_gold_premix)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_mass_g": self.sample_mass_g,
            "sample_mass_provenance": self.sample_mass_provenance,
            "planned_test_mass_g": self.planned_test_mass_g,
            "planned_test_mass_note": self.planned_test_mass_note,
            "hv_wiring": self.hv_wiring.to_dict(),
            "electrodes": self.electrodes.to_dict(),
            "bleed_resistor": self.bleed_resistor.to_dict(),
            "igbt": self.igbt.to_dict(),
            "side_electrolytic_bank": self.side_electrolytic_bank.to_dict(),
            "sample_prep": self.sample_prep.to_dict(),
            "hardware_control_enabled": False,
        }


def default_physical_hardware() -> PhysicalLabHardware:
    return PhysicalLabHardware()
