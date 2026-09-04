"""
FJH Reactor configuration model.

Structured reactor configuration with explicit UNKNOWN values.
Software calculates derived quantities (capacitance, stored energy).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .types import (
    UNKNOWN,
    AtmosphereType,
    CapacitorConnection,
    UnknownValue,
    is_unknown,
)


def _serialize_value(v: Any) -> Any:
    if is_unknown(v):
        return {"__unknown__": True, "reason": v.reason}
    if isinstance(v, (AtmosphereType, CapacitorConnection)):
        return v.value
    return v


@dataclass
class TemperatureResistanceModel:
    """Temperature-dependent sample resistance model."""

    reference_resistance_ohm: float | UnknownValue = UNKNOWN
    reference_temperature_K: float = 298.15
    temperature_coefficient: float | UnknownValue = UNKNOWN
    model_type: str = "linear"  # linear, exponential, tabulated


@dataclass
class IGBTModel:
    """IGBT conduction and switching loss models."""

    conduction_loss_model: str | UnknownValue = UNKNOWN
    switching_loss_model: str | UnknownValue = UNKNOWN
    on_resistance_ohm: float | UnknownValue = UNKNOWN
    switching_energy_J: float | UnknownValue = UNKNOWN


@dataclass
class SampleGeometry:
    """Sample geometry specification."""

    length_mm: float | UnknownValue = UNKNOWN
    width_mm: float | UnknownValue = UNKNOWN
    thickness_mm: float | UnknownValue = UNKNOWN
    cross_section_mm2: float | UnknownValue = UNKNOWN


@dataclass
class ElectrodeGeometry:
    """Electrode geometry specification."""

    contact_area_mm2: float | UnknownValue = UNKNOWN
    gap_mm: float | UnknownValue = UNKNOWN
    material: str | UnknownValue = UNKNOWN


@dataclass
class GasComposition:
    """Gas composition with residual oxygen tracking."""

    primary_gas: str = "argon"
    fractions: dict[str, float] = field(default_factory=dict)
    residual_oxygen_fraction: float | UnknownValue = UNKNOWN


@dataclass
class ReactorConfiguration:
    """
    Structured FJH reactor configuration.

    Known nominal values are defaults; measured values override when provided.
    Unknown values remain explicitly UNKNOWN.
    """

    # Capacitor bank — known nominal configuration
    capacitor_count: int = 12
    capacitor_nominal_voltage_V: float = 450.0
    capacitor_each_capacitance_uF: float = 900.0
    capacitor_connection: CapacitorConnection = CapacitorConnection.PARALLEL

    # Measured / uncertain capacitor properties
    measured_capacitance_each_uF: float | UnknownValue = UNKNOWN
    measured_ESR_each_ohm: float | UnknownValue = UNKNOWN
    measured_ESL_H: float | UnknownValue = UNKNOWN

    # Busbar and connection
    busbar_resistance_ohm: float | UnknownValue = UNKNOWN
    busbar_inductance_H: float | UnknownValue = UNKNOWN
    connection_resistance_ohm: float | UnknownValue = UNKNOWN
    electrode_contact_resistance_ohm: float | UnknownValue = UNKNOWN

    # Sample electrical
    sample_resistance_ohm: float | UnknownValue = UNKNOWN
    sample_resistance_vs_temperature: TemperatureResistanceModel = field(
        default_factory=TemperatureResistanceModel
    )

    # IGBT / switch
    igbt: IGBTModel = field(default_factory=IGBTModel)

    # Chamber / atmosphere
    chamber_pressure_Pa: float | UnknownValue = UNKNOWN
    gas_composition: GasComposition = field(default_factory=GasComposition)
    atmosphere_type: AtmosphereType = AtmosphereType.ARGON

    # Sample physical
    sample_mass_g: float | UnknownValue = UNKNOWN
    sample_geometry: SampleGeometry = field(default_factory=SampleGeometry)
    electrode_geometry: ElectrodeGeometry = field(default_factory=ElectrodeGeometry)

    # Thermal boundary
    ambient_temperature_K: float = 298.15
    initial_sample_temperature_K: float = 298.15

    # Discharge parameters
    initial_voltage_V: float | None = None  # defaults to nominal if None
    pulse_duration_s: float = 0.005  # 5 ms default

    # Safety: hardware control disabled for this phase
    hardware_control_enabled: bool = False

    def effective_capacitance_each_uF(self) -> float:
        """Return measured or nominal per-capacitor capacitance."""
        if not is_unknown(self.measured_capacitance_each_uF):
            return float(self.measured_capacitance_each_uF)
        return self.capacitor_each_capacitance_uF

    def total_capacitance_F(self) -> float:
        """Calculate total bank capacitance in Farads."""
        c_each = self.effective_capacitance_each_uF() * 1e-6  # uF -> F
        if self.capacitor_connection == CapacitorConnection.PARALLEL:
            return self.capacitor_count * c_each
        return c_each / self.capacitor_count

    def total_capacitance_uF(self) -> float:
        return self.total_capacitance_F() * 1e6

    def initial_stored_energy_J(self) -> float:
        """E = 0.5 * C * V^2"""
        v = self.initial_voltage_V or self.capacitor_nominal_voltage_V
        return 0.5 * self.total_capacitance_F() * v ** 2

    def effective_ESR_ohm(self) -> float:
        """Total ESR: parallel ESRs combine as 1/n, series as n*."""
        if is_unknown(self.measured_ESR_each_ohm):
            return 0.01  # placeholder for LEVEL 0/1 when ESR unknown
        esr_each = float(self.measured_ESR_each_ohm)
        if self.capacitor_connection == CapacitorConnection.PARALLEL:
            return esr_each / self.capacitor_count
        return esr_each * self.capacitor_count

    def effective_sample_resistance_ohm(self, temperature_K: float | None = None) -> float:
        """Sample resistance at given temperature."""
        T = temperature_K or self.initial_sample_temperature_K
        model = self.sample_resistance_vs_temperature
        if not is_unknown(model.reference_resistance_ohm):
            r_ref = float(model.reference_resistance_ohm)
            if not is_unknown(model.temperature_coefficient):
                alpha = float(model.temperature_coefficient)
                return r_ref * (1 + alpha * (T - model.reference_temperature_K))
            return r_ref
        if not is_unknown(self.sample_resistance_ohm):
            return float(self.sample_resistance_ohm)
        return 0.1  # placeholder for simulation when unknown

    def config_hash(self) -> str:
        """Deterministic hash for reproducibility."""
        d = self._to_dict_no_hash()
        raw = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _to_dict_no_hash(self) -> dict[str, Any]:
        return {
            "capacitor_count": self.capacitor_count,
            "capacitor_nominal_voltage_V": self.capacitor_nominal_voltage_V,
            "capacitor_each_capacitance_uF": self.capacitor_each_capacitance_uF,
            "capacitor_connection": self.capacitor_connection.value,
            "total_capacitance_uF": self.total_capacitance_uF(),
            "total_capacitance_F": self.total_capacitance_F(),
            "initial_stored_energy_J": self.initial_stored_energy_J(),
            "measured_capacitance_each_uF": _serialize_value(
                self.measured_capacitance_each_uF
            ),
            "measured_ESR_each_ohm": _serialize_value(self.measured_ESR_each_ohm),
            "measured_ESL_H": _serialize_value(self.measured_ESL_H),
            "busbar_resistance_ohm": _serialize_value(self.busbar_resistance_ohm),
            "busbar_inductance_H": _serialize_value(self.busbar_inductance_H),
            "connection_resistance_ohm": _serialize_value(
                self.connection_resistance_ohm
            ),
            "electrode_contact_resistance_ohm": _serialize_value(
                self.electrode_contact_resistance_ohm
            ),
            "sample_resistance_ohm": _serialize_value(self.sample_resistance_ohm),
            "chamber_pressure_Pa": _serialize_value(self.chamber_pressure_Pa),
            "atmosphere_type": self.atmosphere_type.value,
            "sample_mass_g": _serialize_value(self.sample_mass_g),
            "ambient_temperature_K": self.ambient_temperature_K,
            "initial_sample_temperature_K": self.initial_sample_temperature_K,
            "hardware_control_enabled": self.hardware_control_enabled,
        }

    def to_dict(self) -> dict[str, Any]:
        d = self._to_dict_no_hash()
        d["config_hash"] = self.config_hash()
        return d

    def unknown_parameters(self) -> list[str]:
        """List all parameters still marked UNKNOWN."""
        unknowns = []
        for key, val in asdict(self).items():
            if is_unknown(val):
                unknowns.append(key)
        return unknowns

    @classmethod
    def default_fjh_bank(cls) -> ReactorConfiguration:
        """Default configuration matching known 12x900uF @ 450V bank."""
        return cls()
