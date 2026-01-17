"""Shared validation status metadata and helpers."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class ValidationRange:
    """A validated numeric range."""

    name: str
    keys: Sequence[str]
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    units: Optional[str] = None
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def describe_violation(self, value: Any) -> Optional[str]:
        """Return a human-readable warning if value is outside the range."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None

        low_breach = self.minimum is not None and numeric < self.minimum
        high_breach = self.maximum is not None and numeric > self.maximum
        if not (low_breach or high_breach):
            return None

        bounds = []
        if self.minimum is not None:
            bounds.append(f"min {self.minimum}")
        if self.maximum is not None:
            bounds.append(f"max {self.maximum}")
        bound_text = " to ".join(bounds) if len(bounds) == 2 else bounds[0]
        units = f" {self.units}" if self.units else ""
        return (
            f"{self.name}={numeric}{units} is outside validated bounds ({bound_text}{units})."
            + (f" {self.note}" if self.note else "")
        )


def _range_list(*ranges: ValidationRange) -> List[ValidationRange]:
    return list(ranges)


VALIDATION_GATES: Dict[str, Dict[str, Any]] = {
    "materials": {
        "md_error_bounds_percent": 5.0,
        "coverage": {
            "reference_pairs": 42,
            "validation_focus": "NIST + Materials Project cross-checks on steel, aluminum, aerogel, and carbon composites.",
        },
        "ranges": _range_list(
            ValidationRange(
                name="temperature",
                keys=("temperature", "temperature_c"),
                minimum=-50.0,
                maximum=1200.0,
                units="°C",
                note="MD furnace calibration",
            ),
            ValidationRange(
                name="strain",
                keys=("strain", "max_strain", "engineering_strain"),
                minimum=0.0,
                maximum=0.2,
                units="ΔL/L",
                note="Validated tensile strain coverage",
            ),
        ),
    },
    "chemistry": {
        "md_error_bounds_percent": 5.0,
        "coverage": {
            "benchmarks": "Arrhenius/Eyring kinetics and spectroscopy datasets",
            "validated_reaction_window_k": "250K–1200K",
        },
        "ranges": _range_list(
            ValidationRange(
                name="temperature",
                keys=("temperature_k", "temperature"),
                minimum=250.0,
                maximum=1200.0,
                units="K",
                note="Kinetics benchmarks calibrated in this window",
            ),
            ValidationRange(
                name="pressure",
                keys=("pressure_bar", "pressure"),
                minimum=0.1,
                maximum=50.0,
                units="bar",
                note="Validated reactor pressure envelope",
            ),
        ),
    },
    "quantum": {
        "coverage": {
            "statevector_qubits": {"max": 30, "fidelity_floor": 0.99},
            "tensor_network_qubits": {"max": 50, "fidelity_floor": 0.97},
            "validated_gate_set": ["h", "x", "rx", "ry", "rz", "cnot", "cz"],
        },
        "ranges": _range_list(
            ValidationRange(
                name="num_qubits",
                keys=("num_qubits",),
                minimum=1,
                maximum=50,
                units="qubits",
                note="Statevector exact ≤30; tensor network approximation ≤50",
            ),
            ValidationRange(
                name="circuit_depth",
                keys=("circuit_depth",),
                minimum=1,
                maximum=200,
                units="layers",
                note="Depth >200 not benchmarked for fidelity drift",
            ),
        ),
    },
    "spectroscopy": {
        "coverage": {
            "resolution": "Validated on 64–8192 points with Gaussian noise injection.",
        },
        "ranges": _range_list(
            ValidationRange(
                name="data_points",
                keys=("data_points",),
                minimum=64,
                maximum=8192,
                units="samples",
                note="Outside range may reduce alignment accuracy",
            ),
        ),
    },
}


def _find_first(spec: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for key in keys:
        if key in spec:
            return spec[key]
    return None


def get_validation_status() -> Dict[str, Any]:
    """Return the current validation gates and coverage metadata."""
    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "gates": {
            lab: {
                "md_error_bounds_percent": details.get("md_error_bounds_percent"),
                "coverage": details.get("coverage", {}),
                "ranges": [rng.to_dict() for rng in details.get("ranges", [])],
            }
            for lab, details in VALIDATION_GATES.items()
        },
    }


def get_validation_warnings(lab_name: str, parameters: Dict[str, Any]) -> List[str]:
    """Emit warnings when a request exceeds validated ranges."""
    details = VALIDATION_GATES.get(lab_name)
    if not details:
        return []

    warnings: List[str] = []
    for rng in details.get("ranges", []):
        value = _find_first(parameters, rng.keys)
        violation = rng.describe_violation(value)
        if violation:
            warnings.append(violation)

    # Specialized quantum messaging for fidelity coverage.
    if lab_name == "quantum":
        num_qubits = parameters.get("num_qubits")
        try:
            qubits = int(num_qubits) if num_qubits is not None else None
        except (TypeError, ValueError):
            qubits = None

        if qubits:
            statevector_max = details["coverage"]["statevector_qubits"]["max"]
            tensor_max = details["coverage"]["tensor_network_qubits"]["max"]
            if qubits > statevector_max and qubits <= tensor_max:
                warnings.append(
                    f"num_qubits={qubits} uses tensor network approximation; "
                    f"validated fidelity floor {details['coverage']['tensor_network_qubits']['fidelity_floor']:.2f}."
                )
            elif qubits > tensor_max:
                warnings.append(
                    f"num_qubits={qubits} exceeds validated tensor network limit ({tensor_max} qubits). "
                    "Fidelity guarantees are not established."
                )

    return warnings
