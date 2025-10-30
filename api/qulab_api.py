"""QuLabInfinite simulation orchestrator.

This module stitches together a workable subset of the QuLabInfinite codebase
so engineers can run consistent, reproducible virtual experiments. The focus is
on material testing, environmental conditioning, and a simple physics probe.
The routines are not “perfect replicas of reality”, but they provide a solid
baseline that keeps units consistent and exposes the datasets already present
in the repository.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# The materials laboratory is a single module rather than a package. We add the
# directory explicitly so `import materials_lab` resolves correctly.
MATERIALS_DIR = BASE_DIR / "materials_lab"
if MATERIALS_DIR.exists() and str(MATERIALS_DIR) not in sys.path:
    sys.path.insert(0, str(MATERIALS_DIR))

VALIDATION_DIR = BASE_DIR / "validation"
if VALIDATION_DIR.exists() and str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

from environmental_sim import EnvironmentalSimulator
from materials_lab import MaterialsLab
from physics_engine import PhysicsCore, SimulationConfig, SimulationScale
from validation.results_validator import (
    ResultsValidator,
    ValidationResult,
    ValidationStatus,
)


class ExperimentType(Enum):
    """Supported experiment categories."""

    MATERIAL_TEST = "material_test"
    ENVIRONMENT_ANALYSIS = "environment_analysis"
    PHYSICS_PROBE = "physics_probe"
    INTEGRATED = "integrated_stack"
    SAFETY_QUERY = "safety_query"
    ICE_ANALYSIS = "ice_analysis"


@dataclass
class ExperimentRequest:
    """Structured request used by the simulator."""

    experiment_type: ExperimentType
    description: str
    parameters: Dict[str, Any]


@dataclass
class ExperimentResult:
    """Result returned after executing an experiment."""

    experiment_id: str
    success: bool
    data: Dict[str, Any]
    validation: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None
    error_message: Optional[str] = None


class QuLabSimulator:
    """High-level façade that coordinates the individual laboratory modules."""

    _VALIDATION_KEYS: Dict[str, str] = {
        "SS 304": "steel_304_yield_strength",
        "Al 6061-T6": "aluminum_6061_yield_strength",
    }

    def __init__(self) -> None:
        self.materials_lab = MaterialsLab()
        self.environment = EnvironmentalSimulator()
        self.validator = ResultsValidator()
        self._experiment_counter = 0
        self._default_gravity = np.array([0.0, 0.0, -9.80665])
        self._dispatch_table = {
            ExperimentType.MATERIAL_TEST: self._run_material_test,
            ExperimentType.ENVIRONMENT_ANALYSIS: self._run_environment_analysis,
            ExperimentType.PHYSICS_PROBE: self._run_physics_probe,
            ExperimentType.INTEGRATED: self._run_integrated_stack,
            ExperimentType.SAFETY_QUERY: self._run_safety_query,
            ExperimentType.ICE_ANALYSIS: self._run_ice_analysis,
        }

    # ------------------------------------------------------------------ public

    def run(self, request: ExperimentRequest | str) -> ExperimentResult:
        """Execute an experiment request or simple natural-language command."""
        if isinstance(request, str):
            request = self._parse_text_request(request)

        experiment_id = self._next_experiment_id()
        handler = self._dispatch_table.get(request.experiment_type)
        if handler is None:
            return ExperimentResult(
                experiment_id=experiment_id,
                success=False,
                data={},
                error_message=f"Unsupported experiment type: {request.experiment_type}",
            )

        try:
            data, validation, notes = handler(request.parameters)
            return ExperimentResult(
                experiment_id=experiment_id,
                success=True,
                data=data,
                validation=validation,
                notes=notes,
            )
        except Exception as exc:
            return ExperimentResult(
                experiment_id=experiment_id,
                success=False,
                data={},
                error_message=str(exc),
            )

    def simulate_environment(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience wrapper around the environment controller."""
        temperature_c = float(parameters.get("temperature", 25.0))
        pressure_bar = float(parameters.get("pressure", 1.0))
        wind_speed = float(parameters.get("wind_speed", 0.0))
        wind_units = str(parameters.get("wind_units", "mph")).lower()
        if wind_units in {"m/s", "mps", "meters_per_second"}:
            wind_m_s = wind_speed
        else:
            wind_m_s = wind_speed * 0.44704

        humidity = float(parameters.get("relative_humidity", 0.6))
        duration_hours = float(parameters.get("duration_hours", 1.0))

        summary = self._configure_environment(temperature_c, pressure_bar, wind_m_s)
        summary["relative_humidity"] = humidity
        summary["duration_hours"] = duration_hours

        material_name = parameters.get("material")
        if material_name:
            try:
                ice = self.materials_lab.simulate_ice_growth(
                    material_name,
                    temperature_c + 273.15,
                    humidity,
                    duration_hours,
                )
                summary["ice_analysis"] = ice
            except Exception as exc:  # pragma: no cover - best effort
                summary["ice_analysis_error"] = str(exc)

        return summary

    def find_materials(self, criteria: Dict[str, Any]) -> List[str]:
        """Search materials matching high-level criteria."""
        criteria = criteria or {}
        mapped: Dict[str, Any] = {}

        if "category" in criteria:
            mapped["category"] = criteria["category"]
        if "density_min" in criteria:
            mapped["min_density"] = criteria["density_min"]
        if "density_max" in criteria:
            mapped["max_density"] = criteria["density_max"]
        if "yield_strength_min" in criteria:
            mapped["min_strength"] = criteria["yield_strength_min"]
        if "tensile_strength_min" in criteria:
            mapped["min_strength"] = max(
                mapped.get("min_strength", 0),
                criteria["tensile_strength_min"],
            )
        if "thermal_conductivity_min" in criteria:
            mapped["min_thermal_conductivity"] = criteria["thermal_conductivity_min"]
        if "cost_max" in criteria:
            mapped["max_cost"] = criteria["cost_max"]

        matches = self.materials_lab.search_materials(**mapped)
        return [material.name for material in matches]

    # ------------------------------------------------------------- material lab

    def _run_material_test(
        self,
        params: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[str]]:
        material = params.get("material")
        if not material:
            raise ValueError("Parameter 'material' is required for material tests.")

        temperature_c = float(params.get("temperature_c", 25.0))
        test_type = params.get("test_type", "tensile").lower()

        if self.materials_lab.get_material(material) is None:
            raise ValueError(f"Material '{material}' not found in database.")

        if test_type != "tensile":
            raise NotImplementedError(
                f"Only tensile testing is wired up in this integration (requested '{test_type}')."
            )

        temperature_k = temperature_c + 273.15
        test_result = self.materials_lab.tensile_test(
            material, temperature=temperature_k
        )
        summary = self._summarize_tensile_test(test_result.data, temperature_c)
        validation = self._validate_material_result(material, summary)
        notes = "Stress-strain curve is down-sampled to keep the payload compact."
        return summary, validation, notes

    # -------------------------------------------------------------- environment

    def _run_environment_analysis(
        self,
        params: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[str]]:
        env_params = {
            "temperature": params.get("temperature_c", 25.0),
            "pressure": params.get("pressure_bar", 1.0),
            "wind_speed": params.get("wind_m_s", params.get("wind_mph", 0.0)),
            "wind_units": "m/s" if "wind_m_s" in params else "mph",
            "relative_humidity": params.get("relative_humidity", 0.6),
            "duration_hours": params.get("duration_hours", 1.0),
        }
        if "material" in params:
            env_params["material"] = params["material"]

        summary = self.simulate_environment(env_params)
        return summary, None, None

    # ----------------------------------------------------------------- physics

    def _run_physics_probe(
        self,
        params: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[str]]:
        mass_kg = float(params.get("mass_kg", 0.25))
        start_height_m = float(params.get("initial_height_m", 5.0))
        duration_s = float(params.get("duration_s", 1.0))
        summary = self._simulate_free_fall(mass_kg, start_height_m, duration_s)
        notes = "Simple single-body free-fall probe using the mechanics engine."
        return summary, None, notes

    # --------------------------------------------------------------- integrated

    def _run_integrated_stack(
        self,
        params: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[str]]:
        material = params.get("material", "SS 304")
        temperature_c = float(params.get("temperature_c", 25.0))
        pressure_bar = float(params.get("pressure_bar", 1.0))
        wind_mph = float(params.get("wind_mph", 0.0))
        specimen_mass = float(params.get("specimen_mass_kg", 0.25))
        drop_height = float(params.get("drop_height_m", 5.0))
        duration_s = float(params.get("simulation_duration_s", 1.0))

        environment, _, _ = self._run_environment_analysis(
            {
                "temperature_c": temperature_c,
                "pressure_bar": pressure_bar,
                "wind_mph": wind_mph,
                "relative_humidity": params.get("relative_humidity", 0.6),
                "duration_hours": params.get("duration_hours", 1.0),
                "material": material,
            }
        )
        material_data, validation, _ = self._run_material_test(
            {
                "material": material,
                "temperature_c": temperature_c,
                "test_type": params.get("test_type", "tensile"),
            }
        )
        physics, _, _ = self._run_physics_probe(
            {
                "mass_kg": specimen_mass,
                "initial_height_m": drop_height,
                "duration_s": duration_s,
            }
        )

        combined = {
            "environment": environment,
            "material_test": material_data,
            "mechanics_probe": physics,
            "safety": self.materials_lab.get_material_safety(material) or {},
        }
        notes = (
            "Integrated stack: environmental state feeds the tensile test; "
            "free-fall probe provides a quick loading check for the same specimen mass."
        )
        return combined, validation, notes

    def _run_safety_query(
        self,
        params: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[str]]:
        material = params.get("material")
        if not material:
            raise ValueError("Parameter 'material' is required for safety queries.")

        safety = self.materials_lab.get_material_safety(material)
        data = {
            "material": material,
            "available": bool(safety),
            "safety": safety or {},
        }
        notes = None if safety else "No safety data available for requested material."
        return data, None, notes

    def _run_ice_analysis(
        self,
        params: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[str]]:
        material = params.get("material")
        if not material:
            raise ValueError("Parameter 'material' is required for ice analysis.")

        temperature_k = float(params.get("temperature_k", (params.get("temperature_c", -10.0) + 273.15)))
        humidity = float(params.get("relative_humidity", 0.7))
        duration_hours = float(params.get("duration_hours", 2.0))

        analysis = self.materials_lab.simulate_ice_growth(material, temperature_k, humidity, duration_hours)
        data = {
            "material": material,
            "inputs": {
                "temperature_k": temperature_k,
                "relative_humidity": humidity,
                "duration_hours": duration_hours,
            },
            "ice_analysis": analysis,
        }
        return data, None, None

    # ------------------------------------------------------------ helper logic

    def _summarize_tensile_test(
        self,
        data: Dict[str, Any],
        temperature_c: float,
    ) -> Dict[str, Any]:
        strain = data.get("strain", [])
        stress = data.get("stress", [])
        sample = self._downsample_curve(strain, stress, points=50)

        summary = {
            "temperature_C": temperature_c,
            "yield_strength_MPa": float(data.get("yield_strength", 0.0)),
            "ultimate_strength_MPa": float(data.get("ultimate_strength", 0.0)),
            "youngs_modulus_MPa": float(data.get("youngs_modulus", 0.0)),
            "elongation_percent": float(data.get("elongation_at_break", 0.0)),
            "toughness_MJ_m3": float(data.get("toughness", 0.0)),
            "stress_strain_sample": sample,
        }
        return summary

    def _validate_material_result(
        self,
        material: str,
        summary: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        key = self._VALIDATION_KEYS.get(material)
        if not key:
            return None

        simulated = summary.get("yield_strength_MPa")
        if simulated is None:
            return None

        validation = self.validator.validate(simulated, key)
        return self._validation_to_dict(validation)

    def _configure_environment(
        self,
        temperature_c: float,
        pressure_bar: float,
        wind_m_s: float,
    ) -> Dict[str, Any]:
        controller = self.environment.controller
        controller.temperature.set_temperature(temperature_c, unit="C")
        controller.pressure.set_pressure(pressure_bar, unit="bar")
        controller.fluid.set_wind((wind_m_s, 0.0, 0.0), unit="m/s")

        snapshot = controller.get_conditions_at_position((0.0, 0.0, 0.0))
        environment_summary = {
            "temperature_C": float(snapshot["temperature_C"]),
            "temperature_K": float(snapshot["temperature_K"]),
            "pressure_bar": float(snapshot["pressure_bar"]),
            "pressure_Pa": float(snapshot["pressure_Pa"]),
            "wind_velocity_m_s": [float(v) for v in snapshot["wind_velocity_m_s"]],
            "gravity_m_s2": [float(v) for v in snapshot["gravity_m_s2"]],
        }
        return environment_summary

    def _simulate_free_fall(
        self,
        mass_kg: float,
        start_height_m: float,
        duration_s: float,
    ) -> Dict[str, Any]:
        config = SimulationConfig(
            scale=SimulationScale.MACRO,
            domain_size=(1, 1, 1),
            resolution=0.1,
            timestep=0.001,
            duration=duration_s,
            enable_mechanics=True,
            gravity=self._default_gravity,
        )
        core = PhysicsCore(config)
        core.add_particle(
            mass=mass_kg,
            position=np.array([0.0, 0.0, start_height_m]),
            velocity=np.zeros(3),
            radius=0.05,
        )
        core.simulate()

        particle = core.mechanics.particles[0]
        summary = {
            "mass_kg": mass_kg,
            "initial_height_m": start_height_m,
            "final_height_m": float(particle.position[2]),
            "final_velocity_m_s": float(particle.velocity[2]),
            "kinetic_energy_J": float(core.mechanics.kinetic_energy()),
            "potential_energy_J": float(core.mechanics.potential_energy()),
            "energy_error_fraction": float(core.mechanics.energy_error()),
            "steps_completed": core.step_count,
        }
        return summary

    def _downsample_curve(
        self,
        strain: List[float],
        stress: List[float],
        points: int,
    ) -> List[Dict[str, float]]:
        if not strain or not stress:
            return []

        total = min(len(strain), len(stress))
        if total <= points:
            indices = range(total)
        else:
            step = max(total // points, 1)
            indices = range(0, total, step)

        return [
            {
                "strain": float(strain[i]),
                "stress": float(stress[i]),
            }
            for i in list(indices)[:points]
        ]

    def _validation_to_dict(self, result: ValidationResult) -> Dict[str, Any]:
        return {
            "status": result.status.value
            if isinstance(result.status, ValidationStatus)
            else str(result.status),
            "error_percent": float(result.error_percent),
            "z_score": float(result.z_score),
            "simulated_value": float(result.simulated_value),
            "reference_value": float(result.reference_value),
            "uncertainty": float(result.uncertainty),
            "message": result.message,
            "passed_tests": list(result.passed_tests),
            "failed_tests": list(result.failed_tests),
        }

    def _parse_text_request(self, text: str) -> ExperimentRequest:
        lower = text.lower()
        material_guess = self._infer_material_from_text(lower)

        if any(word in lower for word in ["safety", "msds", "hazard"]):
            return ExperimentRequest(
                experiment_type=ExperimentType.SAFETY_QUERY,
                description=text,
                parameters={"material": material_guess},
            )

        if any(word in lower for word in ["ice", "frost", "nucleation", "crystal growth"]):
            return ExperimentRequest(
                experiment_type=ExperimentType.ICE_ANALYSIS,
                description=text,
                parameters={
                    "material": material_guess,
                    "temperature_c": -20.0,
                    "relative_humidity": 0.75,
                    "duration_hours": 2.0,
                },
            )

        if any(word in lower for word in ["material", "tensile", "yield"]):
            return ExperimentRequest(
                experiment_type=ExperimentType.MATERIAL_TEST,
                description=text,
                parameters={
                    "material": material_guess,
                    "temperature_c": 25.0,
                    "test_type": "tensile",
                },
            )

        if any(word in lower for word in ["environment", "temperature", "pressure"]):
            return ExperimentRequest(
                experiment_type=ExperimentType.ENVIRONMENT_ANALYSIS,
                description=text,
                parameters={"material": material_guess},
            )

        if "integrated" in lower or "combined" in lower:
            return ExperimentRequest(
                experiment_type=ExperimentType.INTEGRATED,
                description=text,
                parameters={"material": material_guess},
            )

        return ExperimentRequest(
            experiment_type=ExperimentType.PHYSICS_PROBE,
            description=text,
            parameters={},
        )

    def _infer_material_from_text(self, lower: str, default: str = "SS 304") -> str:
        mapping = [
            ("airloy", "Airloy X103"),
            ("aerogel", "Airloy X103"),
            ("hastelloy", "Hastelloy X"),
            ("magnesium", "Magnesium AZ31B"),
            ("az31", "Magnesium AZ31B"),
            ("peek", "PEEK 450G"),
            ("kapton", "Kapton HN Polyimide"),
            ("kevlar", "Kevlar 49 Fabric"),
            ("graphene", "Graphene Aerogel"),
            ("tungsten", "Tungsten Carbide WC-Co K20"),
            ("zirconia", "Zirconia PSZ"),
            ("inconel", "Inconel 718"),
            ("y b c o", "YBCO Superconductor"),
            ("electrolyte", "Lithium Hexafluorophosphate Electrolyte"),
            ("stainless", "SS 304"),
            ("ss 304", "SS 304"),
            ("6061", "Al 6061-T6"),
            ("aluminum", "Al 6061-T6"),
        ]
        for token, material in mapping:
            if token in lower:
                return material
        return default

    def _next_experiment_id(self) -> str:
        self._experiment_counter += 1
        return f"exp-{self._experiment_counter:05d}"

    # ------------------------------------------------------------------- demo

    def demo(self) -> Dict[str, Any]:
        """Run a representative integrated experiment for quick inspection."""
        request = ExperimentRequest(
            experiment_type=ExperimentType.INTEGRATED,
            description="Stainless steel tensile test under cold, windy conditions.",
            parameters={
                "material": "SS 304",
                "temperature_c": -40.0,
                "pressure_bar": 0.8,
                "wind_mph": 30.0,
                "specimen_mass_kg": 0.5,
                "drop_height_m": 3.0,
                "simulation_duration_s": 0.75,
            },
        )
        result = self.run(request)
        return {
            "request": request.description,
            "result": {
                "experiment_id": result.experiment_id,
                "success": result.success,
                "data": result.data,
                "validation": result.validation,
                "notes": result.notes,
                "error": result.error_message,
            },
        }


def main() -> None:
    simulator = QuLabSimulator()
    demo_payload = simulator.demo()
    print(json.dumps(demo_payload, indent=2))


if __name__ == "__main__":
    main()
