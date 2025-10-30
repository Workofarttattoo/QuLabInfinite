"""ECH0 bridge for QuLabInfinite.

This module gives the ECH0 LLM a single entry point to every laboratory
department plus a few cross-department workflows. Each experiment returns a
structured dictionary that is easy for downstream agents to parse or narrate.

Usage from the command line:

    python api/ech0_bridge.py --department materials --experiment tensile-test \
        --params '{"material": "SS 304", "temperature_c": 20}'

Usage from Python:

    from api.ech0_bridge import ECH0Bridge
    bridge = ECH0Bridge()
    result = bridge.run_experiment("materials", "tensile-test",
                                   material="SS 304", temperature_c=25)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

MATERIALS_DIR = BASE_DIR / "materials_lab"
if MATERIALS_DIR.exists() and str(MATERIALS_DIR) not in sys.path:
    sys.path.insert(0, str(MATERIALS_DIR))

VALIDATION_DIR = BASE_DIR / "validation"
if VALIDATION_DIR.exists() and str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

from chemistry_lab import ChemistryLaboratory, Compound  # type: ignore
from environmental_sim import EnvironmentalSimulator, create_leo_simulation  # type: ignore
from hive_mind import HiveMind, create_standard_agents  # type: ignore
from materials_lab import MaterialsLab  # type: ignore
from materials_lab.analysis_tools import (  # type: ignore
    compare_materials as _analysis_compare,
    pareto_front as _analysis_pareto,
    detect_outliers as _analysis_outliers,
    cli_summary as _analysis_summary,
)
from materials_lab.materials_database import MaterialProperties, MaterialsDatabase  # type: ignore
from physics_engine import PhysicsCore, SimulationConfig, SimulationScale, create_benchmark_simulation  # type: ignore
from quantum_lab import QuantumLabSimulator  # type: ignore
from validation.results_validator import ResultsValidator, ValidationStatus  # type: ignore

from qulab_api import QuLabSimulator, ExperimentRequest, ExperimentType  # type: ignore


class ECH0Bridge:
    """Unified control surface for all QuLabInfinite departments."""

    def __init__(self) -> None:
        self.materials_lab = MaterialsLab()
        self.environment = EnvironmentalSimulator()
        self.physics_gravity = np.array([0.0, 0.0, -9.80665])
        self.chemistry_lab = ChemistryLaboratory()
        self.quantum_lab = QuantumLabSimulator(num_qubits=4, verbose=False)
        self.simulator = QuLabSimulator()
        self.validator = ResultsValidator()
        self.hive_mind = HiveMind()
        # The agent creation is now handled by the main application that uses the bridge
        # for agent in create_standard_agents():
        #     self.hive_mind.register_agent(agent)

        self._department_map = {
            "materials": {
                "description": "Materials Science Laboratory: testing, design, and property prediction.",
                "experiments": {
                    "tensile-test": self._materials_tensile_test,
                    "compression-test": self._materials_compression_test,
                    "material-profile": self._materials_profile,
                    "compare": self._materials_compare,
                    "batch-tests": self._materials_batch_tests,
                    "pareto-front": self._materials_pareto,
                    "outlier-scan": self._materials_outliers,
                    "summary": self._materials_summary,
                    "search": self._materials_search,
                },
            },
            "environment": {
                "description": "Environmental Simulator: thermal, pressure, atmospheric, and wind profiles.",
                "experiments": {
                    "custom-environment": self._environment_custom,
                    "aerogel-preset": self._environment_aerogel,
                    "leo-cycle": self._environment_leo_cycle,
                },
            },
            "physics": {
                "description": "Physics Engine: mechanics benchmarks and energy diagnostics.",
                "experiments": {
                    "free-fall": self._physics_free_fall,
                    "projectile": self._physics_projectile,
                    "heat-conduction": self._physics_heat_conduction,
                },
            },
            "chemistry": {
                "description": "Chemistry Laboratory: retrosynthesis planning and spectroscopy predictions.",
                "experiments": {
                    "synthesis-plan": self._chemistry_synthesis_plan,
                    "spectroscopy": self._chemistry_spectroscopy,
                },
            },
            "quantum": {
                "description": "Quantum Laboratory: circuit templates and quantum materials metrics.",
                "experiments": {
                    "bell-pair": self._quantum_bell_pair,
                    "hadamard-chain": self._quantum_hadamard_chain,
                },
            },
            "hive-mind": {
                "description": "Hive Mind coordination layer: agent registry introspection and scheduling.",
                "experiments": {
                    "status": self._hive_status,
                },
            },
            "integrated": {
                "description": "Cross-department workflows orchestrated by QuLabSimulator.",
                "experiments": {
                    "cold-tensile-drop": self._integrated_cold_tensile_drop,
                    "custom-integrated": self._integrated_custom,
                },
            },
        }

    # ------------------------------------------------------------------ public

    def subscribe_to_hearing_channel(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Subscribes an external entity (like ECH0) to the user's voice commands.

        Args:
            callback: A function to be called when a new message is received.
                      The callback will receive a dictionary with the message data.
        """
        # We'll use the hive_mind's existing knowledge sharing system.
        # We need a unique ID for ech0 to subscribe.
        ech0_subscriber_id = "ech0_listener_process"
        
        self.hive_mind.knowledge.subscribe(ech0_subscriber_id, "hearing_channel")

        # To make this work, we need to inject the callback into the agent
        # processing logic. The simplest way is to create a pseudo-agent for ech0.
        class ECH0ListenerAgent:
            def process_broadcast(self, topic: str, data: Dict[str, Any]):
                if topic == "hearing_channel":
                    callback(data)

        # We register this pseudo-agent so it can receive callbacks.
        # This is a bit of a hack, but it cleanly integrates with the existing system.
        if self.hive_mind.registry.get_agent_instance(ech0_subscriber_id) is None:
            # The 'agent' part of the registration is just for the registry's data model.
            from hive_mind.hive_mind_core import Agent, AgentType
            pseudo_agent_data = Agent(agent_id=ech0_subscriber_id, agent_type=AgentType.ORCHESTRATION, capabilities=["listening"])
            self.hive_mind.registry.register_agent(pseudo_agent_data, ECH0ListenerAgent())

        print("ECH0 is now subscribed to the hearing channel.")


    def departments(self) -> Dict[str, Any]:
        """Return metadata for every department and its experiments."""
        return {
            name: {
                "description": info["description"],
                "experiments": sorted(info["experiments"].keys()),
            }
            for name, info in self._department_map.items()
        }

    def describe_experiment(self, department: str, experiment: str) -> str:
        """Return a short sentence describing the experiment."""
        dept = self._department_map.get(department)
        if dept is None or experiment not in dept["experiments"]:
            raise ValueError(f"Unknown experiment '{department}:{experiment}'")
        descriptions = {
            "tensile-test": "Run tensile test and summarize stress-strain metrics.",
            "compression-test": "Simulate compression response under a target strain.",
            "material-profile": "Retrieve detailed mechanical, thermal, and electrical curves for a material.",
            "compare": "Compare multiple materials across selected properties (normalized scores).",
            "batch-tests": "Execute a batch of materials experiments in one request.",
            "pareto-front": "Compute the Pareto front for objective combinations (e.g., strength vs density).",
            "outlier-scan": "Detect statistical outliers for a given property.",
            "summary": "Return tabular material summaries for downstream tools.",
            "search": "Search the materials database using lightweight filters.",
            "custom-environment": "Configure temperature, pressure, and wind, then snapshot the state.",
            "aerogel-preset": "Load the aerogel extreme cold preset.",
            "leo-cycle": "Simulate Low Earth Orbit thermal cycles for two orbits.",
            "free-fall": "Drop a particle and report energy consistency.",
            "projectile": "Simulate projectile motion over a configurable interval.",
            "heat-conduction": "Track heat transfer between hot and cold nodes.",
            "synthesis-plan": "Generate a retrosynthetic plan for a target compound.",
            "spectroscopy": "Predict basic spectroscopy scores for a prototype molecule.",
            "bell-pair": "Prepare a Bell state and report probability amplitudes.",
            "hadamard-chain": "Apply Hadamards to all qubits, measure distribution.",
            "status": "Summarize registered agents and queue depth.",
            "cold-tensile-drop": "Integrated aerogel tensile + drop test demo.",
            "custom-integrated": "Run a user-specified integrated request through QuLabSimulator.",
        }
        return descriptions.get(experiment, "")

    def run_experiment(self, department: str, experiment: str, **params: Any) -> Dict[str, Any]:
        """Execute an experiment and return structured output."""
        dept = self._department_map.get(department)
        if dept is None:
            raise ValueError(f"Unknown department '{department}'")

        handler = dept["experiments"].get(experiment)
        if handler is None:
            raise ValueError(f"Unknown experiment '{experiment}' for department '{department}'")

        payload = handler(params)
        payload["department"] = department
        payload["experiment"] = experiment
        return payload

    # -------------------------------------------------------------- materials

    def _materials_tensile_test(self, params: Dict[str, Any]) -> Dict[str, Any]:
        material = params.get("material", "SS 304")
        temperature_c = float(params.get("temperature_c", 25.0))
        temperature_k = temperature_c + 273.15
        result = self.materials_lab.tensile_test(material, temperature=temperature_k)

        validator_key = {
            "SS 304": "steel_304_yield_strength",
            "Al 6061-T6": "aluminum_6061_yield_strength",
        }.get(material)

        validation = None
        if validator_key:
            validation_raw = self.validator.validate(result.data["yield_strength"], validator_key)
            validation = self._validation_to_dict(validation_raw)

        return {
            "material": material,
            "temperature_C": temperature_c,
            "yield_strength_MPa": result.data["yield_strength"],
            "ultimate_strength_MPa": result.data["ultimate_strength"],
            "elongation_percent": result.data["elongation_at_break"],
            "youngs_modulus_MPa": result.data["youngs_modulus"],
            "stress_strain_sample": self._downsample_curve(
                result.data["strain"], result.data["stress"], target_points=60
            ),
            "validation": validation,
        }

    def _materials_compression_test(self, params: Dict[str, Any]) -> Dict[str, Any]:
        material = params.get("material", "SS 304")
        temperature_c = float(params.get("temperature_c", 25.0))
        temperature_k = temperature_c + 273.15
        max_strain = float(params.get("max_strain", 0.2))

        result = self.materials_lab.compression_test(
            material, temperature=temperature_k, max_strain=max_strain
        )

        return {
            "material": material,
            "temperature_C": temperature_c,
            "max_strain": max_strain,
            "compressive_strength_MPa": result.data["compressive_strength"],
            "compressive_modulus_MPa": result.data["compressive_modulus"],
            "stress_strain_sample": self._downsample_curve(
                result.data["strain"], result.data["stress"], target_points=60
            ),
        }

    def _materials_profile(self, params: Dict[str, Any]) -> Dict[str, Any]:
        material = params.get("material", "SS 304")
        profile = self.materials_lab.get_material_profile(material)
        return profile

    def _materials_compare(self, params: Dict[str, Any]) -> Dict[str, Any]:
        materials = params.get("materials") or ["SS 304", "Al 6061-T6"]
        properties = params.get("properties") or [["tensile_strength", "MPa"], ["density", "kg/m³"]]
        comparison = _analysis_compare(materials, [(p[0], p[1]) for p in properties])
        return {
            "materials": materials,
            "properties": properties,
            "records": [
                {
                    "property": record.property_name,
                    "unit": record.unit,
                    "values": record.values,
                    "best_material": record.best_material,
                    "normalized_scores": record.normalized_scores,
                }
                for record in comparison
            ],
        }

    def _materials_batch_tests(self, params: Dict[str, Any]) -> Dict[str, Any]:
        batch = params.get("experiments")
        if not isinstance(batch, list):
            raise ValueError("'experiments' must be a list of specifications")
        results = self.materials_lab.run_batch_experiments(batch)
        return {"results": results}

    def _materials_pareto(self, params: Dict[str, Any]) -> Dict[str, Any]:
        objectives = params.get("objectives") or {"tensile_strength": "max", "density": "min"}
        category = params.get("category")
        db = MaterialsDatabase()
        mats = db.materials.values()
        if category:
            mats = [m for m in mats if m.category.lower() == category.lower()]
        front = _analysis_pareto(mats, objectives)
        return {
            "objectives": objectives,
            "category": category,
            "materials": [m.name for m in front],
        }

    def _materials_outliers(self, params: Dict[str, Any]) -> Dict[str, Any]:
        property_name = params.get("property") or "density"
        threshold = float(params.get("z_threshold", 2.5))
        db = MaterialsDatabase()
        mats = db.materials.values()
        inliers, outliers = _analysis_outliers(mats, property_name, threshold)
        return {
            "property": property_name,
            "z_threshold": threshold,
            "outliers": [m.name for m in outliers],
            "inlier_count": len(inliers),
            "outlier_count": len(outliers),
        }

    def _materials_summary(self, params: Dict[str, Any]) -> Dict[str, Any]:
        materials = params.get("materials") or ["SS 304", "Al 6061-T6", "Ti-6Al-4V"]
        properties = params.get("properties") or [
            "density",
            "tensile_strength",
            "yield_strength",
            "thermal_conductivity",
        ]
        summary = _analysis_summary(materials, properties, lab=self.materials_lab)
        return summary

    def _materials_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        filters = {k: v for k, v in params.items() if k not in {"limit"}}
        results = self.materials_lab.search_materials(**filters)
        limit = int(params.get("limit", 10))
        return {
            "query": filters,
            "count": len(results),
            "materials": [m.name for m in results[:limit]],
        }

    # -------------------------------------------------------------- environment

    def _environment_custom(self, params: Dict[str, Any]) -> Dict[str, Any]:
        temperature_c = float(params.get("temperature_c", 25.0))
        pressure_bar = float(params.get("pressure_bar", 1.0))
        wind_m_s = float(params.get("wind_m_s", 0.0))
        controller = self.environment.controller
        controller.temperature.set_temperature(temperature_c, unit="C")
        controller.pressure.set_pressure(pressure_bar, unit="bar")
        controller.fluid.set_wind((wind_m_s, 0.0, 0.0), unit="m/s")
        snapshot = controller.get_conditions_at_position((0.0, 0.0, 0.0))
        return self._environment_snapshot(snapshot)

    def _environment_aerogel(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.environment.setup_aerogel_test(
            temperature_c=float(params.get("temperature_c", -200)),
            pressure_bar=float(params.get("pressure_bar", 0.001)),
            wind_mph=float(params.get("wind_mph", 30.0)),
        )
        snapshot = self.environment.controller.get_conditions_at_position((0.0, 0.0, 0.0))
        return self._environment_snapshot(snapshot)

    def _environment_leo_cycle(self, params: Dict[str, Any]) -> Dict[str, Any]:
        altitude = float(params.get("altitude_km", 400.0))
        simulator = create_leo_simulation(altitude)
        history = simulator.run_simulation(duration=5400 * 2, time_step=60.0)
        temperatures = [
            state["temperature"].get("base_temperature_K", 0.0)
            for state in history
            if "temperature" in state
        ]
        return {
            "altitude_km": altitude,
            "steps": len(history),
            "min_temperature_K": float(np.min(temperatures)),
            "max_temperature_K": float(np.max(temperatures)),
        }

    def _environment_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "temperature_C": float(snapshot["temperature_C"]),
            "temperature_K": float(snapshot["temperature_K"]),
            "pressure_bar": float(snapshot["pressure_bar"]),
            "pressure_Pa": float(snapshot["pressure_Pa"]),
            "wind_velocity_m_s": [float(v) for v in snapshot["wind_velocity_m_s"]],
            "gravity_m_s2": [float(v) for v in snapshot["gravity_m_s2"]],
        }

    # ---------------------------------------------------------------- physics

    def _physics_free_fall(self, params: Dict[str, Any]) -> Dict[str, Any]:
        mass_kg = float(params.get("mass_kg", 1.0))
        height_m = float(params.get("height_m", 10.0))
        duration_s = float(params.get("duration_s", 1.0))
        config = SimulationConfig(
            scale=SimulationScale.MACRO,
            domain_size=(1, 1, 1),
            resolution=0.01,
            timestep=0.001,
            duration=duration_s,
            enable_mechanics=True,
            gravity=self.physics_gravity,
        )

        core = PhysicsCore(config)
        core.add_particle(
            mass=mass_kg,
            position=np.array([0.0, 0.0, height_m]),
            velocity=np.zeros(3),
            radius=0.05,
        )
        core.simulate()
        particle = core.mechanics.particles[0]

        return {
            "mass_kg": mass_kg,
            "initial_height_m": height_m,
            "final_height_m": float(particle.position[2]),
            "final_velocity_m_s": float(particle.velocity[2]),
            "energy_error_fraction": float(core.mechanics.energy_error()),
            "steps": core.step_count,
        }

    def _physics_projectile(self, params: Dict[str, Any]) -> Dict[str, Any]:
        angle_deg = float(params.get("angle_deg", 45.0))
        speed_m_s = float(params.get("speed_m_s", 20.0))
        duration_s = float(params.get("duration_s", 2.0))
        config = SimulationConfig(
            scale=SimulationScale.MACRO,
            domain_size=(1, 1, 1),
            resolution=0.01,
            timestep=0.001,
            duration=duration_s,
            enable_mechanics=True,
            gravity=self.physics_gravity,
        )
        core = PhysicsCore(config)
        angle_rad = np.deg2rad(angle_deg)
        velocity = np.array(
            [speed_m_s * np.cos(angle_rad), 0.0, speed_m_s * np.sin(angle_rad)]
        )
        core.add_particle(
            mass=float(params.get("mass_kg", 0.25)),
            position=np.array([0.0, 0.0, 0.0]),
            velocity=velocity,
            radius=0.02,
        )
        core.simulate()
        particle = core.mechanics.particles[0]
        return {
            "launch_speed_m_s": speed_m_s,
            "launch_angle_deg": angle_deg,
            "flight_time_s": duration_s,
            "position_m": [float(x) for x in particle.position],
            "velocity_m_s": [float(v) for v in particle.velocity],
            "energy_error_fraction": float(core.mechanics.energy_error()),
        }

    def _physics_heat_conduction(self, params: Dict[str, Any]) -> Dict[str, Any]:
        simulation = create_benchmark_simulation("heat_conduction")
        simulation.simulate()
        hot = simulation.thermodynamics.nodes[0].temperature
        cold = simulation.thermodynamics.nodes[1].temperature
        stats = simulation.get_statistics()
        return {
            "final_hot_temperature_K": float(hot),
            "final_cold_temperature_K": float(cold),
            "total_internal_energy_J": float(stats["thermodynamics"]["total_internal_energy"]),
            "total_entropy_J_per_K": float(stats["thermodynamics"]["total_entropy"]),
        }

    # -------------------------------------------------------------- chemistry

    def _chemistry_synthesis_plan(self, params: Dict[str, Any]) -> Dict[str, Any]:
        target_name = params.get("target", "aspirin")
        target = self._chemistry_targets().get(target_name.lower())
        if target is None:
            raise ValueError(f"Unsupported target compound '{target_name}'")
        try:
            route = self.chemistry_lab.plan_synthesis(target, max_depth=int(params.get("max_depth", 3)))
        except ValueError as error:
            return {
                "target": target.name,
                "success": False,
                "error": str(error),
            }
        return {
            "target": target.name,
            "success": True,
            "overall_yield": float(route.overall_yield),
            "total_steps": int(route.total_steps),
            "difficulty_score": float(route.difficulty_score),
            "starting_materials": [m.name for m in route.starting_materials],
            "steps": [
                {
                    "name": step.name,
                    "type": step.reaction_type.value,
                    "yield_range": [float(step.yield_range[0]), float(step.yield_range[1])],
                    "reagents": step.reagents,
                    "conditions": step.conditions,
                }
                for step in route.steps
            ],
        }

    def _chemistry_spectroscopy(self, params: Dict[str, Any]) -> Dict[str, Any]:
        molecule_key = params.get("molecule", "aspirin").lower()
        targets = self._chemistry_targets()
        compound = targets.get(molecule_key)
        if compound is None:
            raise ValueError(f"Unsupported molecule '{molecule_key}'")

        molecule_dict = {
            "name": compound.name,
            "smiles": compound.smiles,
            "functional_groups": compound.functional_groups,
        }

        nmr_nucleus = params.get("nucleus", "1H")
        nmr = self.chemistry_lab.predict_nmr(molecule_dict, nucleus=nmr_nucleus)
        ir = self.chemistry_lab.predict_ir(molecule_dict)
        uv = self.chemistry_lab.predict_uv_vis(molecule_dict)

        return {
            "molecule": compound.name,
            "nmr": self._serialize_spectrum(nmr, sample_points=80),
            "ir": self._serialize_spectrum(ir, sample_points=80),
            "uv_vis": self._serialize_spectrum(uv, sample_points=80),
        }


    def _chemistry_targets(self) -> Dict[str, Compound]:
        """Predefined target compounds for quick retrosynthesis demos."""
        return {
            "aspirin": Compound(
                name="Aspirin",
                smiles="CC(=O)Oc1ccccc1C(=O)O",
                molecular_weight=180.16,
                functional_groups=["ester", "aromatic", "carboxylic_acid"],
                complexity=32.0,
                cost_per_gram=0.12,
                availability="synthesis_required",
            ),
            "paracetamol": Compound(
                name="Paracetamol",
                smiles="CC(=O)NC1=CC=C(C=C1)O",
                molecular_weight=151.16,
                functional_groups=["amide", "hydroxyl", "aromatic"],
                complexity=28.0,
                cost_per_gram=0.18,
                availability="synthesis_required",
            ),
        }

    # ----------------------------------------------------------------- quantum

    def _quantum_bell_pair(self, params: Dict[str, Any]) -> Dict[str, Any]:
        lab = self.quantum_lab
        lab.h(0)
        lab.cnot(0, 1)
        probabilities = lab.get_probabilities()
        return {
            "num_qubits": lab.num_qubits,
            "probabilities": {k: float(v) for k, v in probabilities.items()},
        }

    def _quantum_hadamard_chain(self, params: Dict[str, Any]) -> Dict[str, Any]:
        qubits = int(params.get("qubits", 4))
        lab = QuantumLabSimulator(num_qubits=qubits, verbose=False)
        for q in range(qubits):
            lab.h(q)
        probs = lab.get_probabilities()
        return {
            "num_qubits": qubits,
            "probabilities": {k: float(v) for k, v in probs.items()},
        }

    # --------------------------------------------------------------- hive mind

    def _hive_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        status = self.hive_mind.get_status()
        queue_info = status.get("queue", {})
        queued = queue_info.get("queued", {})
        return {
            "total_agents": status["registry"]["total_agents"],
            "active_agents": status["registry"]["active_agents"],
            "capabilities": status["registry"]["capabilities"],
            "queue_depth": int(sum(queued.values())),
        }

    # ------------------------------------------------------------- integrated

    def _integrated_cold_tensile_drop(self, params: Dict[str, Any]) -> Dict[str, Any]:
        request = ExperimentRequest(
            experiment_type=ExperimentType.INTEGRATED,
            description="Cold tensile test with drop probe",
            parameters={
                "material": params.get("material", "SS 304"),
                "temperature_c": params.get("temperature_c", -40.0),
                "pressure_bar": params.get("pressure_bar", 0.8),
                "wind_mph": params.get("wind_mph", 30.0),
                "specimen_mass_kg": params.get("specimen_mass_kg", 0.5),
                "drop_height_m": params.get("drop_height_m", 3.0),
                "simulation_duration_s": params.get("simulation_duration_s", 0.75),
            },
        )
        result = self.simulator.run(request)
        return {
            "experiment_id": result.experiment_id,
            "success": result.success,
            "data": result.data,
            "validation": result.validation,
            "notes": result.notes,
            "error": result.error_message,
        }

    def _integrated_custom(self, params: Dict[str, Any]) -> Dict[str, Any]:
        description = params.get("description", "ECH0 custom request")
        exp_type_str = params.get("experiment_type", "integrated_stack")
        try:
            exp_type = ExperimentType(exp_type_str)
        except ValueError as err:
            raise ValueError(f"Invalid ExperimentType '{exp_type_str}'") from err

        request = ExperimentRequest(
            experiment_type=exp_type,
            description=description,
            parameters=params.get("parameters", {}),
        )
        result = self.simulator.run(request)
        return {
            "experiment_id": result.experiment_id,
            "success": result.success,
            "data": result.data,
            "validation": result.validation,
            "notes": result.notes,
            "error": result.error_message,
        }

    # -------------------------------------------------------------- utilities

    def _validation_to_dict(self, validation) -> Dict[str, Any]:
        return {
            "status": validation.status.value
            if isinstance(validation.status, ValidationStatus)
            else str(validation.status),
            "error_percent": float(validation.error_percent),
            "z_score": float(validation.z_score),
            "simulated_value": float(validation.simulated_value),
            "reference_value": float(validation.reference_value),
            "uncertainty": float(validation.uncertainty),
            "message": validation.message,
            "passed_tests": list(validation.passed_tests),
            "failed_tests": list(validation.failed_tests),
        }

    def _downsample_curve(
        self,
        strain: List[float],
        stress: List[float],
        target_points: int,
    ) -> List[Dict[str, float]]:
        if not strain or not stress:
            return []

        total = min(len(strain), len(stress))
        if total <= target_points:
            indices = range(total)
        else:
            step = max(total // target_points, 1)
            indices = range(0, total, step)

        return [
            {"strain": float(strain[i]), "stress": float(stress[i])}
            for i in list(indices)[:target_points]
        ]

    def _serialize_spectrum(self, spectrum, sample_points: int) -> Dict[str, Any]:
        peaks = [
            {
                "position": float(peak.position),
                "intensity": float(peak.intensity),
                "width": float(peak.width),
                "multiplicity": peak.multiplicity,
                "assignment": peak.assignment,
            }
            for peak in spectrum.peaks[:10]
        ]

        x_axis = spectrum.x_axis
        y_axis = spectrum.y_axis
        total = len(x_axis)
        if total == 0:
            samples = []
        else:
            step = max(total // sample_points, 1)
            indices = range(0, total, step)
            samples = [
                {"x": float(x_axis[i]), "y": float(y_axis[i])}
                for i in list(indices)[:sample_points]
            ]

        return {
            "type": spectrum.spectrum_type.value,
            "x_label": spectrum.x_label,
            "y_label": spectrum.y_label,
            "peaks": peaks,
            "sample": samples,
        }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ECH0 bridge for QuLabInfinite.")
    parser.add_argument(
        "--department",
        help="Department name (materials, environment, physics, chemistry, quantum, hive-mind, integrated).",
    )
    parser.add_argument(
        "--experiment",
        help="Experiment name within the department.",
    )
    parser.add_argument(
        "--params",
        default="{}",
        help="JSON string with experiment parameters.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List departments and experiments.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    bridge = ECH0Bridge()

    if args.list or not args.department:
        print(json.dumps(bridge.departments(), indent=2))
        return

    params = json.loads(args.params)
    result = bridge.run_experiment(args.department, args.experiment, **params)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
