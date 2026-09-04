"""
FJH Reactor Digital Twin — Main laboratory integration.

Flash Joule Heating reactor simulation for virtual DOE.
SIMULATION ONLY — no hardware control in this phase.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qulab.core.base_lab import BaseLab, register_lab

from .ai_tools import FJHAIScientistTools
from .atmosphere import compare_atmospheres, create_atmosphere_state
from .calibration import CalibrationDatabase
from .config import ReactorConfiguration
from .dashboard import build_dashboard
from .doe import (
    DEFAULT_FJH_FACTORS,
    DEFAULT_OBJECTIVES,
    compute_pareto_front,
    latin_hypercube_doe,
    one_factor_at_a_time,
)
from .electrical import simulate_electrical
from .energy import compute_energy_accounting
from .ledger import ExperimentLedger
from .material import default_fjh_material_hypothesis
from .sanity import run_sanity_checks
from .scoring import compute_hypothesis_scores
from .thermal import simulate_thermal_lumped
from .types import AtmosphereType, ModelLevel, SimulationResult
from .ultrasound import UltrasoundConfig, compare_ultrasound_hypothesis
from .uncertainty import run_monte_carlo

logger = logging.getLogger(__name__)

MODEL_VERSION = "fjh_twin_v1"


@register_lab(
    name="fjh_reactor",
    category="engineering",
    description="FJH Reactor Digital Twin + Virtual DOE for Flash Joule Heating",
    version="1.0.0",
    tags=("fjh", "flash-joule-heating", "digital-twin", "doe", "simulation-only"),
)
class FJHReactorLab(BaseLab):
    """
    FJH Reactor Digital Twin laboratory.

    SIMULATION ONLY — hardware_control_enabled must remain False.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.ledger = ExperimentLedger(
            self.config.get("ledger_db", "data/fjh_experiment_ledger.db")
        )
        self.calibration_db = CalibrationDatabase(
            self.config.get("calibration_db", "data/fjh_calibration.db")
        )
        self.ai_tools = FJHAIScientistTools(self.ledger)

    def run_experiment(self, experiment_spec: dict[str, Any]) -> dict[str, Any]:
        """Route experiment by type."""
        self._track_experiment()
        exp_type = experiment_spec.get("experiment_type", "simulate_pulse")

        handlers = {
            "simulate_pulse": self._simulate_pulse,
            "sanity_check": self._sanity_check_only,
            "doe_latin_hypercube": self._run_doe_lhs,
            "doe_ofat": self._run_doe_ofat,
            "compare_atmospheres": self._compare_atmospheres,
            "compare_ultrasound": self._compare_ultrasound,
            "monte_carlo": self._run_monte_carlo,
            "dashboard": self._build_dashboard,
            "ai_query": self._ai_query,
        }

        handler = handlers.get(exp_type, self._simulate_pulse)
        return handler(experiment_spec)

    def _build_config(self, spec: dict[str, Any]) -> ReactorConfiguration:
        cfg = ReactorConfiguration.default_fjh_bank()
        overrides = spec.get("reactor_config", {})
        for key, val in overrides.items():
            if hasattr(cfg, key):
                if key == "atmosphere_type" and isinstance(val, str):
                    setattr(cfg, key, AtmosphereType(val))
                else:
                    setattr(cfg, key, val)
        if "initial_voltage_V" in spec:
            cfg.initial_voltage_V = spec["initial_voltage_V"]
        if "pulse_duration_s" in spec:
            cfg.pulse_duration_s = spec["pulse_duration_s"]
        if "atmosphere_type" in spec:
            at = spec["atmosphere_type"]
            cfg.atmosphere_type = AtmosphereType(at) if isinstance(at, str) else at
        # Enforce simulation-only phase
        cfg.hardware_control_enabled = False
        return cfg

    def _simulate_pulse(self, spec: dict[str, Any]) -> dict[str, Any]:
        cfg = self._build_config(spec)
        model_level = ModelLevel(spec.get("model_level", ModelLevel.LEVEL_1.value))
        if isinstance(model_level, int):
            model_level = ModelLevel(model_level)

        contact_R = spec.get("contact_resistance_ohm")
        if contact_R is not None:
            cfg.electrode_contact_resistance_ohm = contact_R

        electrical = simulate_electrical(
            cfg, model_level=model_level, duration_s=cfg.pulse_duration_s
        )
        if "error" in electrical:
            return {"status": "error", "message": electrical["error"]}

        ml = electrical.get("model_level", model_level)
        if isinstance(ml, ModelLevel):
            pass
        else:
            ml = model_level

        energy = compute_energy_accounting(cfg, electrical, ml, contact_R)

        thermal = None
        T_sample_ts = None
        if model_level.value >= 2:
            thermal = simulate_thermal_lumped(cfg, electrical["P_sample"])
            T_sample_ts = thermal.T_sample

        max_I = max(electrical["current"].values) if electrical["current"].values else 0
        min_V = min(electrical["V_cap"].values) if electrical["V_cap"].values else 0

        sanity = run_sanity_checks(
            cfg, energy=energy, model_level=ml,
            max_current_A=max_I, min_voltage_V=min_V,
            rectangular_pulse=spec.get("rectangular_pulse"),
        )

        material = default_fjh_material_hypothesis()
        scores = compute_hypothesis_scores(
            thermal=thermal,
            material=material,
            peak_current_A=max_I,
            delivered_energy_J=energy.sample_energy_J,
            atmosphere_type=(
                cfg.atmosphere_type.value
                if hasattr(cfg.atmosphere_type, "value")
                else str(cfg.atmosphere_type)
            ),
            ultrasound_enabled=spec.get("ultrasound_enabled", False),
        )

        experiment_id = spec.get("experiment_id") or str(uuid.uuid4())[:12]
        result = SimulationResult(
            experiment_id=experiment_id,
            model_level=ml,
            sanity_status=sanity.status,
            sanity_messages=sanity.messages,
            energy=energy,
            V_cap=electrical["V_cap"],
            V_sample=electrical["V_sample"],
            current=electrical["current"],
            P_sample=electrical["P_sample"],
            T_sample=T_sample_ts,
            hypothesis_scores=scores,
            uncertainty=None,
            metadata={"model_version": MODEL_VERSION},
        )

        assumptions = [
            "Capacitor label values are nominal, not measured",
            "ESR/ESL use placeholders when unknown",
            "Hypothesis scores are uncalibrated",
            "Hardware control disabled (simulation-only phase)",
        ]

        self.ledger.record(
            configuration_hash=cfg.config_hash(),
            input_parameters=cfg.to_dict(),
            assumptions=assumptions,
            unknown_parameters=cfg.unknown_parameters(),
            simulation_results=result.to_dict(),
            experiment_id=experiment_id,
        )

        atm = create_atmosphere_state(cfg)
        dashboard = build_dashboard(cfg, result, atmosphere=atm)

        return {
            "status": "success",
            "experiment_id": experiment_id,
            "simulation_result": result.to_dict(),
            "dashboard": dashboard,
            "sanity": sanity.to_dict(),
            "hardware_control_enabled": False,
            "simulation_only": True,
        }

    def _sanity_check_only(self, spec: dict[str, Any]) -> dict[str, Any]:
        cfg = self._build_config(spec)
        rectangular = spec.get("rectangular_pulse", {"V": 450, "I": 1000, "t_s": 0.005})
        sanity = run_sanity_checks(
            cfg, rectangular_pulse=rectangular,
            model_level=ModelLevel.LEVEL_0,
        )
        return {
            "status": "success",
            "sanity": sanity.to_dict(),
            "stored_energy_J": cfg.initial_stored_energy_J(),
            "rectangular_pulse_energy_J": 450 * 1000 * 0.005,
        }

    def _run_doe_lhs(self, spec: dict[str, Any]) -> dict[str, Any]:
        n_samples = spec.get("n_samples", 5)
        runs = latin_hypercube_doe(DEFAULT_FJH_FACTORS, n_samples=n_samples)
        results = []
        for run in runs[:n_samples]:
            run_spec = {
                "experiment_type": "simulate_pulse",
                "reactor_config": self._params_to_config(run.parameters),
                "model_level": ModelLevel.LEVEL_2.value,
            }
            r = self._simulate_pulse(run_spec)
            obj = self._extract_objectives(r)
            results.append({
                "run_id": run.run_id,
                "parameters": run.parameters,
                "objectives": obj,
                "sanity_status": r.get("sanity", {}).get("status"),
            })
        pareto = compute_pareto_front(results, DEFAULT_OBJECTIVES)
        return {
            "status": "success",
            "design_method": "latin_hypercube",
            "n_runs": len(results),
            "results": results,
            "pareto_front": [
                {"run_id": p.run_id, "objectives": p.objectives, "parameters": p.parameters}
                for p in pareto
            ],
        }

    def _run_doe_ofat(self, spec: dict[str, Any]) -> dict[str, Any]:
        baseline = spec.get("baseline", {"sample_resistance_ohm": 0.1})
        factors = DEFAULT_FJH_FACTORS[:3]
        runs = one_factor_at_a_time(factors, baseline, n_levels=2)
        results = []
        for run in runs[:6]:
            run_spec = {
                "experiment_type": "simulate_pulse",
                "reactor_config": self._params_to_config(run.parameters),
                "model_level": ModelLevel.LEVEL_1.value,
            }
            r = self._simulate_pulse(run_spec)
            results.append({
                "run_id": run.run_id,
                "parameters": run.parameters,
                "peak_current_A": max(r["simulation_result"]["current"]["values"]),
                "delivered_energy_J": r["simulation_result"]["energy"]["sample_energy_J"],
            })
        return {"status": "success", "design_method": "one_factor_at_a_time", "results": results}

    def _compare_atmospheres(self, spec: dict[str, Any]) -> dict[str, Any]:
        cfg_vac = self._build_config(spec)
        cfg_vac.atmosphere_type = AtmosphereType.VACUUM
        cfg_ar = self._build_config(spec)
        cfg_ar.atmosphere_type = AtmosphereType.ARGON
        comparison = compare_atmospheres(cfg_vac, cfg_ar)

        r_vac = self._simulate_pulse({**spec, "reactor_config": {"atmosphere_type": "vacuum"}})
        r_ar = self._simulate_pulse({**spec, "reactor_config": {"atmosphere_type": "argon"}})

        return {
            "status": "success",
            "atmosphere_comparison": comparison,
            "vacuum_simulation": {
                "experiment_id": r_vac.get("experiment_id"),
                "hypothesis_scores": r_vac.get("simulation_result", {}).get("hypothesis_scores"),
            },
            "argon_simulation": {
                "experiment_id": r_ar.get("experiment_id"),
                "hypothesis_scores": r_ar.get("simulation_result", {}).get("hypothesis_scores"),
            },
        }

    def _compare_ultrasound(self, spec: dict[str, Any]) -> dict[str, Any]:
        cfg = self._build_config(spec)
        ultrasound = UltrasoundConfig(enabled=True)

        def _sim(c, ultrasound_enabled=False, ultrasound=None):
            s = {"experiment_type": "simulate_pulse", "ultrasound_enabled": ultrasound_enabled}
            return self._simulate_pulse(s)

        comparison = compare_ultrasound_hypothesis(_sim, cfg, ultrasound)
        return {"status": "success", **comparison}

    def _run_monte_carlo(self, spec: dict[str, Any]) -> dict[str, Any]:
        cfg = self._build_config(spec)
        n_samples = spec.get("n_samples", 50)

        def _sim_fn(c: ReactorConfiguration) -> dict[str, float]:
            elec = simulate_electrical(c, model_level=ModelLevel.LEVEL_2)
            thermal = simulate_thermal_lumped(c, elec["P_sample"])
            energy = compute_energy_accounting(c, elec, ModelLevel.LEVEL_2)
            return {
                "peak_current_A": max(elec["current"].values),
                "peak_temperature_K": thermal.peak_temperature_K,
                "delivered_energy_J": energy.sample_energy_J,
                "max_heating_rate_K_s": max(thermal.heating_rate_K_s.values),
            }

        uq = run_monte_carlo(cfg, _sim_fn, n_samples=n_samples)
        return {"status": "success", "uncertainty": uq.to_dict()}

    def _build_dashboard(self, spec: dict[str, Any]) -> dict[str, Any]:
        r = self._simulate_pulse(spec)
        return {"status": "success", "dashboard": r.get("dashboard")}

    def _ai_query(self, spec: dict[str, Any]) -> dict[str, Any]:
        query_type = spec.get("query_type", "dominant_uncertainty")
        if query_type == "dominant_uncertainty":
            uq = self._run_monte_carlo({"n_samples": 20})
            return self.ai_tools.what_assumption_dominates_uncertainty(uq["uncertainty"])
        if query_type == "measurement_recommendation":
            cfg = self._build_config(spec)
            uq = self._run_monte_carlo({"n_samples": 20})
            return self.ai_tools.which_measurement_would_improve_model(cfg, uq["uncertainty"])
        if query_type == "falsify_hypothesis":
            return self.ai_tools.evidence_to_falsify_hypothesis(spec.get("hypothesis", "isolated_Au_atoms"))
        if query_type == "characterization_methods":
            return self.ai_tools.characterization_for_isolated_vs_nanoparticle()
        return {"error": f"Unknown query_type: {query_type}"}

    def _params_to_config(self, params: dict[str, Any]) -> dict[str, Any]:
        mapping = {
            "sample_resistance_ohm": "sample_resistance_ohm",
            "contact_resistance_ohm": "electrode_contact_resistance_ohm",
            "chamber_pressure_Pa": "chamber_pressure_Pa",
            "ambient_temperature_K": "ambient_temperature_K",
        }
        cfg = {}
        for k, v in params.items():
            if k in mapping:
                cfg[mapping[k]] = v
            elif k == "atmosphere":
                cfg["atmosphere_type"] = str(v)
        return cfg

    def _extract_objectives(self, result: dict[str, Any]) -> dict[str, float]:
        sim = result.get("simulation_result", {})
        scores = sim.get("hypothesis_scores", {}) or {}
        current = sim.get("current", {})
        peak_I = max(current.get("values", [0]))
        return {
            "graphene_conversion_score": scores.get("graphene_conversion_score", 0),
            "au_single_atom_retention_score": scores.get("au_single_atom_retention_score", 0),
            "au_cluster_risk": scores.get("au_cluster_risk", 0),
            "au_loss_risk": scores.get("au_loss_risk", 0),
            "electrical_stress": peak_I / 1000.0,
            "thermal_stress": scores.get("carbon_damage_risk", 0),
            "confidence": (
                0.0 if result.get("sanity", {}).get("status") == "PHYSICALLY_INVALID" else 1.0
            ),
        }

    def get_status(self) -> dict[str, Any]:
        return {
            "lab": "fjh_reactor",
            "status": "operational",
            "mode": "simulation_only",
            "hardware_control_enabled": False,
            "model_version": MODEL_VERSION,
            "experiments_recorded": len(self.ledger.list_experiments(limit=1000)),
            "capabilities": [
                "simulate_pulse", "sanity_check", "doe_latin_hypercube",
                "doe_ofat", "compare_atmospheres", "compare_ultrasound",
                "monte_carlo", "dashboard", "ai_query",
            ],
        }

    def get_capabilities(self) -> dict[str, Any]:
        base = super().get_capabilities()
        base["simulation_only"] = True
        base["hardware_control"] = False
        return base
