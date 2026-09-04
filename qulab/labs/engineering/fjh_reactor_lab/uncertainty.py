"""
Uncertainty propagation engine for FJH reactor simulations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import ReactorConfiguration
from .types import is_unknown


@dataclass
class UncertainParameter:
    """Parameter with uncertainty distribution."""

    name: str
    nominal: float
    std_fraction: float = 0.1  # relative std dev
    unit: str = ""


@dataclass
class UncertaintyResult:
    """Monte Carlo uncertainty propagation output."""

    n_samples: int
    peak_current_A: dict[str, float]
    peak_temperature_K: dict[str, float]
    delivered_energy_J: dict[str, float]
    max_heating_rate_K_s: dict[str, float]
    sensitivity: dict[str, float]
    dominant_uncertain_parameters: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "peak_current_A": self.peak_current_A,
            "peak_temperature_K": self.peak_temperature_K,
            "delivered_energy_J": self.delivered_energy_J,
            "max_heating_rate_K_s": self.max_heating_rate_K_s,
            "sensitivity": self.sensitivity,
            "dominant_uncertain_parameters": self.dominant_uncertain_parameters,
        }


def default_uncertain_parameters(config: ReactorConfiguration) -> list[UncertainParameter]:
    """Default uncertain parameters for FJH reactor."""
    return [
        UncertainParameter("sample_resistance_ohm", config.effective_sample_resistance_ohm(), 0.2, "ohm"),
        UncertainParameter("contact_resistance_ohm", 0.01, 0.5, "ohm"),
        UncertainParameter("ESR_ohm", config.effective_ESR_ohm(), 0.15, "ohm"),
        UncertainParameter("specific_heat_J_kg_K", 710.0, 0.3, "J/kg/K"),
        UncertainParameter(
            "sample_mass_g",
            float(config.sample_mass_g) if not is_unknown(config.sample_mass_g) else 1.0,
            0.05 if not is_unknown(config.sample_mass_g) else 0.2,
            "g",
        ),
        UncertainParameter("precursor_uniformity", 0.5, 0.3, "dimensionless"),
        UncertainParameter("residual_oxygen_fraction", 1e-4, 1.0, "fraction"),
        UncertainParameter("thermal_contact", 0.85, 0.1, "dimensionless"),
    ]


def run_monte_carlo(
    config: ReactorConfiguration,
    simulate_fn: Callable[[ReactorConfiguration], dict[str, float]],
    uncertain_params: list[UncertainParameter] | None = None,
    n_samples: int = 100,
    seed: int = 42,
) -> UncertaintyResult:
    """
    Monte Carlo uncertainty propagation.

    simulate_fn: takes modified config, returns dict with keys
      peak_current_A, peak_temperature_K, delivered_energy_J, max_heating_rate_K_s
    """
    rng = np.random.default_rng(seed)
    params = uncertain_params or default_uncertain_parameters(config)

    outputs = {
        "peak_current_A": [],
        "peak_temperature_K": [],
        "delivered_energy_J": [],
        "max_heating_rate_K_s": [],
    }
    param_samples: dict[str, list[float]] = {p.name: [] for p in params}

    for _ in range(n_samples):
        cfg = _sample_config(config, params, rng)
        for p in params:
            val = getattr(cfg, p.name, None)
            if val is None:
                if p.name == "sample_resistance_ohm":
                    val = cfg.effective_sample_resistance_ohm()
                elif p.name == "ESR_ohm":
                    val = cfg.effective_ESR_ohm()
                else:
                    val = p.nominal * (1 + rng.normal(0, p.std_fraction))
            param_samples[p.name].append(float(val) if val is not None else p.nominal)

        try:
            result = simulate_fn(cfg)
            for key in outputs:
                outputs[key].append(result.get(key, 0.0))
        except Exception:
            for key in outputs:
                outputs[key].append(float("nan"))

    def _stats(vals: list[float]) -> dict[str, float]:
        arr = np.array(vals)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return {"mean": 0, "std": 0, "p5": 0, "p95": 0}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p5": float(np.percentile(arr, 5)),
            "p95": float(np.percentile(arr, 95)),
        }

    sensitivity: dict[str, float] = {}
    for p in params:
        samples = np.array(param_samples[p.name])
        out_arr = np.array(outputs["peak_temperature_K"])
        mask = ~np.isnan(out_arr)
        if np.sum(mask) > 5 and np.std(samples[mask]) > 0:
            corr = float(np.corrcoef(samples[mask], out_arr[mask])[0, 1])
            sensitivity[p.name] = abs(corr)
        else:
            sensitivity[p.name] = 0.0

    dominant = sorted(sensitivity, key=sensitivity.get, reverse=True)[:3]

    return UncertaintyResult(
        n_samples=n_samples,
        peak_current_A=_stats(outputs["peak_current_A"]),
        peak_temperature_K=_stats(outputs["peak_temperature_K"]),
        delivered_energy_J=_stats(outputs["delivered_energy_J"]),
        max_heating_rate_K_s=_stats(outputs["max_heating_rate_K_s"]),
        sensitivity=sensitivity,
        dominant_uncertain_parameters=dominant,
    )


def _sample_config(
    config: ReactorConfiguration,
    params: list[UncertainParameter],
    rng: np.random.Generator,
) -> ReactorConfiguration:
    """Create config copy with sampled uncertain parameters."""
    import copy
    cfg = copy.deepcopy(config)
    for p in params:
        factor = 1 + rng.normal(0, p.std_fraction)
        if p.name == "sample_resistance_ohm":
            cfg.sample_resistance_ohm = p.nominal * factor
        elif p.name == "contact_resistance_ohm":
            cfg.electrode_contact_resistance_ohm = p.nominal * factor
        elif p.name == "ESR_ohm":
            cfg.measured_ESR_each_ohm = (p.nominal * factor) * cfg.capacitor_count
        elif p.name == "sample_mass_g":
            cfg.sample_mass_g = p.nominal * factor
    return cfg
