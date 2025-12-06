#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

material_profiles - Part of Materials Lab
"""

"""
Synthetic material profile generation for QuLab Infinite.

This module does not claim to be an authoritative source of empirical data.
Instead it derives smooth, internally consistent property curves from the
tabulated values already present in the materials database so that downstream
experiments have something richer than single scalars to work with.

The generator focuses on:
  * Mechanical curves (stress-strain, modulus vs. temperature)
  * Thermal curves (thermal conductivity & specific heat vs. temperature)
  * Electrical curves (conductivity/resistivity vs. temperature)
  * Variability estimates (simple ± spreads derived from confidence scores)
  * Anisotropy factors (lightweight tensors for orthotropic materials)
  * Manufacturing metadata (process hints inferred from category)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from materials_database import MaterialProperties, MaterialsDatabase


@dataclass
class PropertyCurve:
    """Simple timeseries-like property curve."""

    abscissa: List[float]
    ordinate: List[float]
    unit: str
    label: str


class MaterialProfileGenerator:
    """Generate synthetic but internally consistent property profiles."""

    def __init__(self, database: MaterialsDatabase):
        self.database = database

    # ------------------------------------------------------------------ public

    def build_profile(self, material_name: str) -> Dict[str, object]:
        """Return mechanical/thermal/electrical curves & metadata."""
        props = self.database.get_material(material_name)
        if props is None:
            raise ValueError(f"Material '{material_name}' not found in database")

        mechanical = {name: self._curve_to_dict(curve) for name, curve in self._mechanical_profile(props).items()}
        thermal = {name: self._curve_to_dict(curve) for name, curve in self._thermal_profile(props).items()}
        electrical = {name: self._curve_to_dict(curve) for name, curve in self._electrical_profile(props).items()}

        profile = {
            "material": props.name,
            "category": props.category,
            "mechanical": mechanical,
            "thermal": thermal,
            "electrical": electrical,
            "variability": self._variability_summary(props),
            "anisotropy": self._anisotropy_tensor(props),
            "manufacturing_history": self._manufacturing_history(props),
        }
        safety = self.database.get_safety_data(props.name)
        if safety:
            profile["safety"] = safety
        return profile

    # --------------------------------------------------------------- mechanical

    def _mechanical_profile(self, props: MaterialProperties) -> Dict[str, PropertyCurve]:
        # Stress-strain (0 → elongation_at_break)
        elongation = max(props.elongation_at_break / 100.0, 0.01)
        strain_axis = np.linspace(0.0, elongation, 200)

        youngs_modulus_mpa = props.youngs_modulus * 1000.0
        yield_strength = max(props.yield_strength, youngs_modulus_mpa * 0.002)
        uts = max(props.tensile_strength, yield_strength * 1.05)

        elastic_limit = yield_strength / youngs_modulus_mpa
        stress = np.empty_like(strain_axis)
        elastic_mask = strain_axis <= elastic_limit
        stress[elastic_mask] = youngs_modulus_mpa * strain_axis[elastic_mask]
        plastic_mask = strain_axis > elastic_limit
        if np.any(plastic_mask):
            plastic_strain = (strain_axis[plastic_mask] - elastic_limit) / max(elongation - elastic_limit, 1e-6)
            hardening = yield_strength + (uts - yield_strength) * (1 - np.exp(-4 * plastic_strain))
            softening = uts * np.clip(1 - (strain_axis[plastic_mask] / max(elongation, 1e-6))**1.5, 0.0, 1.0)
            stress[plastic_mask] = np.maximum(hardening, softening)

        temperature_axis = np.linspace(200.0, 1000.0, 64)
        modulus_temp = self._temperature_dependent_property(
            base=youngs_modulus_mpa,
            temperature_axis=temperature_axis,
            softening_rate=self._category_softening_factor(props.category),
        )

        return {
            "stress_strain": PropertyCurve(
                abscissa=strain_axis.tolist(),
                ordinate=stress.tolist(),
                unit="MPa",
                label="Stress vs. Strain",
            ),
            "modulus_temperature": PropertyCurve(
                abscissa=temperature_axis.tolist(),
                ordinate=modulus_temp.tolist(),
                unit="MPa",
                label="Young's modulus vs. temperature",
            ),
        }

    # ------------------------------------------------------------------ thermal

    def _thermal_profile(self, props: MaterialProperties) -> Dict[str, PropertyCurve]:
        temperature_axis = np.linspace(200.0, 1200.0, 64)
        conductivity_base = max(props.thermal_conductivity, 0.1)
        specific_heat_base = max(props.specific_heat, 100.0)

        conductivity = self._temperature_dependent_property(
            base=conductivity_base,
            temperature_axis=temperature_axis,
            softening_rate=0.0003 if props.category == "metal" else 0.0006,
            floor_ratio=0.25,
        )

        heat_capacity = self._temperature_dependent_property(
            base=specific_heat_base,
            temperature_axis=temperature_axis,
            softening_rate=-0.0002,
            floor_ratio=1.4,
        )

        expansion_axis = np.linspace(293.15, 973.15, 32)
        expansion_coeff = self._temperature_dependent_property(
            base=max(props.thermal_expansion, 1e-6),
            temperature_axis=expansion_axis,
            softening_rate=-0.00015,
            floor_ratio=1.8,
        )

        return {
            "conductivity_temperature": PropertyCurve(
                abscissa=temperature_axis.tolist(),
                ordinate=conductivity.tolist(),
                unit="W/(m·K)",
                label="Thermal conductivity vs. temperature",
            ),
            "heat_capacity_temperature": PropertyCurve(
                abscissa=temperature_axis.tolist(),
                ordinate=heat_capacity.tolist(),
                unit="J/(kg·K)",
                label="Specific heat vs. temperature",
            ),
            "expansion_temperature": PropertyCurve(
                abscissa=expansion_axis.tolist(),
                ordinate=expansion_coeff.tolist(),
                unit="1/K",
                label="Linear expansion coefficient vs. temperature",
            ),
        }

    # --------------------------------------------------------------- electrical

    def _electrical_profile(self, props: MaterialProperties) -> Dict[str, PropertyCurve]:
        base_cond = max(props.electrical_conductivity, 1e3 if props.category == "metal" else 1e-6)
        base_res = 1.0 / base_cond if base_cond > 0 else props.electrical_resistivity
        temperature_axis = np.linspace(200.0, 800.0, 64)
        temperature_coeff = self._electrical_temp_coeff(props.category)

        conductivity = base_cond / (1.0 + temperature_coeff * (temperature_axis - 293.15))
        resistivity = base_res * (1.0 + temperature_coeff * (temperature_axis - 293.15))

        return {
            "conductivity_temperature": PropertyCurve(
                abscissa=temperature_axis.tolist(),
                ordinate=np.clip(conductivity, 1e-8, None).tolist(),
                unit="S/m",
                label="Electrical conductivity vs. temperature",
            ),
            "resistivity_temperature": PropertyCurve(
                abscissa=temperature_axis.tolist(),
                ordinate=np.clip(resistivity, 1e-12, None).tolist(),
                unit="Ω·m",
                label="Electrical resistivity vs. temperature",
            ),
        }

    # ----------------------------------------------------------- helper methods

    def _temperature_dependent_property(
        self,
        base: float,
        temperature_axis: np.ndarray,
        softening_rate: float,
        floor_ratio: float = 0.3,
    ) -> np.ndarray:
        gradient = softening_rate * (temperature_axis - 293.15)
        factor = np.clip(1.0 - gradient, floor_ratio, None)
        return base * factor

    def _category_softening_factor(self, category: str) -> float:
        if category == "metal":
            return 0.00025
        if category == "ceramic":
            return 0.00005
        if category == "composite":
            return 0.00035
        if category == "polymer":
            return 0.0006
        return 0.0003

    def _electrical_temp_coeff(self, category: str) -> float:
        if category == "metal":
            return 0.0038
        if category == "semiconductor":
            return -0.0025
        if category == "polymer":
            return -0.01
        if category == "composite":
            return 0.0015
        return 0.0005

    def _variability_summary(self, props: MaterialProperties) -> Dict[str, Dict[str, float]]:
        spread = max(1.0 - props.confidence, 0.05)
        variability = {
            "yield_strength": self._mean_std(props.yield_strength, spread),
            "ultimate_strength": self._mean_std(props.tensile_strength, spread),
            "youngs_modulus": self._mean_std(props.youngs_modulus * 1000.0, spread),
            "thermal_conductivity": self._mean_std(props.thermal_conductivity, spread),
            "specific_heat": self._mean_std(props.specific_heat, spread),
            "electrical_conductivity": self._mean_std(props.electrical_conductivity, spread),
        }
        return variability

    def _mean_std(self, mean_value: float, spread: float) -> Dict[str, float]:
        std = abs(mean_value) * spread
        return {"mean": float(mean_value), "std": float(std)}

    def _anisotropy_tensor(self, props: MaterialProperties) -> Dict[str, List[List[float]]]:
        if props.category in {"metal", "ceramic"}:
            factor = 1.0 + max(0.02, props.confidence * 0.05)
            tensor = np.diag([factor, factor, factor / 1.02])
        elif props.category == "composite":
            factor = 1.0 + max(0.3, (1.0 - props.confidence) * 0.6)
            tensor = np.diag([factor, factor * 0.6, factor * 0.4])
        elif props.category == "polymer":
            factor = 1.0 + max(0.1, (1.0 - props.confidence) * 0.4)
            tensor = np.diag([factor, factor * 0.8, factor * 0.5])
        else:
            tensor = np.diag([1.0, 1.0, 1.0])
        return {"coordinate_system": "principal_material_axes", "tensor": tensor.tolist()}

    def _manufacturing_history(self, props: MaterialProperties) -> Dict[str, object]:
        history = []
        if props.category == "metal":
            history.extend(["rolled", "heat_treated", "annealed"])
        elif props.category == "composite":
            history.extend(["layup_cured", "autoclave_cured"])
        elif props.category == "polymer":
            history.extend(["injection_molded"])
        elif props.category == "ceramic":
            history.extend(["sintered"])
        else:
            history.append("as_fabricated")

        if props.notes:
            history.append("notes_attached")

        return {
            "typical_processes": history,
            "data_source": props.data_source,
            "confidence": props.confidence,
        }

    # ------------------------------------------------------------ serialization

    def _curve_to_dict(self, curve: PropertyCurve) -> Dict[str, object]:
        return {
            "abscissa": curve.abscissa,
            "ordinate": curve.ordinate,
            "unit": curve.unit,
            "label": curve.label,
        }


def generate_profile(database: MaterialsDatabase, material_name: str) -> Dict[str, object]:
    """Convenience wrapper."""
    generator = MaterialProfileGenerator(database)
    return generator.build_profile(material_name)
