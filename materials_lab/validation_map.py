"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

validation_map - Part of Materials Lab
"""

"""
Mapping between material properties and validation reference keys.

Centralising this metadata lets MaterialsLab wire simulation outputs into the
shared validation framework without duplicating constants across tests, CLIs,
or database tooling.
"""

from __future__ import annotations

from typing import Dict, Mapping

# Each entry maps a material name to the properties we can validate.  For a
# given property we record the ResultsValidator reference key along with the
# attribute name on MaterialProperties (defaults to the property key).
MaterialValidationMap = Mapping[str, Mapping[str, Dict[str, str]]]


MATERIAL_PROPERTY_REFERENCE_MAP: MaterialValidationMap = {
    "SS 304": {
        "density": {"reference_key": "steel_304_density"},
        "youngs_modulus": {"reference_key": "steel_304_youngs_modulus"},
        "yield_strength": {"reference_key": "steel_304_yield_strength"},
        "thermal_conductivity": {"reference_key": "steel_304_thermal_conductivity"},
    },
    "Al 6061-T6": {
        "density": {"reference_key": "aluminum_6061_density"},
        "youngs_modulus": {"reference_key": "aluminum_6061_youngs_modulus"},
        "yield_strength": {"reference_key": "aluminum_6061_yield_strength"},
        "thermal_conductivity": {"reference_key": "aluminum_6061_thermal_conductivity"},
    },
    "Ti-6Al-4V": {
        "density": {"reference_key": "ti_6al4v_density"},
        "youngs_modulus": {"reference_key": "ti_6al4v_youngs_modulus"},
        "yield_strength": {"reference_key": "ti_6al4v_yield_strength"},
        "thermal_conductivity": {"reference_key": "ti_6al4v_thermal_conductivity"},
    },
}

