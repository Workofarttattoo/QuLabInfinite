#!/usr/bin/env python3
"""
Materials Validation Module
Compares simulation results against reference data from:
- Materials Project (computational predictions)
- Experimental databases (NIST, literature)
- QuLabInfinite internal database

Provides confidence scoring and accuracy metrics
"""

import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from pathlib import Path
import logging

try:
    from .materials_database import MaterialProperties
    from .materials_project_client import MaterialsProjectClient, MPMaterialData
except ImportError:
    from materials_database import MaterialProperties
    from materials_project_client import MaterialsProjectClient, MPMaterialData


class ValidationStatus(Enum):
    """Validation result status"""
    EXCELLENT = "excellent"      # <5% error
    GOOD = "good"                # 5-15% error
    ACCEPTABLE = "acceptable"    # 15-30% error
    POOR = "poor"                # 30-50% error
    FAILED = "failed"            # >50% error
    UNKNOWN = "unknown"          # No reference data


@dataclass
class PropertyComparison:
    """Comparison of a single property"""
    property_name: str
    simulated_value: float
    reference_value: float
    reference_source: str
    absolute_error: float
    relative_error_percent: float
    status: ValidationStatus
    units: str


@dataclass
class MaterialValidation:
    """Complete validation report for a material"""
    material_name: str
    material_id: Optional[str]
    simulated_properties: Dict[str, float]
    reference_source: str
    comparisons: List[PropertyComparison]
    overall_status: ValidationStatus
    confidence_score: float  # 0-100
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "material_name": self.material_name,
            "material_id": self.material_id,
            "simulated_properties": self.simulated_properties,
            "reference_source": self.reference_source,
            "comparisons": [asdict(c) for c in self.comparisons],
            "overall_status": self.overall_status.value,
            "confidence_score": self.confidence_score,
            "summary": self.summary
        }


class MaterialsValidator:
    """
    Validates simulation results against reference data

    Features:
    - Multi-source validation (Materials Project, experimental)
    - Property-specific tolerances
    - Confidence scoring
    - Detailed comparison reports
    """

    # Property-specific tolerances (relative error %)
    PROPERTY_TOLERANCES = {
        # Mechanical properties (typically 10-20% variation)
        "density": 5.0,
        "density_g_cm3": 5.0,
        "youngs_modulus": 15.0,
        "shear_modulus": 20.0,
        "bulk_modulus": 20.0,
        "tensile_strength": 25.0,
        "compressive_strength": 25.0,

        # Thermal properties (typically 10-30% variation)
        "thermal_conductivity": 30.0,
        "specific_heat": 20.0,
        "melting_point": 5.0,

        # Electronic properties (can vary widely)
        "band_gap_ev": 10.0,
        "electrical_conductivity": 50.0,
    }

    def __init__(self, mp_client: Optional[MaterialsProjectClient] = None):
        """
        Initialize validator

        Args:
            mp_client: Materials Project client for reference data
        """
        self.mp_client = mp_client
        self.logger = logging.getLogger(__name__)

    def _calculate_status(self, relative_error: float, property_name: str) -> ValidationStatus:
        """Determine validation status from relative error"""
        tolerance = self.PROPERTY_TOLERANCES.get(property_name, 20.0)

        if relative_error < tolerance * 0.25:
            return ValidationStatus.EXCELLENT
        elif relative_error < tolerance * 0.75:
            return ValidationStatus.GOOD
        elif relative_error < tolerance * 1.5:
            return ValidationStatus.ACCEPTABLE
        elif relative_error < tolerance * 2.5:
            return ValidationStatus.POOR
        else:
            return ValidationStatus.FAILED

    def compare_properties(
        self,
        simulated: MaterialProperties,
        reference: MaterialProperties,
        properties_to_check: Optional[List[str]] = None
    ) -> List[PropertyComparison]:
        """
        Compare simulated properties against reference

        Args:
            simulated: Simulated material properties
            reference: Reference material properties
            properties_to_check: List of property names to compare. If None, checks all common properties

        Returns:
            List of PropertyComparison objects
        """
        if properties_to_check is None:
            # Default properties to check
            properties_to_check = [
                "density_g_cm3", "youngs_modulus", "shear_modulus",
                "bulk_modulus", "thermal_conductivity", "band_gap_ev"
            ]

        comparisons = []

        for prop_name in properties_to_check:
            sim_value = getattr(simulated, prop_name, 0.0)
            ref_value = getattr(reference, prop_name, 0.0)

            # Skip if either value is missing or zero
            if sim_value == 0.0 or ref_value == 0.0:
                continue

            # Calculate errors
            abs_error = abs(sim_value - ref_value)
            rel_error = (abs_error / abs(ref_value)) * 100.0

            # Determine status
            status = self._calculate_status(rel_error, prop_name)

            # Determine units
            units = self._get_units(prop_name)

            comparison = PropertyComparison(
                property_name=prop_name,
                simulated_value=sim_value,
                reference_value=ref_value,
                reference_source="QuLabInfinite Database",
                absolute_error=abs_error,
                relative_error_percent=rel_error,
                status=status,
                units=units
            )
            comparisons.append(comparison)

        return comparisons

    def validate_against_mp(
        self,
        simulated: MaterialProperties,
        mp_id: str,
        properties_to_check: Optional[List[str]] = None
    ) -> MaterialValidation:
        """
        Validate simulation against Materials Project data

        Args:
            simulated: Simulated material properties
            mp_id: Materials Project ID
            properties_to_check: Properties to validate

        Returns:
            MaterialValidation report
        """
        if not self.mp_client:
            raise ValueError("Materials Project client not initialized")

        # Fetch MP data
        mp_data = self.mp_client.get_material(mp_id)
        if not mp_data:
            return MaterialValidation(
                material_name=simulated.name,
                material_id=mp_id,
                simulated_properties={},
                reference_source="Materials Project",
                comparisons=[],
                overall_status=ValidationStatus.UNKNOWN,
                confidence_score=0.0,
                summary=f"Material {mp_id} not found in Materials Project"
            )

        # Convert MP data to MaterialProperties
        mp_props = mp_data.to_material_properties()

        # Compare properties
        comparisons = self.compare_properties(simulated, mp_props, properties_to_check)

        # Calculate overall metrics
        if comparisons:
            avg_error = np.mean([c.relative_error_percent for c in comparisons])
            max_error = max([c.relative_error_percent for c in comparisons])

            # Determine overall status
            if avg_error < 5.0:
                overall_status = ValidationStatus.EXCELLENT
            elif avg_error < 15.0:
                overall_status = ValidationStatus.GOOD
            elif avg_error < 30.0:
                overall_status = ValidationStatus.ACCEPTABLE
            elif avg_error < 50.0:
                overall_status = ValidationStatus.POOR
            else:
                overall_status = ValidationStatus.FAILED

            # Confidence score (100 = perfect match, 0 = complete mismatch)
            confidence = max(0.0, 100.0 - avg_error * 1.5)

            summary = (
                f"Validated {len(comparisons)} properties against Materials Project ({mp_id}). "
                f"Average error: {avg_error:.1f}%, Max error: {max_error:.1f}%. "
                f"Status: {overall_status.value.upper()}"
            )

        else:
            overall_status = ValidationStatus.UNKNOWN
            confidence = 0.0
            summary = "No comparable properties found"

        # Extract simulated values
        sim_values = {c.property_name: c.simulated_value for c in comparisons}

        return MaterialValidation(
            material_name=simulated.name,
            material_id=mp_id,
            simulated_properties=sim_values,
            reference_source=f"Materials Project ({mp_id})",
            comparisons=comparisons,
            overall_status=overall_status,
            confidence_score=confidence,
            summary=summary
        )

    def validate_aerogel(self, simulated_results: Dict[str, float]) -> MaterialValidation:
        """
        Validate aerogel simulation results against known Airloy X103 properties

        Args:
            simulated_results: Dictionary of simulated property values

        Returns:
            MaterialValidation report
        """
        # Known Airloy X103 properties (from manufacturer and tests)
        reference_data = {
            "density_kg_m3": 144.0,  # kg/m³
            "thermal_conductivity": 0.014,  # W/(m·K) = 14 mW/(m·K)
            "tensile_strength": 0.31,  # MPa
            "compressive_strength": 1.65,  # MPa
            "min_service_temp": 73.0,  # K (-200°C)
        }

        comparisons = []

        for prop_name, ref_value in reference_data.items():
            if prop_name not in simulated_results:
                continue

            sim_value = simulated_results[prop_name]

            # Calculate errors
            abs_error = abs(sim_value - ref_value)
            rel_error = (abs_error / abs(ref_value)) * 100.0 if ref_value != 0 else 0.0

            # Determine status
            status = self._calculate_status(rel_error, prop_name)

            units = self._get_units(prop_name)

            comparison = PropertyComparison(
                property_name=prop_name,
                simulated_value=sim_value,
                reference_value=ref_value,
                reference_source="Airloy X103 Datasheet + QuLabInfinite Tests",
                absolute_error=abs_error,
                relative_error_percent=rel_error,
                status=status,
                units=units
            )
            comparisons.append(comparison)

        # Calculate overall metrics
        if comparisons:
            avg_error = np.mean([c.relative_error_percent for c in comparisons])
            max_error = max([c.relative_error_percent for c in comparisons])

            if avg_error < 5.0:
                overall_status = ValidationStatus.EXCELLENT
            elif avg_error < 15.0:
                overall_status = ValidationStatus.GOOD
            elif avg_error < 30.0:
                overall_status = ValidationStatus.ACCEPTABLE
            else:
                overall_status = ValidationStatus.POOR

            confidence = max(0.0, 100.0 - avg_error * 1.5)

            summary = (
                f"Aerogel simulation validated against Airloy X103. "
                f"{len(comparisons)} properties checked. "
                f"Average error: {avg_error:.1f}%, Max error: {max_error:.1f}%. "
                f"Status: {overall_status.value.upper()}"
            )
        else:
            overall_status = ValidationStatus.UNKNOWN
            confidence = 0.0
            summary = "No reference data available for comparison"

        return MaterialValidation(
            material_name="Airloy X103 Aerogel",
            material_id="airloy-x103",
            simulated_properties=simulated_results,
            reference_source="Airloy X103 Datasheet + Experimental Tests",
            comparisons=comparisons,
            overall_status=overall_status,
            confidence_score=confidence,
            summary=summary
        )

    def _get_units(self, property_name: str) -> str:
        """Get units for a property"""
        units_map = {
            "density": "kg/m³",
            "density_g_cm3": "g/cm³",
            "density_kg_m3": "kg/m³",
            "youngs_modulus": "GPa",
            "shear_modulus": "GPa",
            "bulk_modulus": "GPa",
            "tensile_strength": "MPa",
            "compressive_strength": "MPa",
            "yield_strength": "MPa",
            "thermal_conductivity": "W/(m·K)",
            "specific_heat": "J/(kg·K)",
            "melting_point": "K",
            "band_gap_ev": "eV",
            "electrical_conductivity": "S/m",
            "min_service_temp": "K",
        }
        return units_map.get(property_name, "")

    def print_validation_report(self, validation: MaterialValidation):
        """Print a formatted validation report"""
        print("\n" + "=" * 80)
        print(f"VALIDATION REPORT: {validation.material_name}")
        print("=" * 80)
        print(f"\nMaterial ID: {validation.material_id}")
        print(f"Reference Source: {validation.reference_source}")
        print(f"Overall Status: {validation.overall_status.value.upper()}")
        print(f"Confidence Score: {validation.confidence_score:.1f}/100")
        print(f"\n{validation.summary}")

        if validation.comparisons:
            print(f"\nDETAILED COMPARISON ({len(validation.comparisons)} properties):")
            print("-" * 80)
            print(f"{'Property':<25} {'Simulated':<15} {'Reference':<15} {'Error':<12} {'Status':<12}")
            print("-" * 80)

            for comp in validation.comparisons:
                status_symbol = {
                    ValidationStatus.EXCELLENT: "✓✓",
                    ValidationStatus.GOOD: "✓",
                    ValidationStatus.ACCEPTABLE: "~",
                    ValidationStatus.POOR: "⚠",
                    ValidationStatus.FAILED: "✗",
                }.get(comp.status, "?")

                print(
                    f"{comp.property_name:<25} "
                    f"{comp.simulated_value:<15.4g} "
                    f"{comp.reference_value:<15.4g} "
                    f"{comp.relative_error_percent:<11.1f}% "
                    f"{status_symbol} {comp.status.value:<10}"
                )

        print("=" * 80 + "\n")

    def save_validation_report(self, validation: MaterialValidation, output_path: str):
        """Save validation report to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(validation.to_dict(), f, indent=2, default=str)

        self.logger.info(f"Validation report saved to {output_path}")


if __name__ == "__main__":
    # Test validation
    logging.basicConfig(level=logging.INFO)

    print("Testing Materials Validator...")
    print("=" * 60)

    validator = MaterialsValidator()

    # Test aerogel validation
    print("\n1. Testing Aerogel Validation...")

    # Simulated results (slightly off to show validation)
    simulated_aerogel = {
        "density_kg_m3": 150.0,  # Ref: 144.0 (4.2% error)
        "thermal_conductivity": 0.015,  # Ref: 0.014 (7.1% error)
        "tensile_strength": 0.33,  # Ref: 0.31 (6.5% error)
        "compressive_strength": 1.70,  # Ref: 1.65 (3.0% error)
    }

    validation = validator.validate_aerogel(simulated_aerogel)
    validator.print_validation_report(validation)

    print("\n✓ Validation test completed!")
