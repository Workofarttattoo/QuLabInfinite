#!/usr/bin/env python3
"""
Confidence Scoring System for Materials Simulations

Provides confidence scores (0-100) for simulation results based on:
- Reference data availability
- Data quality and provenance
- Validation against multiple sources
- Material characterization completeness
- Property estimation uncertainty

Confidence Levels:
- 90-100: Excellent - Multiple experimental validations
- 80-89:  Very Good - Computational + some experimental validation
- 70-79:  Good - Well-validated computational predictions
- 60-69:  Acceptable - Single-source computational predictions
- 50-59:  Fair - Estimates with limited validation
- 40-49:  Poor - Rough estimates, high uncertainty
- 0-39:   Unreliable - Insufficient data or large errors
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging


class DataSource(Enum):
    """Data source types with quality scores"""
    EXPERIMENTAL_PEER_REVIEWED = 100  # Published experimental data
    EXPERIMENTAL_DATASHEET = 95       # Manufacturer datasheets
    NIST_DATABASE = 95                # NIST reference data
    MATERIALS_PROJECT_DFT = 80        # Materials Project DFT calculations
    COMPUTATIONAL_VALIDATED = 75      # Computational with experimental validation
    COMPUTATIONAL_ONLY = 60           # Computational without validation
    ESTIMATED_CORRELATION = 40        # Estimated from correlations
    ROUGH_ESTIMATE = 20               # Rough estimation


class PropertyType(Enum):
    """Property types with confidence modifiers"""
    FUNDAMENTAL = 1.0      # Well-defined fundamental properties (density, melting point)
    MECHANICAL = 0.9       # Mechanical properties (strength, modulus)
    THERMAL = 0.85         # Thermal properties (conductivity, heat capacity)
    ELECTRONIC = 0.8       # Electronic properties (band gap, conductivity)
    MAGNETIC = 0.7         # Magnetic properties (harder to predict)
    SURFACE = 0.6          # Surface properties (very condition-dependent)


@dataclass
class PropertyConfidence:
    """Confidence information for a single property"""
    property_name: str
    value: float
    units: str
    confidence_score: float  # 0-100
    data_source: DataSource
    property_type: PropertyType
    uncertainty_percent: float
    validation_count: int = 0  # Number of validation sources
    notes: str = ""

    def get_confidence_level(self) -> str:
        """Get confidence level as human-readable string"""
        if self.confidence_score >= 90:
            return "Excellent"
        elif self.confidence_score >= 80:
            return "Very Good"
        elif self.confidence_score >= 70:
            return "Good"
        elif self.confidence_score >= 60:
            return "Acceptable"
        elif self.confidence_score >= 50:
            return "Fair"
        elif self.confidence_score >= 40:
            return "Poor"
        else:
            return "Unreliable"


@dataclass
class MaterialConfidenceReport:
    """Complete confidence report for a material"""
    material_name: str
    material_id: Optional[str]
    property_confidences: List[PropertyConfidence]
    overall_confidence: float  # Weighted average
    data_completeness: float   # Percentage of properties with data
    validation_coverage: float # Percentage validated against external sources

    # Summary statistics
    excellent_properties: int = 0
    good_properties: int = 0
    poor_properties: int = 0

    def __post_init__(self):
        """Calculate summary statistics"""
        for prop in self.property_confidences:
            if prop.confidence_score >= 80:
                self.excellent_properties += 1
            elif prop.confidence_score >= 60:
                self.good_properties += 1
            else:
                self.poor_properties += 1


class ConfidenceScorer:
    """
    Calculates confidence scores for material properties and simulations

    Scoring Methodology:
    1. Base score from data source quality
    2. Adjustment for property type difficulty
    3. Bonus for multiple validation sources
    4. Penalty for high uncertainty
    5. Penalty for missing related properties
    """

    # Expected properties for complete characterization
    CORE_PROPERTIES = {
        "fundamental": ["density_g_cm3", "density_kg_m3"],
        "mechanical": ["youngs_modulus", "shear_modulus", "bulk_modulus", "poissons_ratio"],
        "thermal": ["thermal_conductivity", "specific_heat", "melting_point"],
        "electronic": ["band_gap_ev", "electrical_conductivity"],
    }

    def __init__(self):
        """Initialize confidence scorer"""
        self.logger = logging.getLogger(__name__)

    def score_property(
        self,
        property_name: str,
        value: float,
        data_source: DataSource,
        uncertainty_percent: float = 0.0,
        validation_sources: int = 0,
        has_related_properties: bool = True
    ) -> PropertyConfidence:
        """
        Calculate confidence score for a single property

        Args:
            property_name: Name of the property
            value: Property value
            data_source: Source of the data
            uncertainty_percent: Estimated uncertainty (%)
            validation_sources: Number of independent validation sources
            has_related_properties: Whether related properties are available

        Returns:
            PropertyConfidence object with score
        """
        # 1. Base score from data source (0-100)
        base_score = data_source.value

        # 2. Property type modifier
        property_type = self._infer_property_type(property_name)
        type_modifier = property_type.value

        # 3. Validation bonus (up to +15 points)
        validation_bonus = min(15.0, validation_sources * 5.0)

        # 4. Uncertainty penalty (0-30 points)
        # High uncertainty reduces confidence
        if uncertainty_percent > 0:
            uncertainty_penalty = min(30.0, uncertainty_percent * 0.5)
        else:
            uncertainty_penalty = 0.0

        # 5. Completeness bonus/penalty
        # Having related properties increases confidence
        completeness_adjustment = 5.0 if has_related_properties else -10.0

        # Calculate final score
        score = (base_score * type_modifier) + validation_bonus - uncertainty_penalty + completeness_adjustment
        score = max(0.0, min(100.0, score))  # Clamp to [0, 100]

        # Units
        units = self._get_units(property_name)

        # Notes
        notes = self._generate_notes(
            data_source=data_source,
            validation_sources=validation_sources,
            uncertainty_percent=uncertainty_percent,
            has_related_properties=has_related_properties
        )

        return PropertyConfidence(
            property_name=property_name,
            value=value,
            units=units,
            confidence_score=score,
            data_source=data_source,
            property_type=property_type,
            uncertainty_percent=uncertainty_percent,
            validation_count=validation_sources,
            notes=notes
        )

    def score_material(
        self,
        material_name: str,
        properties: Dict[str, float],
        data_sources: Dict[str, DataSource],
        uncertainties: Optional[Dict[str, float]] = None,
        validation_counts: Optional[Dict[str, int]] = None,
        material_id: Optional[str] = None
    ) -> MaterialConfidenceReport:
        """
        Calculate confidence scores for all properties of a material

        Args:
            material_name: Material name
            properties: Dictionary of property_name -> value
            data_sources: Dictionary of property_name -> DataSource
            uncertainties: Optional dictionary of property_name -> uncertainty_percent
            validation_counts: Optional dictionary of property_name -> validation_count
            material_id: Optional material identifier

        Returns:
            MaterialConfidenceReport with detailed scores
        """
        if uncertainties is None:
            uncertainties = {}
        if validation_counts is None:
            validation_counts = {}

        property_confidences = []

        # Score each property
        for prop_name, value in properties.items():
            if value == 0.0:
                continue  # Skip zero/missing values

            data_source = data_sources.get(prop_name, DataSource.ROUGH_ESTIMATE)
            uncertainty = uncertainties.get(prop_name, 20.0)  # Default 20% uncertainty
            validation_count = validation_counts.get(prop_name, 0)

            # Check for related properties
            has_related = self._has_related_properties(prop_name, properties)

            confidence = self.score_property(
                property_name=prop_name,
                value=value,
                data_source=data_source,
                uncertainty_percent=uncertainty,
                validation_sources=validation_count,
                has_related_properties=has_related
            )

            property_confidences.append(confidence)

        # Calculate overall metrics
        if property_confidences:
            # Weighted average (higher value properties get more weight)
            weights = [1.0] * len(property_confidences)  # Could weight by importance
            scores = [p.confidence_score for p in property_confidences]
            overall_confidence = np.average(scores, weights=weights)

            # Data completeness
            expected_count = sum(len(props) for props in self.CORE_PROPERTIES.values())
            actual_count = len(property_confidences)
            data_completeness = min(100.0, (actual_count / expected_count) * 100.0)

            # Validation coverage
            validated_count = sum(1 for p in property_confidences if p.validation_count > 0)
            validation_coverage = (validated_count / len(property_confidences)) * 100.0

        else:
            overall_confidence = 0.0
            data_completeness = 0.0
            validation_coverage = 0.0

        return MaterialConfidenceReport(
            material_name=material_name,
            material_id=material_id,
            property_confidences=property_confidences,
            overall_confidence=overall_confidence,
            data_completeness=data_completeness,
            validation_coverage=validation_coverage
        )

    def _infer_property_type(self, property_name: str) -> PropertyType:
        """Infer property type from name"""
        name_lower = property_name.lower()

        if any(x in name_lower for x in ["density", "volume", "mass", "molar"]):
            return PropertyType.FUNDAMENTAL
        elif any(x in name_lower for x in ["modulus", "strength", "hardness", "poisson"]):
            return PropertyType.MECHANICAL
        elif any(x in name_lower for x in ["thermal", "heat", "melting", "boiling", "temperature"]):
            return PropertyType.THERMAL
        elif any(x in name_lower for x in ["band_gap", "conductivity", "resistivity", "dielectric"]):
            return PropertyType.ELECTRONIC
        elif any(x in name_lower for x in ["magnetic", "permeability", "coercivity"]):
            return PropertyType.MAGNETIC
        elif any(x in name_lower for x in ["surface", "contact", "adhesion"]):
            return PropertyType.SURFACE
        else:
            return PropertyType.MECHANICAL  # Default

    def _has_related_properties(self, property_name: str, properties: Dict[str, float]) -> bool:
        """Check if related properties are available"""
        # Determine property group
        prop_type = self._infer_property_type(property_name)

        # Check if other properties in the same group exist
        if prop_type == PropertyType.MECHANICAL:
            related = ["youngs_modulus", "shear_modulus", "bulk_modulus", "poissons_ratio"]
            return any(prop in properties and properties[prop] > 0 for prop in related if prop != property_name)

        elif prop_type == PropertyType.THERMAL:
            related = ["thermal_conductivity", "specific_heat", "melting_point"]
            return any(prop in properties and properties[prop] > 0 for prop in related if prop != property_name)

        elif prop_type == PropertyType.ELECTRONIC:
            related = ["band_gap_ev", "electrical_conductivity"]
            return any(prop in properties and properties[prop] > 0 for prop in related if prop != property_name)

        return True  # Assume related properties exist for others

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
        }
        return units_map.get(property_name, "")

    def _generate_notes(
        self,
        data_source: DataSource,
        validation_sources: int,
        uncertainty_percent: float,
        has_related_properties: bool
    ) -> str:
        """Generate explanatory notes for confidence score"""
        notes = []

        # Data source note
        if data_source in [DataSource.EXPERIMENTAL_PEER_REVIEWED, DataSource.NIST_DATABASE]:
            notes.append("High-quality experimental data")
        elif data_source == DataSource.MATERIALS_PROJECT_DFT:
            notes.append("DFT computational prediction")
        elif data_source in [DataSource.ESTIMATED_CORRELATION, DataSource.ROUGH_ESTIMATE]:
            notes.append("Estimated value - use with caution")

        # Validation note
        if validation_sources > 0:
            notes.append(f"Validated against {validation_sources} source(s)")
        else:
            notes.append("No external validation")

        # Uncertainty note
        if uncertainty_percent > 30:
            notes.append(f"High uncertainty ({uncertainty_percent:.0f}%)")
        elif uncertainty_percent > 0:
            notes.append(f"Uncertainty: ±{uncertainty_percent:.0f}%")

        # Completeness note
        if not has_related_properties:
            notes.append("Limited related property data")

        return "; ".join(notes)

    def print_confidence_report(self, report: MaterialConfidenceReport):
        """Print formatted confidence report"""
        print("\n" + "=" * 80)
        print(f"CONFIDENCE REPORT: {report.material_name}")
        print("=" * 80)

        if report.material_id:
            print(f"\nMaterial ID: {report.material_id}")

        print(f"\nOVERALL CONFIDENCE: {report.overall_confidence:.1f}/100")
        print(f"Data Completeness: {report.data_completeness:.1f}%")
        print(f"Validation Coverage: {report.validation_coverage:.1f}%")

        print(f"\nProperty Quality Distribution:")
        print(f"  Excellent (≥80): {report.excellent_properties}")
        print(f"  Good (60-79):    {report.good_properties}")
        print(f"  Poor (<60):      {report.poor_properties}")

        if report.property_confidences:
            print(f"\nPROPERTY DETAILS ({len(report.property_confidences)} properties):")
            print("-" * 80)
            print(f"{'Property':<25} {'Value':<15} {'Confidence':<12} {'Level':<12} {'Source':<20}")
            print("-" * 80)

            # Sort by confidence (highest first)
            sorted_props = sorted(report.property_confidences, key=lambda x: x.confidence_score, reverse=True)

            for prop in sorted_props:
                value_str = f"{prop.value:.3g} {prop.units}"
                confidence_str = f"{prop.confidence_score:.1f}/100"
                level = prop.get_confidence_level()

                print(f"{prop.property_name:<25} {value_str:<15} {confidence_str:<12} {level:<12} {prop.data_source.name:<20}")

        print("=" * 80 + "\n")


if __name__ == "__main__":
    # Test confidence scoring
    logging.basicConfig(level=logging.INFO)

    print("Testing Confidence Scorer...")
    print("=" * 60)

    scorer = ConfidenceScorer()

    # Example: Score a material with mixed data quality
    material_properties = {
        "density_g_cm3": 2.33,  # Experimental data
        "youngs_modulus": 165.0,  # Well-validated
        "band_gap_ev": 1.12,  # Multiple sources
        "thermal_conductivity": 148.0,  # Computational
        "shear_modulus": 62.0,  # Estimated
    }

    data_sources = {
        "density_g_cm3": DataSource.EXPERIMENTAL_PEER_REVIEWED,
        "youngs_modulus": DataSource.COMPUTATIONAL_VALIDATED,
        "band_gap_ev": DataSource.MATERIALS_PROJECT_DFT,
        "thermal_conductivity": DataSource.COMPUTATIONAL_ONLY,
        "shear_modulus": DataSource.ESTIMATED_CORRELATION,
    }

    uncertainties = {
        "density_g_cm3": 1.0,  # ±1%
        "youngs_modulus": 10.0,  # ±10%
        "band_gap_ev": 5.0,  # ±5%
        "thermal_conductivity": 20.0,  # ±20%
        "shear_modulus": 30.0,  # ±30%
    }

    validation_counts = {
        "density_g_cm3": 3,  # Validated by 3 sources
        "youngs_modulus": 2,  # 2 sources
        "band_gap_ev": 2,  # 2 sources
        "thermal_conductivity": 1,  # 1 source
        "shear_modulus": 0,  # No validation
    }

    # Generate report
    report = scorer.score_material(
        material_name="Silicon",
        properties=material_properties,
        data_sources=data_sources,
        uncertainties=uncertainties,
        validation_counts=validation_counts,
        material_id="mp-149"
    )

    scorer.print_confidence_report(report)

    print("\n✓ Confidence scoring test completed!")
