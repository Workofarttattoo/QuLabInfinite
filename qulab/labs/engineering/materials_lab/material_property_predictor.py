#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Material Property Predictor - ML-based property prediction with uncertainty
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from materials_database import MaterialsDatabase, MaterialProperties


@dataclass
class PredictionResult:
    """Property prediction result with uncertainty"""
    property_name: str
    predicted_value: float
    uncertainty: float  # Standard deviation
    confidence: float  # 0-1 scale
    method: str
    notes: str = ""


class MaterialPropertyPredictor:
    """ML-based material property prediction"""

    def __init__(self, database: MaterialsDatabase):
        self.db = database
        self.trained = False

    def predict_from_composition(self,
                                 composition: Dict[str, float],
                                 properties: List[str]) -> List[PredictionResult]:
        """
        Predict properties from chemical composition

        Args:
            composition: Dict of element: weight_fraction
            properties: List of property names to predict

        Returns:
            List of prediction results
        """
        results = []

        # Normalize composition
        total = sum(composition.values())
        if total > 0:
            composition = {k: v/total for k, v in composition.items()}

        for prop in properties:
            # Use rule of mixtures with known materials
            prediction = self._predict_by_mixing_rule(composition, prop)
            results.append(prediction)

        return results

    def _predict_by_mixing_rule(self,
                                composition: Dict[str, float],
                                property_name: str) -> PredictionResult:
        """Predict property using rule of mixtures"""
        # Find materials containing these elements
        relevant_materials = []

        for mat in self.db.materials.values():
            # Simple heuristic: check if material name contains element
            for element in composition.keys():
                if element in mat.name:
                    relevant_materials.append(mat)
                    break

        if not relevant_materials:
            # No relevant materials, return uncertain prediction
            return PredictionResult(
                property_name=property_name,
                predicted_value=0.0,
                uncertainty=999.0,
                confidence=0.0,
                method="no_data",
                notes="No relevant materials in database"
            )

        # Get property values
        values = []
        for mat in relevant_materials:
            val = getattr(mat, property_name, None)
            if val is not None and val > 0:
                values.append(val)

        if not values:
            return PredictionResult(
                property_name=property_name,
                predicted_value=0.0,
                uncertainty=999.0,
                confidence=0.0,
                method="no_data",
                notes="Property not available"
            )

        # Simple averaging (could be weighted by composition similarity)
        mean_val = np.mean(values)
        std_val = np.std(values) if len(values) > 1 else mean_val * 0.2
        confidence = 1.0 / (1.0 + std_val / mean_val)  # Lower relative std = higher confidence

        return PredictionResult(
            property_name=property_name,
            predicted_value=mean_val,
            uncertainty=std_val,
            confidence=confidence,
            method="rule_of_mixtures",
            notes=f"Based on {len(relevant_materials)} similar materials"
        )

    def predict_from_structure(self,
                              crystal_structure: str,
                              bonding_type: str,
                              properties: List[str]) -> List[PredictionResult]:
        """
        Predict properties from crystal structure

        Args:
            crystal_structure: FCC, BCC, HCP, amorphous, etc.
            bonding_type: metallic, covalent, ionic, etc.
            properties: Properties to predict
        """
        results = []

        # Structure-property relationships
        structure_props = {
            ("FCC", "metallic"): {
                "youngs_modulus": (100, 30),  # (mean GPa, std GPa)
                "density": (8000, 2000),
                "thermal_conductivity": (80, 30)
            },
            ("BCC", "metallic"): {
                "youngs_modulus": (200, 50),
                "density": (7800, 1000),
                "thermal_conductivity": (50, 20)
            },
            ("HCP", "metallic"): {
                "youngs_modulus": (120, 40),
                "density": (6000, 2000),
                "thermal_conductivity": (60, 25)
            }
        }

        key = (crystal_structure, bonding_type)
        if key not in structure_props:
            key = ("FCC", "metallic")  # Default

        props = structure_props[key]

        for prop in properties:
            if prop in props:
                mean, std = props[prop]
                confidence = 0.7  # Moderate confidence from structure alone
            else:
                mean, std = 0.0, 999.0
                confidence = 0.0

            results.append(PredictionResult(
                property_name=prop,
                predicted_value=mean,
                uncertainty=std,
                confidence=confidence,
                method="structure_property_relationship",
                notes=f"From {crystal_structure} {bonding_type} structure"
            ))

        return results

    def predict_by_similarity(self,
                            reference_material: MaterialProperties,
                            property_name: str) -> PredictionResult:
        """Predict property by finding similar materials"""
        # Find similar materials (by category and density)
        similar = []

        for mat in self.db.materials.values():
            if mat.category != reference_material.category:
                continue

            # Check density similarity (within 20%)
            if reference_material.density > 0:
                density_ratio = mat.density / reference_material.density
                if not (0.8 < density_ratio < 1.2):
                    continue

            prop_val = getattr(mat, property_name, None)
            if prop_val is not None and prop_val > 0:
                similar.append(prop_val)

        if not similar:
            return PredictionResult(
                property_name=property_name,
                predicted_value=0.0,
                uncertainty=999.0,
                confidence=0.0,
                method="similarity",
                notes="No similar materials found"
            )

        mean_val = np.mean(similar)
        std_val = np.std(similar) if len(similar) > 1 else mean_val * 0.3
        confidence = min(0.9, len(similar) / 10.0)  # More similar materials = higher confidence

        return PredictionResult(
            property_name=property_name,
            predicted_value=mean_val,
            uncertainty=std_val,
            confidence=confidence,
            method="similarity",
            notes=f"Based on {len(similar)} similar materials"
        )

    def uncertainty_propagation(self,
                               predictions: List[PredictionResult],
                               formula: str) -> PredictionResult:
        """
        Propagate uncertainty through a formula

        Args:
            predictions: List of input predictions with uncertainties
            formula: Formula string (e.g., "A / B" for specific strength)
        """
        # Simple case: ratio of two properties
        if "/" in formula:
            if len(predictions) != 2:
                raise ValueError("Division requires exactly 2 predictions")

            A, B = predictions
            # f = A / B
            # σ_f = f * sqrt((σ_A/A)^2 + (σ_B/B)^2)
            result = A.predicted_value / B.predicted_value
            relative_error = np.sqrt(
                (A.uncertainty / A.predicted_value)**2 +
                (B.uncertainty / B.predicted_value)**2
            )
            uncertainty = result * relative_error
            confidence = min(A.confidence, B.confidence)

            return PredictionResult(
                property_name=f"{A.property_name}/{B.property_name}",
                predicted_value=result,
                uncertainty=uncertainty,
                confidence=confidence,
                method="uncertainty_propagation",
                notes=f"Derived from {formula}"
            )

        # Simple case: sum of properties
        elif "+" in formula:
            result = sum(p.predicted_value for p in predictions)
            # Independent errors add in quadrature
            uncertainty = np.sqrt(sum(p.uncertainty**2 for p in predictions))
            confidence = min(p.confidence for p in predictions)

            return PredictionResult(
                property_name="+".join(p.property_name for p in predictions),
                predicted_value=result,
                uncertainty=uncertainty,
                confidence=confidence,
                method="uncertainty_propagation",
                notes=f"Derived from {formula}"
            )

        else:
            raise ValueError(f"Unsupported formula: {formula}")


if __name__ == "__main__":
    db = MaterialsDatabase()
    predictor = MaterialPropertyPredictor(db)

    print("="*70)
    print("MATERIAL PROPERTY PREDICTOR")
    print("="*70)

    # Example 1: Predict from composition
    print("\n1. Predict properties of unknown steel alloy")
    print("-" * 70)

    composition = {
        "Fe": 96.5,
        "Cr": 2.0,
        "Ni": 1.0,
        "Mo": 0.5
    }

    predictions = predictor.predict_from_composition(
        composition,
        ["youngs_modulus", "tensile_strength", "density"]
    )

    print(f"Composition: {composition}")
    for pred in predictions:
        print(f"\n{pred.property_name}:")
        print(f"  Predicted: {pred.predicted_value:.2f} ± {pred.uncertainty:.2f}")
        print(f"  Confidence: {pred.confidence:.2%}")
        print(f"  Method: {pred.method}")
        print(f"  {pred.notes}")

    # Example 2: Predict from structure
    print("\n2. Predict properties from crystal structure")
    print("-" * 70)

    predictions = predictor.predict_from_structure(
        crystal_structure="BCC",
        bonding_type="metallic",
        properties=["youngs_modulus", "density"]
    )

    print("Structure: BCC metallic")
    for pred in predictions:
        print(f"\n{pred.property_name}:")
        print(f"  Predicted: {pred.predicted_value:.2f} ± {pred.uncertainty:.2f}")
        print(f"  Confidence: {pred.confidence:.2%}")

    # Example 3: Predict by similarity
    print("\n3. Predict unknown property by similarity")
    print("-" * 70)

    ti = db.get_material("Ti-6Al-4V")
    pred = predictor.predict_by_similarity(ti, "fracture_toughness")

    print(f"Reference: {ti.name}")
    print(f"\n{pred.property_name}:")
    print(f"  Predicted: {pred.predicted_value:.2f} ± {pred.uncertainty:.2f}")
    print(f"  Confidence: {pred.confidence:.2%}")
    print(f"  {pred.notes}")

    # Example 4: Uncertainty propagation
    print("\n4. Calculate specific strength with uncertainty")
    print("-" * 70)

    # Get predictions for strength and density
    strength_pred = PredictionResult(
        property_name="tensile_strength",
        predicted_value=500,
        uncertainty=50,
        confidence=0.85,
        method="test"
    )

    density_pred = PredictionResult(
        property_name="density",
        predicted_value=7800,
        uncertainty=200,
        confidence=0.90,
        method="test"
    )

    specific_strength = predictor.uncertainty_propagation(
        [strength_pred, density_pred],
        formula="A / B"
    )

    print(f"Specific Strength = Strength / Density")
    print(f"  Predicted: {specific_strength.predicted_value:.4f} ± {specific_strength.uncertainty:.4f}")
    print(f"  Confidence: {specific_strength.confidence:.2%}")
    print(f"  {specific_strength.notes}")

    print("\n" + "="*70)
    print("Property Predictor ready! ✓")
