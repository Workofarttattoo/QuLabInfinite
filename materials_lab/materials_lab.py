#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Materials Lab - Main API for all materials science operations
"""

try:  # allow package-style imports
    from .materials_database import MaterialsDatabase, MaterialProperties  # type: ignore
    from .material_testing import (  # type: ignore
        TensileTest, CompressionTest, FatigueTest, ImpactTest,
        HardnessTest, ThermalTest, CorrosionTest, EnvironmentalTest
    )
    from .material_designer import (  # type: ignore
        AlloyOptimizer, CompositeDesigner, NanostructureEngineer,
        SurfaceTreatment, AdditiveManufacturing
    )
    from .material_property_predictor import MaterialPropertyPredictor  # type: ignore
    from .material_profiles import MaterialProfileGenerator  # type: ignore
    from .calibration import CalibrationManager  # type: ignore
    from .uncertainty import estimate_property_uncertainty  # type: ignore
    from .phase_change import run_ice_analysis  # type: ignore
except ImportError:  # pragma: no cover
    from materials_database import MaterialsDatabase, MaterialProperties  # type: ignore
    from material_testing import (  # type: ignore
        TensileTest, CompressionTest, FatigueTest, ImpactTest,
        HardnessTest, ThermalTest, CorrosionTest, EnvironmentalTest
    )
    from material_designer import (  # type: ignore
        AlloyOptimizer, CompositeDesigner, NanostructureEngineer,
        SurfaceTreatment, AdditiveManufacturing
    )
    from material_property_predictor import MaterialPropertyPredictor  # type: ignore
    from material_profiles import MaterialProfileGenerator  # type: ignore
    from calibration import CalibrationManager  # type: ignore
    from uncertainty import estimate_property_uncertainty  # type: ignore
    from phase_change import run_ice_analysis  # type: ignore

from typing import Dict, List, Optional, Any, Sequence
import time


class MaterialsLab:
    """
    Main Materials Science Laboratory API

    Provides unified interface for:
    - Material database access (1000+ materials)
    - Material testing simulations
    - Material design and optimization
    - Property prediction with ML
    """

    def __init__(self):
        """Initialize Materials Lab"""
        print("[info] Initializing Materials Laboratory...")
        start = time.time()

        self.db = MaterialsDatabase()
        self.predictor = MaterialPropertyPredictor(self.db)
        self.profile_generator = MaterialProfileGenerator(self.db)
        self.calibration_manager = CalibrationManager()

        end = time.time()
        print(f"[info] Materials Lab ready in {(end-start)*1000:.1f} ms")
        print(f"[info] Database: {self.db.get_count()} materials")

    # ===== DATABASE ACCESS =====

    def get_material(self, name: str) -> Optional[MaterialProperties]:
        """Get material by name"""
        return self.db.get_material(name)

    def search_materials(self, **criteria) -> List[MaterialProperties]:
        """Search materials by criteria"""
        return self.db.search_materials(**criteria)

    def list_categories(self) -> List[str]:
        """List all material categories"""
        return self.db.list_categories()

    def list_materials(self, category: Optional[str] = None) -> List[str]:
        """List material names, optionally filtered by category."""
        if category:
            return sorted(
                props.name
                for props in self.db.materials.values()
                if props.category.lower() == category.lower()
            )
        return sorted(self.db.materials.keys())

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        return self.db.get_statistics()

    def get_material_profile(self, material_name: str) -> Dict[str, Any]:
        """Generate a detailed property profile with curves and metadata."""
        if not self.db.get_material(material_name):
            raise ValueError(f"Material not found: {material_name}")
        profile = self.profile_generator.build_profile(material_name)
        safety = self.db.get_safety_data(material_name)
        if safety:
            profile["safety"] = safety
        return profile

    def get_material_safety(self, material_name: str) -> Optional[Dict[str, object]]:
        """Return MSDS-style safety information if available."""
        return self.db.get_safety_data(material_name)

    def register_calibration(self, material_name: str, test_type: str, property_name: str,
                             reference_value: float, measured_value: float) -> Dict[str, float]:
        """Register calibration measurement."""
        record = self.calibration_manager.register(material_name, test_type, property_name,
                                                   reference_value, measured_value)
        return record.to_dict()

    def get_calibration_summary(self, material_name: str, test_type: str) -> Dict[str, Dict[str, float]]:
        return self.calibration_manager.summary(material_name, test_type)

    def simulate_ice_growth(self,
                            material_name: str,
                            temperature_k: float,
                            relative_humidity: float,
                            duration_hours: float = 1.0) -> Dict[str, float]:
        material = self.get_material(material_name)
        if material is None:
            raise ValueError(f"Material not found: {material_name}")
        return run_ice_analysis(material, temperature_k, relative_humidity, duration_hours)

    def run_batch_experiments(self, batch_specs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run multiple experiments and return serialisable summaries."""

        from analysis_tools import run_batch_experiments as _batch_runner  # type: ignore

        results = _batch_runner(batch_specs, lab=self)
        return [
            {
                "experiment_id": res.experiment_id,
                "material": res.material,
                "test_type": res.test_type,
                "success": res.success,
                "payload": res.payload,
                "error": res.error,
            }
            for res in results
        ]

    # ===== TESTING =====

    def tensile_test(self, material_name: str, **kwargs):
        """Run tensile test"""
        mat = self.get_material(material_name)
        if not mat:
            raise ValueError(f"Material not found: {material_name}")
        test = TensileTest(mat)
        result = test.run(**kwargs)
        self._attach_uncertainty(mat, "tensile", result.data,
                                 ["youngs_modulus", "yield_strength", "ultimate_strength"])
        self._apply_calibration(mat.name, "tensile", result.data,
                                ["youngs_modulus", "yield_strength", "ultimate_strength"])
        return result

    def compression_test(self, material_name: str, **kwargs):
        """Run compression test"""
        mat = self.get_material(material_name)
        if not mat:
            raise ValueError(f"Material not found: {material_name}")
        test = CompressionTest(mat)
        result = test.run(**kwargs)
        self._attach_uncertainty(mat, "compression", result.data,
                                 ["compressive_modulus", "compressive_strength"])
        self._apply_calibration(mat.name, "compression", result.data,
                                ["compressive_modulus", "compressive_strength"])
        return result

    def fatigue_test(self, material_name: str, **kwargs):
        """Run fatigue test"""
        mat = self.get_material(material_name)
        if not mat:
            raise ValueError(f"Material not found: {material_name}")
        test = FatigueTest(mat)
        result = test.run(**kwargs)
        self._attach_uncertainty(mat, "fatigue", result.data, ["fatigue_limit"])
        self._apply_calibration(mat.name, "fatigue", result.data, ["fatigue_limit"])
        return result

    def impact_test(self, material_name: str, **kwargs):
        """Run impact test"""
        mat = self.get_material(material_name)
        if not mat:
            raise ValueError(f"Material not found: {material_name}")
        test = ImpactTest(mat)
        result = test.run(**kwargs)
        self._attach_uncertainty(mat, "impact", result.data, ["impact_energy"])
        return result

    def hardness_test(self, material_name: str, **kwargs):
        """Run hardness test"""
        mat = self.get_material(material_name)
        if not mat:
            raise ValueError(f"Material not found: {material_name}")
        test = HardnessTest(mat)
        result = test.run(**kwargs)
        self._attach_uncertainty(mat, "hardness", result.data, ["vickers"])
        return result

    def thermal_test(self, material_name: str, test_type: str = "dsc", **kwargs):
        """Run thermal test"""
        mat = self.get_material(material_name)
        if not mat:
            raise ValueError(f"Material not found: {material_name}")
        test = ThermalTest(mat)

        if test_type == "dsc":
            result = test.run_dsc(**kwargs)
        elif test_type == "conductivity":
            result = test.run_thermal_conductivity(**kwargs)
        else:
            raise ValueError(f"Unknown thermal test type: {test_type}")
        self._attach_uncertainty(mat, "thermal", result.data, list(result.data.keys()))
        return result

    def corrosion_test(self, material_name: str, test_type: str = "salt_spray", **kwargs):
        """Run corrosion test"""
        mat = self.get_material(material_name)
        if not mat:
            raise ValueError(f"Material not found: {material_name}")
        test = CorrosionTest(mat)

        if test_type == "salt_spray":
            result = test.run_salt_spray(**kwargs)
        elif test_type == "electrochemical":
            result = test.run_electrochemical(**kwargs)
        else:
            raise ValueError(f"Unknown corrosion test type: {test_type}")
        self._attach_uncertainty(mat, "corrosion", result.data, list(result.data.keys()))
        return result

    def environmental_test(self, material_name: str, **kwargs):
        """Run environmental test"""
        mat = self.get_material(material_name)
        if not mat:
            raise ValueError(f"Material not found: {material_name}")
        test = EnvironmentalTest(mat)
        result = test.run_extreme_cold(**kwargs)
        self._attach_uncertainty(mat, "environmental_extreme_cold", result.data,
                                 ["heat_loss_rate_W_m2", "thermal_stress_MPa", "adjusted_tensile_strength"])
        return result

    # ------------------------------------------------------------ helpers

    def _apply_calibration(self,
                            material_name: str,
                            test_type: str,
                            data: Dict[str, Any],
                            properties: List[str]) -> None:
        """Apply stored calibration biases to selected properties."""
        numeric_values = {
            prop: data[prop]
            for prop in properties
            if prop in data and isinstance(data[prop], (int, float))
        }
        if not numeric_values:
            return

        corrections = self.calibration_manager.apply(material_name, test_type, numeric_values)
        if corrections:
            data.setdefault("calibration", {}).update(corrections)
        for prop, value in numeric_values.items():
            data[prop] = value

    def _attach_uncertainty(self,
                            material: MaterialProperties,
                            test_type: str,
                            data: Dict[str, Any],
                            properties: List[str]) -> None:
        """Attach one-sigma uncertainty estimates to the data payload."""
        if not properties:
            return
        uncertainties = data.setdefault("uncertainty", {})
        for prop in properties:
            if prop in data and isinstance(data[prop], (int, float)):
                uncertainties[prop] = estimate_property_uncertainty(material, prop, data[prop], test_type)

    # ===== DESIGN & OPTIMIZATION =====

    def optimize_alloy(self, base_elements: List, target_properties: Dict, **kwargs):
        """Optimize alloy composition"""
        optimizer = AlloyOptimizer(base_elements, target_properties, **kwargs)
        return optimizer.optimize()

    def design_composite(self, fiber_name: str, matrix_name: str, **kwargs):
        """Design composite material"""
        fiber = self.get_material(fiber_name)
        matrix = self.get_material(matrix_name)
        if not fiber or not matrix:
            raise ValueError("Fiber or matrix material not found")

        designer = CompositeDesigner(fiber, matrix)
        return designer.design_laminate(**kwargs)

    def add_nanoparticles(self, base_material_name: str, **kwargs):
        """Add nanoparticles to material"""
        mat = self.get_material(base_material_name)
        if not mat:
            raise ValueError(f"Material not found: {base_material_name}")

        engineer = NanostructureEngineer(mat)
        return engineer.add_nanoparticles(**kwargs)

    def apply_coating(self, base_material_name: str, **kwargs):
        """Apply surface coating"""
        mat = self.get_material(base_material_name)
        if not mat:
            raise ValueError(f"Material not found: {base_material_name}")

        treatment = SurfaceTreatment(mat)
        return treatment.apply_coating(**kwargs)

    def design_lattice(self, base_material_name: str, **kwargs):
        """Design lattice structure for AM"""
        mat = self.get_material(base_material_name)
        if not mat:
            raise ValueError(f"Material not found: {base_material_name}")

        am = AdditiveManufacturing(mat)
        return am.design_lattice_structure(**kwargs)

    # ===== PREDICTION =====

    def predict_from_composition(self, composition: Dict[str, float], properties: List[str]):
        """Predict properties from composition"""
        return self.predictor.predict_from_composition(composition, properties)

    def predict_from_structure(self, crystal_structure: str, bonding_type: str, properties: List[str]):
        """Predict properties from structure"""
        return self.predictor.predict_from_structure(crystal_structure, bonding_type, properties)

    def predict_by_similarity(self, reference_material_name: str, property_name: str):
        """Predict property by similarity"""
        mat = self.get_material(reference_material_name)
        if not mat:
            raise ValueError(f"Material not found: {reference_material_name}")
        return self.predictor.predict_by_similarity(mat, property_name)

    # ===== CONVENIENCE METHODS =====

    def compare_materials(self, material_names: List[str], properties: List[str]) -> Dict:
        """Compare multiple materials"""
        results = {}

        for name in material_names:
            mat = self.get_material(name)
            if mat:
                results[name] = {
                    prop: getattr(mat, prop, None)
                    for prop in properties
                }

        return results

    def find_best_material(self,
                          category: Optional[str] = None,
                          optimize_for: str = "strength",
                          constraints: Optional[Dict] = None) -> MaterialProperties:
        """
        Find best material for application

        Args:
            category: Material category to search
            optimize_for: Property to maximize
            constraints: Dict of property constraints (min/max values)
        """
        materials = list(self.db.materials.values())

        # Filter by category
        if category:
            materials = [m for m in materials if m.category.lower() == category.lower()]

        # Apply constraints
        if constraints:
            for prop, (min_val, max_val) in constraints.items():
                materials = [
                    m for m in materials
                    if min_val <= getattr(m, prop, 0) <= max_val
                ]

        # Find best
        if not materials:
            return None

        best = max(materials, key=lambda m: getattr(m, optimize_for, 0))
        return best


if __name__ == "__main__":
    # Initialize lab
    lab = MaterialsLab()

    print("\n" + "="*70)
    print("MATERIALS LAB - COMPREHENSIVE DEMONSTRATION")
    print("="*70)

    # Demo 1: Database access
    print("\n1. DATABASE ACCESS")
    print("-" * 70)

    airloy = lab.get_material("Airloy X103")
    print(f"Material: {airloy.name}")
    print(f"  Category: {airloy.category} / {airloy.subcategory}")
    print(f"  Density: {airloy.density} kg/m³")
    print(f"  Thermal Conductivity: {airloy.thermal_conductivity*1000:.1f} mW/(m·K)")
    print(f"  Tensile Strength: {airloy.tensile_strength:.2f} MPa")

    # Demo 2: Material testing
    print("\n2. TENSILE TEST - Ti-6Al-4V")
    print("-" * 70)

    result = lab.tensile_test("Ti-6Al-4V", max_strain=0.15)
    print(f"Test: {result.test_type}")
    print(f"  Young's Modulus: {result.data['youngs_modulus']:.0f} MPa")
    print(f"  Yield Strength: {result.data['yield_strength']:.0f} MPa")
    print(f"  Ultimate Strength: {result.data['ultimate_strength']:.0f} MPa")
    print(f"  Elongation: {result.data['elongation_at_break']:.1f}%")
    print(f"  Toughness: {result.data['toughness']:.1f} MJ/m³")

    # Demo 3: Environmental test
    print("\n3. EXTREME COLD TEST - Airloy X103 at -200°C with 30 mph wind")
    print("-" * 70)

    result = lab.environmental_test(
        "Airloy X103",
        temperature=73,  # -200°C
        wind_speed=13.4,  # 30 mph
        duration_hours=24
    )

    print(f"Status: {result.data['status']}")
    print(f"  Temperature: {result.data['temperature_celsius']:.0f}°C")
    print(f"  Wind: {result.data['wind_speed_mph']:.0f} mph")
    print(f"  Strength Retention: {result.data['strength_retention_percent']:.1f}%")
    print(f"  Heat Loss: {result.data['heat_loss_rate_W_m2']:.1f} W/m²")
    print(f"  Result: {'✓ PASS' if result.success else '✗ FAIL'}")

    # Demo 4: Material design
    print("\n4. DESIGN CARBON FIBER COMPOSITE")
    print("-" * 70)

    result = lab.design_composite(
        "Carbon Fiber Epoxy",
        "Epoxy Resin",
        fiber_volume_fraction=0.65,
        layup=[0, 45, 90, -45]
    )

    print(f"Composite: {result.optimized_properties.name}")
    print(f"  Density: {result.optimized_properties.density:.0f} kg/m³")
    print(f"  Modulus: {result.optimized_properties.youngs_modulus:.1f} GPa")
    print(f"  Strength: {result.optimized_properties.tensile_strength:.0f} MPa")
    print(f"  Specific Strength: {result.fitness_score:.1f}")

    # Demo 5: Material comparison
    print("\n5. COMPARE AEROSPACE MATERIALS")
    print("-" * 70)

    comparison = lab.compare_materials(
        ["Al 7075-T6", "Ti-6Al-4V", "Carbon Fiber Epoxy"],
        ["density", "tensile_strength", "youngs_modulus"]
    )

    print(f"{'Material':<25} {'Density':<12} {'Strength':<12} {'Modulus':<12}")
    print(f"{'':25} {'(kg/m³)':<12} {'(MPa)':<12} {'(GPa)':<12}")
    print("-" * 70)
    for name, props in comparison.items():
        print(f"{name:<25} {props['density']:<12.0f} {props['tensile_strength']:<12.0f} {props['youngs_modulus']:<12.1f}")

    # Demo 6: Find best material
    print("\n6. FIND BEST LIGHTWEIGHT HIGH-STRENGTH MATERIAL")
    print("-" * 70)

    best = lab.find_best_material(
        optimize_for="tensile_strength",
        constraints={
            "density": (0, 2000),  # Low density
            "tensile_strength": (100, 1e6)  # Reasonable strength
        }
    )

    if best:
        print(f"Best material: {best.name}")
        print(f"  Density: {best.density:.0f} kg/m³")
        print(f"  Strength: {best.tensile_strength:.0f} MPa")
        print(f"  Specific Strength: {best.tensile_strength/best.density*1000:.1f} MPa/(g/cm³)")

    # Demo 7: Statistics
    print("\n7. DATABASE STATISTICS")
    print("-" * 70)

    stats = lab.get_statistics()
    print(f"Total Materials: {stats['total_materials']}")
    print(f"\nBy Category:")
    for cat, count in sorted(stats['categories'].items()):
        print(f"  {cat}: {count}")

    print("\n" + "="*70)
    print("Materials Lab demonstration complete! ✓")
    print("="*70)
