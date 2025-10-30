"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

QuLabInfinite Extended API - Advanced Functions
Adds batch processing, comparisons, visualizations, and decision support
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from qulab_api import QuLabSimulator, ExperimentRequest, ExperimentType, ExperimentResult


@dataclass
class MaterialComparison:
    """Result of comparing multiple materials."""
    materials: List[str]
    comparison_metrics: Dict[str, Any]
    rankings: Dict[str, List[Tuple[str, float]]]  # metric -> [(material, score), ...]
    recommendation: str
    confidence: float
    analysis: str


@dataclass
class BatchExperimentResult:
    """Results from running multiple experiments."""
    total_experiments: int
    successful: int
    failed: int
    duration_seconds: float
    results: List[ExperimentResult]
    summary: Dict[str, Any]


class QuLabExtended(QuLabSimulator):
    """Extended QuLab API with advanced analysis capabilities."""

    def __init__(self):
        super().__init__()
        self.experiment_history: List[ExperimentResult] = []

    # ================================================================
    # BATCH PROCESSING
    # ================================================================

    def run_batch(self, requests: List[ExperimentRequest | str]) -> BatchExperimentResult:
        """
        Run multiple experiments in batch.

        Args:
            requests: List of experiment requests

        Returns:
            BatchExperimentResult with all results and summary
        """
        start_time = time.time()
        results = []
        successful = 0
        failed = 0

        for i, request in enumerate(requests):
            try:
                result = self.run(request)
                results.append(result)
                self.experiment_history.append(result)

                if result.success:
                    successful += 1
                else:
                    failed += 1

            except Exception as e:
                failed += 1
                results.append(ExperimentResult(
                    experiment_id=f"batch_{i}",
                    success=False,
                    data={},
                    error_message=str(e)
                ))

        duration = time.time() - start_time

        # Generate summary
        summary = self._summarize_batch(results)

        return BatchExperimentResult(
            total_experiments=len(requests),
            successful=successful,
            failed=failed,
            duration_seconds=duration,
            results=results,
            summary=summary
        )

    def _summarize_batch(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Generate summary statistics from batch results."""
        summary = {
            "success_rate": sum(1 for r in results if r.success) / len(results) if results else 0,
            "experiment_types": {},
            "materials_tested": set(),
            "avg_yield_strength_MPa": [],
            "avg_ultimate_strength_MPa": []
        }

        for result in results:
            if not result.success:
                continue

            # Track materials tested
            if "material" in result.data:
                summary["materials_tested"].add(result.data["material"])

            # Collect strength data
            if "yield_strength_MPa" in result.data:
                summary["avg_yield_strength_MPa"].append(result.data["yield_strength_MPa"])
            if "ultimate_strength_MPa" in result.data:
                summary["avg_ultimate_strength_MPa"].append(result.data["ultimate_strength_MPa"])

        # Calculate averages
        if summary["avg_yield_strength_MPa"]:
            summary["avg_yield_strength_MPa"] = float(np.mean(summary["avg_yield_strength_MPa"]))
        else:
            summary["avg_yield_strength_MPa"] = 0.0

        if summary["avg_ultimate_strength_MPa"]:
            summary["avg_ultimate_strength_MPa"] = float(np.mean(summary["avg_ultimate_strength_MPa"]))
        else:
            summary["avg_ultimate_strength_MPa"] = 0.0

        summary["materials_tested"] = list(summary["materials_tested"])

        return summary

    # ================================================================
    # MATERIALS COMPARISON
    # ================================================================

    def compare_materials(
        self,
        material_names: List[str],
        test_type: str = "tensile",
        optimization_goal: str = "strength_to_weight"
    ) -> MaterialComparison:
        """
        Compare multiple materials across key metrics.

        Args:
            material_names: List of material names to compare
            test_type: Type of test to run (tensile, compression, etc.)
            optimization_goal: What to optimize for
                - "strength_to_weight": Best strength/density ratio
                - "cost_effective": Best performance per dollar
                - "pure_strength": Highest absolute strength
                - "ductility": Best elongation/formability

        Returns:
            MaterialComparison with rankings and recommendation
        """
        # Run tests on all materials
        requests = []
        for mat_name in material_names:
            req = ExperimentRequest(
                experiment_type=ExperimentType.MATERIAL_TEST,
                description=f"Tensile test {mat_name}",
                parameters={"material": mat_name, "test_type": test_type}
            )
            requests.append(req)

        batch_result = self.run_batch(requests)

        # Extract metrics from each material
        materials_data = {}
        for result in batch_result.results:
            if not result.success:
                continue

            mat_name = result.data.get("material", "unknown")
            materials_data[mat_name] = {
                "yield_strength_MPa": result.data.get("yield_strength_MPa", 0),
                "ultimate_strength_MPa": result.data.get("ultimate_strength_MPa", 0),
                "elongation_percent": result.data.get("elongation_percent", 0),
                "density_kg_m3": result.data.get("density_kg_m3", 1000),
                "youngs_modulus_GPa": result.data.get("youngs_modulus_GPa", 0)
            }

        # Calculate comparison metrics
        comparison_metrics = self._calculate_comparison_metrics(
            materials_data, optimization_goal
        )

        # Rank materials
        rankings = self._rank_materials(materials_data, optimization_goal)

        # Generate recommendation
        recommendation, confidence, analysis = self._generate_recommendation(
            materials_data, rankings, optimization_goal
        )

        return MaterialComparison(
            materials=material_names,
            comparison_metrics=comparison_metrics,
            rankings=rankings,
            recommendation=recommendation,
            confidence=confidence,
            analysis=analysis
        )

    def _calculate_comparison_metrics(
        self,
        materials_data: Dict[str, Dict],
        goal: str
    ) -> Dict[str, Any]:
        """Calculate various comparison metrics."""
        metrics = {}

        for mat_name, data in materials_data.items():
            strength_to_weight = data["ultimate_strength_MPa"] / (data["density_kg_m3"] / 1000)
            stiffness_to_weight = data["youngs_modulus_GPa"] * 1000 / (data["density_kg_m3"] / 1000)

            metrics[mat_name] = {
                "strength_to_weight_ratio": strength_to_weight,
                "stiffness_to_weight_ratio": stiffness_to_weight,
                "ductility_score": data["elongation_percent"],
                "absolute_strength": data["ultimate_strength_MPa"]
            }

        return metrics

    def _rank_materials(
        self,
        materials_data: Dict[str, Dict],
        goal: str
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Rank materials by different criteria."""
        rankings = {}

        # Strength-to-weight ranking
        strength_to_weight = []
        for mat_name, data in materials_data.items():
            ratio = data["ultimate_strength_MPa"] / (data["density_kg_m3"] / 1000)
            strength_to_weight.append((mat_name, ratio))
        rankings["strength_to_weight"] = sorted(strength_to_weight, key=lambda x: x[1], reverse=True)

        # Pure strength ranking
        pure_strength = []
        for mat_name, data in materials_data.items():
            pure_strength.append((mat_name, data["ultimate_strength_MPa"]))
        rankings["pure_strength"] = sorted(pure_strength, key=lambda x: x[1], reverse=True)

        # Ductility ranking
        ductility = []
        for mat_name, data in materials_data.items():
            ductility.append((mat_name, data["elongation_percent"]))
        rankings["ductility"] = sorted(ductility, key=lambda x: x[1], reverse=True)

        # Stiffness ranking
        stiffness = []
        for mat_name, data in materials_data.items():
            stiffness.append((mat_name, data["youngs_modulus_GPa"]))
        rankings["stiffness"] = sorted(stiffness, key=lambda x: x[1], reverse=True)

        return rankings

    def _generate_recommendation(
        self,
        materials_data: Dict[str, Dict],
        rankings: Dict[str, List[Tuple[str, float]]],
        goal: str
    ) -> Tuple[str, float, str]:
        """Generate material recommendation based on optimization goal."""
        if goal == "strength_to_weight":
            top_material = rankings["strength_to_weight"][0][0]
            top_score = rankings["strength_to_weight"][0][1]
            runner_up_score = rankings["strength_to_weight"][1][1] if len(rankings["strength_to_weight"]) > 1 else 0

            # Calculate confidence (how much better is #1 than #2)
            if runner_up_score > 0:
                confidence = min(1.0, (top_score - runner_up_score) / runner_up_score)
            else:
                confidence = 0.9

            analysis = f"{top_material} offers the best strength-to-weight ratio at {top_score:.1f} MPa/(kg/m³). "
            analysis += f"This means maximum strength with minimum weight, ideal for aerospace, automotive, and portable applications."

        elif goal == "pure_strength":
            top_material = rankings["pure_strength"][0][0]
            top_score = rankings["pure_strength"][0][1]
            confidence = 0.85

            analysis = f"{top_material} has the highest absolute strength at {top_score:.1f} MPa. "
            analysis += f"Choose this when maximum load-bearing capacity is critical and weight is not a concern."

        elif goal == "ductility":
            top_material = rankings["ductility"][0][0]
            top_score = rankings["ductility"][0][1]
            confidence = 0.8

            analysis = f"{top_material} has the best ductility at {top_score:.1f}% elongation. "
            analysis += f"Excellent for forming, bending, and applications requiring impact resistance."

        else:  # Default to balanced
            # Score based on multiple factors
            scores = {}
            for mat_name in materials_data.keys():
                score = 0
                for ranking_type, ranked_list in rankings.items():
                    position = next((i for i, (m, _) in enumerate(ranked_list) if m == mat_name), len(ranked_list))
                    score += (len(ranked_list) - position)
                scores[mat_name] = score

            top_material = max(scores, key=scores.get)
            confidence = 0.75
            analysis = f"{top_material} offers the best overall balance of properties. "
            analysis += f"Good all-around choice when multiple factors matter equally."

        return top_material, confidence, analysis

    # ================================================================
    # ADVANCED ANALYSIS
    # ================================================================

    def analyze_fatigue_life(
        self,
        material_name: str,
        stress_amplitude_MPa: float,
        mean_stress_MPa: float = 0,
        cycles: int = 1_000_000
    ) -> Dict[str, Any]:
        """
        Estimate fatigue life using S-N curve approximation.

        Args:
            material_name: Material to analyze
            stress_amplitude_MPa: Alternating stress amplitude
            mean_stress_MPa: Mean stress
            cycles: Target number of cycles

        Returns:
            Fatigue analysis results
        """
        # Run basic material test first
        test_result = self.run(ExperimentRequest(
            experiment_type=ExperimentType.MATERIAL_TEST,
            description=f"Fatigue analysis for {material_name}",
            parameters={"material": material_name}
        ))

        if not test_result.success:
            return {"success": False, "error": "Material test failed"}

        ultimate_strength = test_result.data.get("ultimate_strength_MPa", 0)

        # Simplified S-N curve (actual implementation would use material-specific data)
        # Assume fatigue limit is ~40-50% of ultimate strength for steel
        fatigue_limit = 0.45 * ultimate_strength

        # Calculate safety factor
        effective_stress = stress_amplitude_MPa + 0.5 * abs(mean_stress_MPa)  # Simplified mean stress correction

        if effective_stress > fatigue_limit:
            # Basquin's equation approximation
            estimated_cycles = int(10**6 * (fatigue_limit / effective_stress)**8)
            safe = False
            message = f"Expected failure before {estimated_cycles:,} cycles"
        else:
            estimated_cycles = float('inf')
            safe = True
            message = "Infinite life expected (below fatigue limit)"

        safety_factor = fatigue_limit / effective_stress if effective_stress > 0 else float('inf')

        return {
            "success": True,
            "material": material_name,
            "fatigue_limit_MPa": fatigue_limit,
            "effective_stress_MPa": effective_stress,
            "safety_factor": safety_factor,
            "estimated_cycles": estimated_cycles,
            "safe_for_target": safe if estimated_cycles >= cycles else False,
            "analysis": message,
            "note": "Simplified S-N curve model - for screening only, validate with physical testing"
        }

    def thermal_cycling_analysis(
        self,
        material_name: str,
        min_temp_C: float,
        max_temp_C: float,
        cycles: int
    ) -> Dict[str, Any]:
        """
        Analyze material behavior under thermal cycling.

        Args:
            material_name: Material to test
            min_temp_C: Minimum temperature
            max_temp_C: Maximum temperature
            cycles: Number of thermal cycles

        Returns:
            Thermal cycling analysis
        """
        temp_range = max_temp_C - min_temp_C

        # Run environment tests at both extremes
        cold_result = self.simulate_environment({
            "temperature_C": min_temp_C,
            "pressure_Pa": 101325
        })

        hot_result = self.simulate_environment({
            "temperature_C": max_temp_C,
            "pressure_Pa": 101325
        })

        # Get material properties
        mat_result = self.run(ExperimentRequest(
            experiment_type=ExperimentType.MATERIAL_TEST,
            description=f"Thermal cycling {material_name}",
            parameters={"material": material_name}
        ))

        # Simplified thermal fatigue assessment
        # Real implementation would use CTE, thermal conductivity, etc.
        thermal_stress_estimate = temp_range * 0.1  # Simplified

        risk_level = "LOW"
        if temp_range > 200:
            risk_level = "HIGH"
        elif temp_range > 100:
            risk_level = "MODERATE"

        return {
            "success": True,
            "material": material_name,
            "temperature_range_C": temp_range,
            "cycles": cycles,
            "min_environment": cold_result,
            "max_environment": hot_result,
            "thermal_stress_estimate_MPa": thermal_stress_estimate,
            "risk_level": risk_level,
            "recommendation": self._thermal_cycling_recommendation(risk_level, temp_range)
        }

    def _thermal_cycling_recommendation(self, risk_level: str, temp_range: float) -> str:
        """Generate thermal cycling recommendation."""
        if risk_level == "LOW":
            return f"Thermal cycling risk is low. Material should handle {temp_range}°C range well."
        elif risk_level == "MODERATE":
            return f"Moderate thermal stress expected. Consider stress relief features or material with better CTE match."
        else:
            return f"High thermal stress risk. Physical testing strongly recommended. Consider thermal barriers or different material."

    # ================================================================
    # VISUALIZATION & REPORTING
    # ================================================================

    def generate_comparison_report(
        self,
        comparison: MaterialComparison
    ) -> str:
        """
        Generate a formatted comparison report.

        Args:
            comparison: MaterialComparison result

        Returns:
            Formatted markdown report
        """
        report = f"# Material Comparison Report\n\n"
        report += f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += f"## Materials Compared\n\n"

        for mat in comparison.materials:
            report += f"- {mat}\n"

        report += f"\n## Recommendation\n\n"
        report += f"**Best Choice**: {comparison.recommendation}\n"
        report += f"**Confidence**: {comparison.confidence*100:.1f}%\n\n"
        report += f"**Analysis**: {comparison.analysis}\n\n"

        report += f"## Rankings\n\n"
        for metric, ranked_list in comparison.rankings.items():
            report += f"### {metric.replace('_', ' ').title()}\n\n"
            for i, (mat, score) in enumerate(ranked_list, 1):
                report += f"{i}. **{mat}**: {score:.2f}\n"
            report += "\n"

        report += f"## Detailed Metrics\n\n"
        report += "| Material | Strength/Weight | Stiffness/Weight | Ductility | Absolute Strength |\n"
        report += "|----------|----------------|------------------|-----------|------------------|\n"

        for mat, metrics in comparison.comparison_metrics.items():
            report += f"| {mat} | {metrics['strength_to_weight_ratio']:.1f} | "
            report += f"{metrics['stiffness_to_weight_ratio']:.1f} | "
            report += f"{metrics['ductility_score']:.1f}% | "
            report += f"{metrics['absolute_strength']:.1f} MPa |\n"

        report += "\n---\n\n"
        report += "*Note: This is a preliminary screening analysis. Physical validation required for production use.*\n"

        return report

    def export_results_json(self, filepath: str) -> bool:
        """Export experiment history to JSON file."""
        try:
            data = {
                "experiments": [],
                "summary": {
                    "total_experiments": len(self.experiment_history),
                    "successful": sum(1 for e in self.experiment_history if e.success),
                    "failed": sum(1 for e in self.experiment_history if not e.success)
                }
            }

            for exp in self.experiment_history:
                data["experiments"].append({
                    "id": exp.experiment_id,
                    "success": exp.success,
                    "data": exp.data,
                    "notes": exp.notes,
                    "error": exp.error_message
                })

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            return True

        except Exception as e:
            print(f"Export failed: {e}")
            return False


def demo_extended_features():
    """Demonstrate extended QuLab features."""
    print("=" * 70)
    print("QuLabInfinite Extended API - Demo")
    print("=" * 70)

    api = QuLabExtended()

    # 1. Material Comparison
    print("\n1. Comparing materials for aerospace application...")
    print("-" * 70)

    comparison = api.compare_materials(
        material_names=["Al 6061-T6", "Ti-6Al-4V", "SS 304"],
        optimization_goal="strength_to_weight"
    )

    print(f"Recommendation: {comparison.recommendation}")
    print(f"Confidence: {comparison.confidence*100:.1f}%")
    print(f"Analysis: {comparison.analysis}")

    print("\nRankings (Strength-to-Weight):")
    for i, (mat, score) in enumerate(comparison.rankings["strength_to_weight"], 1):
        print(f"  {i}. {mat}: {score:.1f}")

    # 2. Fatigue Analysis
    print("\n\n2. Fatigue Life Analysis...")
    print("-" * 70)

    fatigue_result = api.analyze_fatigue_life(
        material_name="Al 6061-T6",
        stress_amplitude_MPa=150,
        mean_stress_MPa=50,
        cycles=1_000_000
    )

    if fatigue_result["success"]:
        print(f"Material: {fatigue_result['material']}")
        print(f"Fatigue Limit: {fatigue_result['fatigue_limit_MPa']:.1f} MPa")
        print(f"Safety Factor: {fatigue_result['safety_factor']:.2f}")
        print(f"Safe for 1M cycles: {fatigue_result['safe_for_target']}")
        print(f"Analysis: {fatigue_result['analysis']}")

    # 3. Thermal Cycling
    print("\n\n3. Thermal Cycling Analysis...")
    print("-" * 70)

    thermal_result = api.thermal_cycling_analysis(
        material_name="SS 304",
        min_temp_C=-40,
        max_temp_C=150,
        cycles=10000
    )

    print(f"Temperature Range: {thermal_result['temperature_range_C']}°C")
    print(f"Risk Level: {thermal_result['risk_level']}")
    print(f"Recommendation: {thermal_result['recommendation']}")

    # 4. Generate Report
    print("\n\n4. Generating Comparison Report...")
    print("-" * 70)

    report = api.generate_comparison_report(comparison)
    print(report[:500] + "...\n[truncated]")

    print("\n" + "=" * 70)
    print("Demo complete! Extended features working.")
    print("=" * 70)


if __name__ == "__main__":
    demo_extended_features()
