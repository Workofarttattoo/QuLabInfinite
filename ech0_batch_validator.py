#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

ECH0 Batch Invention Validator
Validate multiple inventions in parallel with confidence scoring
"""

from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime
import json

from ech0_invention_accelerator import ECH0_InventionAccelerator, InventionConcept


class ECH0_BatchValidator:
    """
    Batch invention validator with parallel processing and confidence scoring.

    Features:
    - Parallel validation of multiple concepts
    - Automatic rejection of low-scoring concepts
    - Confidence scoring for each validation step
    - Progress tracking
    - Summary report generation
    """

    def __init__(self, max_workers: int = 4):
        """
        Initialize batch validator.

        Args:
            max_workers: Maximum parallel workers (default: 4)
        """
        self.accelerator = ECH0_InventionAccelerator()
        self.max_workers = max_workers

        # Validation thresholds
        self.min_feasibility = 0.3
        self.min_impact = 0.4
        self.min_quantum_score = 0.5

    def validate_all_inventions(self,
                               concepts: List[InventionConcept],
                               requirements: Dict[str, Any],
                               parallel: bool = True) -> List[Dict[str, Any]]:
        """
        Validate multiple invention concepts.

        Args:
            concepts: List of InventionConcept objects to validate
            requirements: Common requirements (application, constraints, budget)
            parallel: Run validations in parallel (default True)

        Returns:
            List of validation results sorted by quantum_score (highest first)
        """
        print(f"\n{'='*70}")
        print(f"  ECH0 BATCH INVENTION VALIDATOR")
        print(f"{'='*70}")
        print(f"Validating {len(concepts)} invention concepts...")
        print(f"Parallel: {parallel} | Workers: {self.max_workers if parallel else 1}")
        print(f"{'='*70}\n")

        start_time = time.time()
        results = []

        if parallel:
            results = self._validate_parallel(concepts, requirements)
        else:
            results = self._validate_sequential(concepts, requirements)

        # Sort by quantum_score (highest first)
        results.sort(key=lambda x: x['quantum_score'], reverse=True)

        elapsed = time.time() - start_time

        # Generate summary
        summary = self._generate_summary(results, elapsed)

        print(f"\n{'='*70}")
        print(f"  VALIDATION COMPLETE")
        print(f"{'='*70}")
        print(summary)
        print(f"{'='*70}\n")

        return results

    def _validate_parallel(self,
                          concepts: List[InventionConcept],
                          requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate concepts in parallel."""
        results = []
        completed = 0
        total = len(concepts)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all validations
            future_to_concept = {
                executor.submit(self._validate_single, concept, requirements): concept
                for concept in concepts
            }

            # Collect results as they complete
            for future in as_completed(future_to_concept):
                concept = future_to_concept[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1

                    # Progress update
                    status = "✅ PASS" if result['passed'] else "❌ FAIL"
                    print(f"[{completed}/{total}] {status} - {concept.name} "
                          f"(score: {result['quantum_score']:.2f})")

                except Exception as exc:
                    print(f"[{completed}/{total}] ❌ ERROR - {concept.name}: {exc}")
                    completed += 1

        return results

    def _validate_sequential(self,
                            concepts: List[InventionConcept],
                            requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate concepts sequentially."""
        results = []

        for i, concept in enumerate(concepts, 1):
            try:
                result = self._validate_single(concept, requirements)
                results.append(result)

                status = "✅ PASS" if result['passed'] else "❌ FAIL"
                print(f"[{i}/{len(concepts)}] {status} - {concept.name} "
                      f"(score: {result['quantum_score']:.2f})")

            except Exception as exc:
                print(f"[{i}/{len(concepts)}] ❌ ERROR - {concept.name}: {exc}")

        return results

    def _validate_single(self,
                        concept: InventionConcept,
                        requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single invention concept.

        Returns:
            Dict with validation results and confidence scores
        """
        # Run full acceleration pipeline
        acceleration_result = self.accelerator.accelerate_invention(concept, requirements)

        # Calculate confidence scores
        confidence_scores = self._calculate_confidence(concept, acceleration_result)

        # Determine if concept passes validation
        passed = self._check_thresholds(concept, confidence_scores)

        # Build result
        result = {
            'concept_name': concept.name,
            'description': concept.description,
            'feasibility': concept.feasibility,
            'impact': concept.impact,
            'cost_estimate': concept.cost_estimate,
            'quantum_score': concept.quantum_score,
            'physics_validated': concept.physics_validated,
            'chemistry_validated': concept.chemistry_validated,
            'required_materials': concept.required_materials,
            'confidence_scores': confidence_scores,
            'passed': passed,
            'rejection_reasons': self._get_rejection_reasons(concept, confidence_scores),
            'timestamp': datetime.now().isoformat(),
            'acceleration_result': acceleration_result
        }

        return result

    def _calculate_confidence(self,
                             concept: InventionConcept,
                             acceleration_result: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate confidence scores for each validation step.

        Returns:
            Dict with confidence scores (0.0-1.0) for each step
        """
        scores = {}

        # Material selection confidence
        if concept.required_materials:
            # High confidence if multiple materials found
            scores['material_selection'] = min(1.0, len(concept.required_materials) / 3)
        else:
            scores['material_selection'] = 0.0

        # Physics validation confidence
        if concept.physics_validated:
            scores['physics_validation'] = 0.9  # High confidence if passed
        else:
            scores['physics_validation'] = 0.3  # Low confidence if failed

        # Chemistry validation confidence
        if concept.chemistry_validated:
            scores['chemistry_validation'] = 0.9
        else:
            scores['chemistry_validation'] = 0.3

        # Cost estimation confidence
        if concept.cost_estimate > 0:
            # Higher confidence if cost is within budget
            budget = acceleration_result.get('budget', float('inf'))
            if concept.cost_estimate <= budget:
                scores['cost_estimation'] = 0.9
            else:
                scores['cost_estimation'] = 0.5  # Over budget
        else:
            scores['cost_estimation'] = 0.2  # No estimate

        # Quantum evaluation confidence
        scores['quantum_evaluation'] = concept.quantum_score

        # Overall confidence (average of all scores)
        scores['overall'] = sum(scores.values()) / len(scores)

        return scores

    def _check_thresholds(self,
                         concept: InventionConcept,
                         confidence_scores: Dict[str, float]) -> bool:
        """Check if concept meets minimum thresholds."""
        return (
            concept.feasibility >= self.min_feasibility and
            concept.impact >= self.min_impact and
            concept.quantum_score >= self.min_quantum_score and
            confidence_scores['overall'] >= 0.5
        )

    def _get_rejection_reasons(self,
                               concept: InventionConcept,
                               confidence_scores: Dict[str, float]) -> List[str]:
        """Get list of reasons why concept failed validation."""
        reasons = []

        if concept.feasibility < self.min_feasibility:
            reasons.append(f"Low feasibility: {concept.feasibility:.2f} < {self.min_feasibility}")

        if concept.impact < self.min_impact:
            reasons.append(f"Low impact: {concept.impact:.2f} < {self.min_impact}")

        if concept.quantum_score < self.min_quantum_score:
            reasons.append(f"Low quantum score: {concept.quantum_score:.2f} < {self.min_quantum_score}")

        if confidence_scores['overall'] < 0.5:
            reasons.append(f"Low overall confidence: {confidence_scores['overall']:.2f} < 0.5")

        if not concept.required_materials:
            reasons.append("No suitable materials found")

        if not concept.physics_validated:
            reasons.append("Physics validation failed")

        return reasons

    def _generate_summary(self,
                         results: List[Dict[str, Any]],
                         elapsed_time: float) -> str:
        """Generate summary report of validation results."""
        total = len(results)
        passed = sum(1 for r in results if r['passed'])
        failed = total - passed

        avg_score = sum(r['quantum_score'] for r in results) / total if total > 0 else 0
        avg_confidence = sum(r['confidence_scores']['overall'] for r in results) / total if total > 0 else 0

        # Top 3 inventions
        top_3 = results[:3]

        summary = f"""
RESULTS:
  Total Concepts: {total}
  Passed: {passed} ({passed/total*100:.1f}%)
  Failed: {failed} ({failed/total*100:.1f}%)

METRICS:
  Average Quantum Score: {avg_score:.3f}
  Average Confidence: {avg_confidence:.3f}
  Validation Time: {elapsed_time:.1f}s
  Avg Time per Concept: {elapsed_time/total:.1f}s

TOP 3 INVENTIONS:
"""

        for i, inv in enumerate(top_3, 1):
            summary += f"  {i}. {inv['concept_name']}\n"
            summary += f"     Score: {inv['quantum_score']:.3f} | "
            summary += f"Feasibility: {inv['feasibility']:.2f} | "
            summary += f"Impact: {inv['impact']:.2f}\n"
            if inv['required_materials']:
                summary += f"     Materials: {', '.join(inv['required_materials'][:3])}\n"

        return summary

    def save_results(self,
                    results: List[Dict[str, Any]],
                    filepath: str = "batch_validation_results.json"):
        """Save validation results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_concepts': len(results),
                'passed': sum(1 for r in results if r['passed']),
                'results': results
            }, f, indent=2)

        print(f"✅ Results saved to: {filepath}")


def demo_batch_validation():
    """Demo: Validate multiple invention concepts."""
    print("\n🤖 ECH0 Batch Validation Demo\n")

    # Create test concepts
    concepts = [
        InventionConcept(
            "Aerogel Heat Shield",
            "Ultra-lightweight heat shield using advanced aerogel composites"
        ),
        InventionConcept(
            "Carbon Nanotube Battery",
            "High-capacity battery using SWCNT electrodes"
        ),
        InventionConcept(
            "Graphene Supercapacitor",
            "Ultra-fast charging capacitor using graphene sheets"
        ),
        InventionConcept(
            "Titanium Alloy Frame",
            "Lightweight aerospace frame using Ti-6Al-4V"
        ),
        InventionConcept(
            "Ceramic Thermal Barrier",
            "High-temperature ceramic coating for turbine blades"
        ),
    ]

    # Common requirements
    requirements = {
        'application': 'aerospace',
        'budget': 10000.0,
        'constraints': {
            'max_weight': 5.0,  # kg
            'min_strength': 500.0,  # MPa
        }
    }

    # Validate all
    validator = ECH0_BatchValidator(max_workers=4)
    results = validator.validate_all_inventions(concepts, requirements, parallel=True)

    # Save results
    validator.save_results(results)

    # Show passed inventions
    passed = [r for r in results if r['passed']]
    print(f"\n✅ {len(passed)} INVENTIONS PASSED VALIDATION:")
    for inv in passed:
        print(f"  - {inv['concept_name']} (score: {inv['quantum_score']:.3f})")


if __name__ == "__main__":
    demo_batch_validation()
