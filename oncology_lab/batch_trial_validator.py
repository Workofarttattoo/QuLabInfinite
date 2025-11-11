#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Batch Clinical Trial Validator for Oncology Lab
Validates QuLabInfinite simulations against 100+ known clinical outcomes
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import sys

from .oncology_lab import OncologyLaboratory, OncologyLabConfig, TumorType, CancerStage
from .drug_response import get_drug_from_database


@dataclass
class ValidationResult:
    """Results from validating a single trial"""
    trial_id: str
    success: bool
    predicted_reduction: float
    actual_reduction: float
    error_percent: float
    within_tolerance: bool
    details: str


class BatchTrialValidator:
    """Validates oncology lab against comprehensive clinical trial database"""

    def __init__(self, dataset_path: str = None, tolerance_percent: float = 20.0):
        """
        Initialize batch validator

        Args:
            dataset_path: Path to clinical_trial_datasets.json
            tolerance_percent: Acceptable error margin (default 20%)
        """
        if dataset_path is None:
            dataset_path = Path(__file__).parent / "clinical_trial_datasets.json"

        self.dataset_path = Path(dataset_path)
        self.tolerance = tolerance_percent
        self.trials = []
        self.results = []

        self._load_datasets()

    def _load_datasets(self):
        """Load clinical trial datasets"""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        with open(self.dataset_path, 'r') as f:
            data = json.load(f)

        self.trials = data['trials']
        self.metadata = data['metadata']

        print(f"✅ Loaded {len(self.trials)} clinical trials from {self.dataset_path}")
        print(f"   Tumor types: {self.metadata['tumor_types']}")
        print(f"   Drug regimens: {self.metadata['drug_regimens']}")

    def validate_single_trial(self, trial: Dict) -> ValidationResult:
        """
        Validate a single clinical trial

        Args:
            trial: Trial data dictionary

        Returns:
            ValidationResult with comparison metrics
        """
        trial_id = trial['trial_id']

        try:
            # Map trial data to lab config
            tumor_type_map = {
                'breast_cancer': TumorType.BREAST_CANCER,
                'lung_cancer': TumorType.LUNG_CANCER,
                'colorectal_cancer': TumorType.COLORECTAL_CANCER,
                'prostate_cancer': TumorType.PROSTATE_CANCER,
                'pancreatic_cancer': TumorType.PANCREATIC_CANCER,
                'glioblastoma': TumorType.GLIOBLASTOMA,
                'melanoma': TumorType.MELANOMA,
                'ovarian_cancer': TumorType.OVARIAN_CANCER
            }

            tumor_type = tumor_type_map.get(trial['tumor_type'])
            if not tumor_type:
                return ValidationResult(
                    trial_id=trial_id,
                    success=False,
                    predicted_reduction=0.0,
                    actual_reduction=trial['tumor_reduction_percent'],
                    error_percent=100.0,
                    within_tolerance=False,
                    details=f"Unknown tumor type: {trial['tumor_type']}"
                )

            stage = CancerStage(trial['stage'])

            # Initialize lab with trial parameters
            config = OncologyLabConfig(
                tumor_type=tumor_type,
                stage=stage,
                initial_tumor_cells=trial['initial_cell_count']
            )

            lab = OncologyLaboratory(config)

            # Apply drugs from regimen
            for drug_name in trial['drug_regimen']:
                drug = get_drug_from_database(drug_name)
                if drug:
                    lab.administer_drug(drug_name, dose_mg=drug.standard_dose_mg)

            # Run simulation for treatment duration (OPTIMIZED for speed)
            treatment_hours = trial['treatment_duration_days'] * 24
            time_step_hours = 12.0  # Use 12-hour timesteps (balance speed/accuracy)
            time_steps = int(treatment_hours / time_step_hours)

            initial_cells = lab.tumor.get_statistics()['alive_cells']

            # Simulate treatment with larger timesteps
            for _ in range(min(time_steps, 400)):  # Cap at 400 steps (~200 days max)
                lab.step(dt=time_step_hours)

            final_cells = lab.tumor.get_statistics()['alive_cells']

            # Calculate reduction
            if initial_cells > 0:
                predicted_reduction = ((initial_cells - final_cells) / initial_cells) * 100.0
            else:
                predicted_reduction = 0.0

            actual_reduction = trial['tumor_reduction_percent']
            error = abs(predicted_reduction - actual_reduction)
            within_tolerance = error <= self.tolerance

            return ValidationResult(
                trial_id=trial_id,
                success=True,
                predicted_reduction=predicted_reduction,
                actual_reduction=actual_reduction,
                error_percent=error,
                within_tolerance=within_tolerance,
                details="Simulation completed successfully"
            )

        except Exception as e:
            return ValidationResult(
                trial_id=trial_id,
                success=False,
                predicted_reduction=0.0,
                actual_reduction=trial.get('tumor_reduction_percent', 0.0),
                error_percent=100.0,
                within_tolerance=False,
                details=f"Simulation error: {str(e)}"
            )

    def validate_all(self, max_trials: int = None) -> Dict:
        """
        Validate all trials in dataset

        Args:
            max_trials: Maximum number of trials to run (None for all)

        Returns:
            Summary statistics dictionary
        """
        print(f"\n{'='*80}")
        print(f"  BATCH VALIDATION: {len(self.trials)} CLINICAL TRIALS")
        print(f"{'='*80}\n")

        trials_to_run = self.trials[:max_trials] if max_trials else self.trials

        self.results = []

        for i, trial in enumerate(trials_to_run, 1):
            print(f"[{i}/{len(trials_to_run)}] Validating {trial['trial_id']}...", end=" ")

            result = self.validate_single_trial(trial)
            self.results.append(result)

            if result.success and result.within_tolerance:
                print(f"✅ PASS (Error: {result.error_percent:.1f}%)")
            elif result.success:
                print(f"⚠️  HIGH ERROR ({result.error_percent:.1f}%)")
            else:
                print(f"❌ FAIL - {result.details}")

        # Calculate statistics
        successful = [r for r in self.results if r.success]
        within_tolerance = [r for r in self.results if r.within_tolerance]

        if successful:
            avg_error = np.mean([r.error_percent for r in successful])
            median_error = np.median([r.error_percent for r in successful])
            max_error = max([r.error_percent for r in successful])
        else:
            avg_error = median_error = max_error = 0.0

        summary = {
            'total_trials': len(self.results),
            'successful_simulations': len(successful),
            'within_tolerance': len(within_tolerance),
            'success_rate': len(successful) / len(self.results) * 100 if self.results else 0,
            'accuracy_rate': len(within_tolerance) / len(self.results) * 100 if self.results else 0,
            'average_error_percent': avg_error,
            'median_error_percent': median_error,
            'max_error_percent': max_error,
            'tolerance_used': self.tolerance
        }

        self._print_summary(summary)
        return summary

    def _print_summary(self, summary: Dict):
        """Print validation summary"""
        print(f"\n{'='*80}")
        print(f"  VALIDATION SUMMARY")
        print(f"{'='*80}\n")

        print(f"Total Trials:              {summary['total_trials']}")
        print(f"Successful Simulations:    {summary['successful_simulations']} ({summary['success_rate']:.1f}%)")
        print(f"Within Tolerance (±{self.tolerance}%):  {summary['within_tolerance']} ({summary['accuracy_rate']:.1f}%)")
        print(f"\nError Statistics:")
        print(f"  Average Error:           {summary['average_error_percent']:.2f}%")
        print(f"  Median Error:            {summary['median_error_percent']:.2f}%")
        print(f"  Maximum Error:           {summary['max_error_percent']:.2f}%")

        if summary['accuracy_rate'] >= 80:
            print(f"\n✅ VALIDATION PASSED - Accuracy: {summary['accuracy_rate']:.1f}%")
        elif summary['accuracy_rate'] >= 60:
            print(f"\n⚠️  VALIDATION MARGINAL - Accuracy: {summary['accuracy_rate']:.1f}%")
        else:
            print(f"\n❌ VALIDATION FAILED - Accuracy: {summary['accuracy_rate']:.1f}%")

    def export_results(self, output_path: str = None):
        """Export validation results to JSON"""
        if output_path is None:
            output_path = Path(__file__).parent / "validation_results.json"

        data = {
            'metadata': self.metadata,
            'tolerance_percent': self.tolerance,
            'results': [
                {
                    'trial_id': r.trial_id,
                    'success': r.success,
                    'predicted_reduction': r.predicted_reduction,
                    'actual_reduction': r.actual_reduction,
                    'error_percent': r.error_percent,
                    'within_tolerance': r.within_tolerance,
                    'details': r.details
                }
                for r in self.results
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n📁 Results exported to: {output_path}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Validate oncology lab against clinical trials")
    parser.add_argument('--dataset', type=str, help="Path to clinical trial dataset JSON")
    parser.add_argument('--tolerance', type=float, default=20.0, help="Error tolerance percent (default: 20)")
    parser.add_argument('--max-trials', type=int, help="Maximum trials to run")
    parser.add_argument('--export', type=str, help="Export results to JSON file")

    args = parser.parse_args()

    try:
        validator = BatchTrialValidator(
            dataset_path=args.dataset,
            tolerance_percent=args.tolerance
        )

        summary = validator.validate_all(max_trials=args.max_trials)

        if args.export:
            validator.export_results(args.export)

        # Exit with appropriate code
        if summary['accuracy_rate'] >= 80:
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
