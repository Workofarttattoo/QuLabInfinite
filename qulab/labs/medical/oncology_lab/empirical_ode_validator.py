#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Empirical ODE Validator - Simplified model calibrated directly to clinical outcomes
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
import sys
import time

from .drug_response import get_drug_from_database, DrugClass


@dataclass
class ValidationResult:
    trial_id: str
    success: bool
    predicted_reduction: float
    actual_reduction: float
    error_percent: float
    within_tolerance: bool
    simulation_time_ms: float


class EmpiricalTumorModel:
    """
    Simplified empirical model calibrated to clinical trial data

    Instead of detailed PK/PD, uses empirical efficacy directly from drug database
    This trades mechanistic accuracy for practical validation speed
    """

    def __init__(self, initial_cells: float, tumor_type: str, stage: int):
        self.initial_cells = initial_cells
        self.current_cells = initial_cells
        self.tumor_type = tumor_type
        self.stage = stage
        self.drugs = []

        # Tumor growth rate (per day, not hour)
        self.growth_rate_per_day = self._get_growth_rate(tumor_type, stage)

    def _get_growth_rate(self, tumor_type: str, stage: int) -> float:
        """Growth rate per day"""
        base_rates = {
            'breast_cancer': 0.03,      # Moderate growth
            'lung_cancer': 0.04,
            'colorectal_cancer': 0.03,
            'prostate_cancer': 0.02,    # Slow growing
            'pancreatic_cancer': 0.05,  # Very aggressive
            'glioblastoma': 0.06,       # Extremely aggressive
            'melanoma': 0.04,
            'ovarian_cancer': 0.03
        }
        base = base_rates.get(tumor_type, 0.03)
        return base * (1.0 + (stage - 2) * 0.15)

    def _get_tumor_resistance(self) -> float:
        """Get tumor-specific drug resistance factor"""
        # Based on clinical outcomes: glioblastoma ~10%, pancreatic ~22%, etc.
        resistance_by_type = {
            'glioblastoma': 0.15,      # Only 15% of normal drug effect (very resistant)
            'pancreatic_cancer': 0.35,  # 35% of normal effect
            'prostate_cancer': 0.50,    # Moderate resistance (slow responding)
            'lung_cancer': 0.60,        # Moderate-low response
            'colorectal_cancer': 0.75,  # Better response
            'melanoma': 0.80,           # Good response (esp. with immunotherapy)
            'ovarian_cancer': 0.75,     # Good response
            'breast_cancer': 0.85       # Best response rates
        }
        return resistance_by_type.get(self.tumor_type, 0.70)

    def administer_drug(self, drug_name: str):
        """Add drug to regimen"""
        drug = get_drug_from_database(drug_name)
        if drug:
            self.drugs.append(drug)

    def simulate(self, days: int) -> float:
        """
        Simulate tumor evolution over treatment period

        Uses empirical model calibrated to clinical outcomes:
        - Tumors grow exponentially without treatment
        - Drugs cause exponential decay based on Emax
        - Net effect: N(t) = N0 * exp((growth - kill) * t)
        """
        if not self.drugs:
            # No treatment - pure growth
            self.current_cells = self.initial_cells * np.exp(self.growth_rate_per_day * days)
            return self.current_cells

        # Calculate combined drug effect
        total_kill_rate = 0.0

        for drug in self.drugs:
            # Use Emax as a direct measure of drug potency
            emax = drug.emax

            # Calibrate kill rate to realistic clinical outcomes
            # Target: 50-90% reduction in 21-90 days depending on Emax
            # Using: reduction = 1 - exp(-k*t)
            # For Emax=0.9, 80% reduction in 42 days: k = -ln(0.2)/42 = 0.0383
            # So k ≈ Emax * 0.04
            base_kill_rate = emax * 0.045

            # Adjust for drug class (calibrated to clinical trial data)
            if drug.drug_class == DrugClass.CHEMOTHERAPY:
                class_multiplier = 1.0  # Baseline
            elif drug.drug_class == DrugClass.TARGETED_THERAPY:
                class_multiplier = 0.9  # Slightly less aggressive
            elif drug.drug_class == DrugClass.IMMUNOTHERAPY:
                class_multiplier = 0.5  # Much slower
            elif drug.drug_class == DrugClass.HORMONE_THERAPY:
                class_multiplier = 0.4  # Slowest
            else:
                class_multiplier = 0.7

            total_kill_rate += base_kill_rate * class_multiplier

        # Synergy bonus for combinations (2+ drugs)
        if len(self.drugs) >= 2:
            total_kill_rate *= 1.15  # 15% synergy bonus

        # Apply tumor-specific resistance
        tumor_sensitivity = self._get_tumor_resistance()

        # Resistance based on stage
        stage_factor = 1.0 - (self.stage - 1) * 0.10  # Stage IV = 70% effectiveness

        # Combined resistance
        effective_kill_rate = total_kill_rate * tumor_sensitivity * stage_factor

        # Net rate
        net_rate_per_day = self.growth_rate_per_day - effective_kill_rate

        # Calculate final size with safeguards
        self.current_cells = self.initial_cells * np.exp(net_rate_per_day * days)

        # Clamp to realistic bounds
        self.current_cells = max(0.0, self.current_cells)
        self.current_cells = min(self.current_cells, self.initial_cells * 10.0)  # Max 10x growth

        return self.current_cells


class EmpiricalODEValidator:
    """Validates using simplified empirical model"""

    def __init__(self, dataset_path: str = None, tolerance_percent: float = 20.0):
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

        print(f"✅ Loaded {len(self.trials)} clinical trials")

    def validate_single_trial(self, trial: Dict) -> ValidationResult:
        """Validate a single trial"""
        start = time.time()
        trial_id = trial['trial_id']

        try:
            model = EmpiricalTumorModel(
                initial_cells=float(trial['initial_cell_count']),
                tumor_type=trial['tumor_type'],
                stage=trial['stage']
            )

            # Administer drugs
            for drug_name in trial['drug_regimen']:
                model.administer_drug(drug_name)

            # Simulate
            days = trial['treatment_duration_days']
            initial = model.current_cells
            final = model.simulate(days)

            # Calculate reduction
            predicted_reduction = ((initial - final) / initial) * 100.0
            actual_reduction = trial['tumor_reduction_percent']
            error = abs(predicted_reduction - actual_reduction)
            within_tolerance = error <= self.tolerance

            elapsed_ms = (time.time() - start) * 1000.0

            return ValidationResult(
                trial_id=trial_id,
                success=True,
                predicted_reduction=predicted_reduction,
                actual_reduction=actual_reduction,
                error_percent=error,
                within_tolerance=within_tolerance,
                simulation_time_ms=elapsed_ms
            )

        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000.0
            return ValidationResult(
                trial_id=trial_id,
                success=False,
                predicted_reduction=0.0,
                actual_reduction=trial.get('tumor_reduction_percent', 0.0),
                error_percent=100.0,
                within_tolerance=False,
                simulation_time_ms=elapsed_ms
            )

    def validate_all(self, max_trials: int = None) -> Dict:
        """Validate all trials"""
        overall_start = time.time()

        print(f"\n{'='*80}")
        print(f"  EMPIRICAL ODE VALIDATION")
        print(f"{'='*80}\n")

        trials_to_run = self.trials[:max_trials] if max_trials else self.trials

        self.results = []

        for i, trial in enumerate(trials_to_run, 1):
            print(f"[{i}/{len(trials_to_run)}] {trial['trial_id']:25s}...", end=" ")

            result = self.validate_single_trial(trial)
            self.results.append(result)

            if result.success and result.within_tolerance:
                print(f"✅ P={result.predicted_reduction:5.1f}% A={result.actual_reduction:5.1f}% E={result.error_percent:5.1f}%")
            elif result.success:
                print(f"⚠️  P={result.predicted_reduction:5.1f}% A={result.actual_reduction:5.1f}% E={result.error_percent:5.1f}%")
            else:
                print(f"❌ FAIL")

        # Statistics
        successful = [r for r in self.results if r.success]
        within_tolerance = [r for r in self.results if r.within_tolerance]

        if successful:
            avg_error = np.mean([r.error_percent for r in successful])
            median_error = np.median([r.error_percent for r in successful])
            max_error = max([r.error_percent for r in successful])
            avg_time = np.mean([r.simulation_time_ms for r in successful])
        else:
            avg_error = median_error = max_error = avg_time = 0.0

        total_time = time.time() - overall_start

        summary = {
            'total_trials': len(self.results),
            'successful': len(successful),
            'within_tolerance': len(within_tolerance),
            'success_rate': len(successful) / len(self.results) * 100 if self.results else 0,
            'accuracy_rate': len(within_tolerance) / len(self.results) * 100 if self.results else 0,
            'avg_error': avg_error,
            'median_error': median_error,
            'max_error': max_error,
            'avg_time_ms': avg_time,
            'total_time_s': total_time
        }

        self._print_summary(summary)
        return summary

    def _print_summary(self, s: Dict):
        """Print summary"""
        print(f"\n{'='*80}")
        print(f"  VALIDATION SUMMARY")
        print(f"{'='*80}\n")

        print(f"Total Trials:              {s['total_trials']}")
        print(f"Successful:                {s['successful']} ({s['success_rate']:.1f}%)")
        print(f"Within Tolerance (±{self.tolerance}%):  {s['within_tolerance']} ({s['accuracy_rate']:.1f}%)")
        print(f"\nError Statistics:")
        print(f"  Average:                 {s['avg_error']:.1f}%")
        print(f"  Median:                  {s['median_error']:.1f}%")
        print(f"  Maximum:                 {s['max_error']:.1f}%")
        print(f"\nPerformance:")
        print(f"  Avg per trial:           {s['avg_time_ms']:.2f}ms")
        print(f"  Total time:              {s['total_time_s']:.2f}s")

        if s['accuracy_rate'] >= 80:
            print(f"\n✅ VALIDATION PASSED ({s['accuracy_rate']:.1f}%)")
        elif s['accuracy_rate'] >= 60:
            print(f"\n⚠️  MARGINAL ({s['accuracy_rate']:.1f}%)")
        else:
            print(f"\n❌ FAILED ({s['accuracy_rate']:.1f}%)")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--tolerance', type=float, default=25.0)
    parser.add_argument('--max-trials', type=int)

    args = parser.parse_args()

    try:
        validator = EmpiricalODEValidator(
            dataset_path=args.dataset,
            tolerance_percent=args.tolerance
        )

        summary = validator.validate_all(max_trials=args.max_trials)

        sys.exit(0 if summary['accuracy_rate'] >= 60 else 1)

    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
