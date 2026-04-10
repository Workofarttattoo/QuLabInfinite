#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Fast ODE-Based Clinical Trial Validator
Uses continuous differential equation model instead of agent-based for 100-10,000x speedup
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import sys

from .drug_response import get_drug_from_database, DrugClass


@dataclass
class ODEValidationResult:
    """Results from validating a single trial using ODE model"""
    trial_id: str
    success: bool
    predicted_reduction: float
    actual_reduction: float
    error_percent: float
    within_tolerance: bool
    details: str
    simulation_time_ms: float


class TumorODEModel:
    """
    Continuous ODE model for tumor growth and drug response

    Uses:
    - Gompertzian growth: dN/dt = r*N*log(K/N)
    - Multi-drug Hill equation: Kill = sum(kill_i * C_i^n / (EC50_i^n + C_i^n))
    - Exponential PK decay: C(t) = C0 * exp(-k*t)
    """

    def __init__(self, initial_cells: float, tumor_type: str, stage: int):
        self.cell_count = initial_cells
        self.tumor_type = tumor_type
        self.stage = stage

        # Tumor growth parameters (calibrated to agent model)
        self.growth_rate = self._get_growth_rate(tumor_type, stage)
        self.carrying_capacity = initial_cells * 50.0  # ~50x growth potential

        # Drug concentrations and parameters
        self.drugs = []  # List of (drug_object, concentration, time_administered)

    def _get_growth_rate(self, tumor_type: str, stage: int) -> float:
        """Get intrinsic growth rate based on tumor type and stage"""
        # Base rates (per hour)
        base_rates = {
            'breast_cancer': 0.0020,      # ~5-10x growth in 30 days
            'lung_cancer': 0.0025,         # Slightly faster
            'colorectal_cancer': 0.0018,
            'prostate_cancer': 0.0015,     # Slower growing
            'pancreatic_cancer': 0.0030,   # Very aggressive
            'glioblastoma': 0.0035,        # Extremely aggressive
            'melanoma': 0.0028,
            'ovarian_cancer': 0.0022
        }

        base_rate = base_rates.get(tumor_type, 0.0020)

        # Stage affects growth (higher stage = faster, more aggressive)
        stage_multiplier = 1.0 + (stage - 2) * 0.15  # Stage II baseline

        return base_rate * stage_multiplier

    def administer_drug(self, drug_name: str, dose_mg: float, time: float):
        """Administer a drug at a specific time"""
        drug = get_drug_from_database(drug_name)
        if not drug:
            return

        # Calculate initial concentration (Cmax)
        # For IV drugs (bioavailability < 0.2), assume actual IV administration with 100% bioavailability
        # For oral drugs, use actual bioavailability
        bioavail = drug.pk_model.bioavailability

        # Most chemotherapy is given IV - if bioavailability is very low, it's likely an IV drug
        if bioavail < 0.2:
            bioavail = 1.0  # Assume IV administration for chemo drugs

        vd = drug.pk_model.volume_of_distribution

        if vd > 0:
            c0 = (dose_mg * bioavail) / vd  # mg/L
        else:
            c0 = dose_mg * bioavail  # Fallback

        self.drugs.append({
            'drug': drug,
            'c0': c0,  # Initial concentration (mg/L)
            'time_start': time,
            'elimination_rate': 0.693 / drug.pk_model.half_life  # k = ln(2)/t_half
        })

    def _get_drug_concentration(self, drug_info: Dict, time: float) -> float:
        """Calculate drug concentration at given time using exponential decay"""
        elapsed = time - drug_info['time_start']
        if elapsed < 0:
            return 0.0

        c0 = drug_info['c0']
        k = drug_info['elimination_rate']

        # C(t) = C0 * exp(-k*t)
        return c0 * np.exp(-k * elapsed)

    def _calculate_growth_rate(self, time: float) -> float:
        """Calculate current growth rate (Gompertzian)"""
        if self.cell_count <= 0 or self.cell_count >= self.carrying_capacity:
            return 0.0

        # dN/dt = r * N * log(K/N)
        return self.growth_rate * self.cell_count * np.log(self.carrying_capacity / self.cell_count)

    def _calculate_kill_rate(self, time: float) -> float:
        """Calculate current drug-induced kill rate (multi-drug Hill equation)"""
        total_kill = 0.0

        for drug_info in self.drugs:
            drug = drug_info['drug']
            concentration = self._get_drug_concentration(drug_info, time)

            if concentration <= 0:
                continue

            # Convert concentration from mg/L to μM for IC50/EC50 comparison
            # concentration_uM = (concentration_mg_L * 1000) / molecular_weight
            conc_uM = (concentration * 1000.0) / drug.molecular_weight

            # Hill equation: Effect = Emax * C^n / (EC50^n + C^n)
            ec50 = drug.ec50  # μM
            n = drug.hill_coefficient
            emax = drug.emax

            # Drug effect (fractional kill rate)
            numerator = conc_uM ** n
            denominator = (ec50 ** n) + numerator

            if denominator > 0:
                effect = emax * (numerator / denominator)

                # Convert effect to kill rate (per hour)
                # Target: 50-90% kill in 21 days (504 hours)
                # If we want 65% reduction: N_final = N_initial * 0.35
                # Exponential decay: N(t) = N0 * exp(-k*t)
                # 0.35 = exp(-k * 504) => k = -ln(0.35)/504 = 0.00209
                # So kill rate ≈ 0.002 * N for ~65% reduction
                #
                # With effect ranging 0-1, we need: kill_rate = effect * constant * N
                # To get k ≈ 0.002 when effect ≈ 0.7: constant = 0.002 / 0.7 ≈ 0.003

                base_kill_constant = 0.006  # Recalibrated for realistic kill rates

                # Adjust for drug class
                if drug.drug_class == DrugClass.CHEMOTHERAPY:
                    kill_multiplier = 1.5  # Strong cytotoxic effect
                elif drug.drug_class == DrugClass.TARGETED_THERAPY:
                    kill_multiplier = 1.2  # Moderate but specific
                elif drug.drug_class == DrugClass.IMMUNOTHERAPY:
                    kill_multiplier = 0.8  # Slower but sustained
                else:
                    kill_multiplier = 1.0

                kill_rate = effect * base_kill_constant * kill_multiplier * self.cell_count
                total_kill += kill_rate

        return total_kill

    def step(self, dt: float, current_time: float) -> float:
        """
        Advance simulation by dt hours

        Args:
            dt: Time step in hours
            current_time: Current simulation time

        Returns:
            New cell count
        """
        if self.cell_count <= 0:
            return 0.0

        # Calculate rates
        growth = self._calculate_growth_rate(current_time)
        kill = self._calculate_kill_rate(current_time)

        # dN/dt = growth - kill
        dN_dt = growth - kill

        # Euler integration (simple but sufficient for validation)
        new_count = self.cell_count + (dN_dt * dt)

        # Ensure non-negative
        self.cell_count = max(0.0, new_count)

        return self.cell_count


class FastODEValidator:
    """Validates oncology lab using fast ODE model instead of agent-based"""

    def __init__(self, dataset_path: str = None, tolerance_percent: float = 20.0):
        """
        Initialize validator

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

        print(f"✅ Loaded {len(self.trials)} clinical trials")

    def validate_single_trial(self, trial: Dict) -> ODEValidationResult:
        """
        Validate a single clinical trial using ODE model

        Args:
            trial: Trial data dictionary

        Returns:
            ODEValidationResult with comparison metrics
        """
        import time
        start_time = time.time()

        trial_id = trial['trial_id']

        try:
            # Initialize ODE model
            model = TumorODEModel(
                initial_cells=float(trial['initial_cell_count']),
                tumor_type=trial['tumor_type'],
                stage=trial['stage']
            )

            initial_cells = model.cell_count

            # Administer drugs at t=0
            for drug_name in trial['drug_regimen']:
                drug = get_drug_from_database(drug_name)
                if drug:
                    model.administer_drug(drug_name, dose_mg=drug.standard_dose_mg, time=0.0)

            # Simulate treatment duration
            treatment_hours = trial['treatment_duration_days'] * 24.0
            dt = 24.0  # 24-hour timesteps (ODE can handle larger steps)
            current_time = 0.0

            while current_time < treatment_hours:
                model.step(dt, current_time)
                current_time += dt

            final_cells = model.cell_count

            # Calculate reduction
            if initial_cells > 0:
                predicted_reduction = ((initial_cells - final_cells) / initial_cells) * 100.0
            else:
                predicted_reduction = 0.0

            actual_reduction = trial['tumor_reduction_percent']
            error = abs(predicted_reduction - actual_reduction)
            within_tolerance = error <= self.tolerance

            elapsed_ms = (time.time() - start_time) * 1000.0

            return ODEValidationResult(
                trial_id=trial_id,
                success=True,
                predicted_reduction=predicted_reduction,
                actual_reduction=actual_reduction,
                error_percent=error,
                within_tolerance=within_tolerance,
                details="ODE simulation completed",
                simulation_time_ms=elapsed_ms
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000.0
            return ODEValidationResult(
                trial_id=trial_id,
                success=False,
                predicted_reduction=0.0,
                actual_reduction=trial.get('tumor_reduction_percent', 0.0),
                error_percent=100.0,
                within_tolerance=False,
                details=f"Simulation error: {str(e)}",
                simulation_time_ms=elapsed_ms
            )

    def validate_all(self, max_trials: int = None) -> Dict:
        """
        Validate all trials in dataset

        Args:
            max_trials: Maximum number of trials to run (None for all)

        Returns:
            Summary statistics dictionary
        """
        import time
        overall_start = time.time()

        print(f"\n{'='*80}")
        print(f"  FAST ODE VALIDATION: {len(self.trials)} CLINICAL TRIALS")
        print(f"{'='*80}\n")

        trials_to_run = self.trials[:max_trials] if max_trials else self.trials

        self.results = []

        for i, trial in enumerate(trials_to_run, 1):
            print(f"[{i}/{len(trials_to_run)}] {trial['trial_id']}...", end=" ")

            result = self.validate_single_trial(trial)
            self.results.append(result)

            if result.success and result.within_tolerance:
                print(f"✅ PASS (Error: {result.error_percent:.1f}%, {result.simulation_time_ms:.1f}ms)")
            elif result.success:
                print(f"⚠️  HIGH ERROR ({result.error_percent:.1f}%, {result.simulation_time_ms:.1f}ms)")
            else:
                print(f"❌ FAIL - {result.details}")

        # Calculate statistics
        successful = [r for r in self.results if r.success]
        within_tolerance = [r for r in self.results if r.within_tolerance]

        if successful:
            avg_error = np.mean([r.error_percent for r in successful])
            median_error = np.median([r.error_percent for r in successful])
            max_error = max([r.error_percent for r in successful])
            avg_time_ms = np.mean([r.simulation_time_ms for r in successful])
        else:
            avg_error = median_error = max_error = avg_time_ms = 0.0

        total_time = time.time() - overall_start

        summary = {
            'total_trials': len(self.results),
            'successful_simulations': len(successful),
            'within_tolerance': len(within_tolerance),
            'success_rate': len(successful) / len(self.results) * 100 if self.results else 0,
            'accuracy_rate': len(within_tolerance) / len(self.results) * 100 if self.results else 0,
            'average_error_percent': avg_error,
            'median_error_percent': median_error,
            'max_error_percent': max_error,
            'tolerance_used': self.tolerance,
            'avg_simulation_time_ms': avg_time_ms,
            'total_time_seconds': total_time
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
        print(f"\nPerformance:")
        print(f"  Avg Time per Trial:      {summary['avg_simulation_time_ms']:.1f}ms")
        print(f"  Total Validation Time:   {summary['total_time_seconds']:.2f}s")
        print(f"  Throughput:              {summary['total_trials'] / summary['total_time_seconds']:.1f} trials/sec")

        if summary['accuracy_rate'] >= 80:
            print(f"\n✅ VALIDATION PASSED - Accuracy: {summary['accuracy_rate']:.1f}%")
        elif summary['accuracy_rate'] >= 60:
            print(f"\n⚠️  VALIDATION MARGINAL - Accuracy: {summary['accuracy_rate']:.1f}%")
        else:
            print(f"\n❌ VALIDATION FAILED - Accuracy: {summary['accuracy_rate']:.1f}%")

    def export_results(self, output_path: str = None):
        """Export validation results to JSON"""
        if output_path is None:
            output_path = Path(__file__).parent / "ode_validation_results.json"

        data = {
            'metadata': self.metadata,
            'model_type': 'ODE',
            'tolerance_percent': self.tolerance,
            'results': [
                {
                    'trial_id': r.trial_id,
                    'success': r.success,
                    'predicted_reduction': r.predicted_reduction,
                    'actual_reduction': r.actual_reduction,
                    'error_percent': r.error_percent,
                    'within_tolerance': r.within_tolerance,
                    'simulation_time_ms': r.simulation_time_ms,
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

    parser = argparse.ArgumentParser(description="Fast ODE-based clinical trial validator")
    parser.add_argument('--dataset', type=str, help="Path to clinical trial dataset JSON")
    parser.add_argument('--tolerance', type=float, default=20.0, help="Error tolerance percent (default: 20)")
    parser.add_argument('--max-trials', type=int, help="Maximum trials to run")
    parser.add_argument('--export', type=str, help="Export results to JSON file")

    args = parser.parse_args()

    try:
        validator = FastODEValidator(
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
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
