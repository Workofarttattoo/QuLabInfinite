"""
Results Validation System - 100% Real-World Accuracy Verification

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Ensures QuLabInfinite simulations match real-world experimental data within uncertainty bounds.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
try:
    from scipy import stats  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    stats = None
import json
from pathlib import Path


class ValidationStatus(Enum):
    """Validation result status."""
    PASS = "pass"
    WARN = "warning"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass
class ReferenceData:
    """Reference experimental data for validation."""
    source: str  # NIST, Materials Project, literature DOI
    value: float
    uncertainty: float
    units: str
    conditions: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class ValidationResult:
    """Result of validation against reference data."""
    status: ValidationStatus
    error_percent: float
    z_score: float
    simulated_value: float
    reference_value: float
    uncertainty: float
    message: str
    passed_tests: List[str]
    failed_tests: List[str]


class ResultsValidator:
    """
    Validate simulation results against experimental reference data.

    Ensures <1% error on well-characterized materials and phenomena.
    """

    def __init__(self, reference_db_path: Optional[str] = None):
        """Initialize validator with reference database."""
        self.reference_db_path = reference_db_path or self._default_db_path()
        self.reference_db = self._load_reference_database()

    def _default_db_path(self) -> str:
        """Get default reference database path."""
        return str(Path(__file__).parent / "reference_data.json")

    def _load_reference_database(self) -> Dict[str, ReferenceData]:
        """Load reference experimental data."""
        if not Path(self.reference_db_path).exists():
            raise FileNotFoundError(f"Reference database not found at {self.reference_db_path}")

        try:
            with open(self.reference_db_path, 'r') as f:
                data = json.load(f)
            
            # Convert to ReferenceData objects
            db = {}
            for key, value in data.items():
                db[key] = ReferenceData(**value)
            return db
        except Exception as e:
            print(f"Warning: Could not load reference database: {e}")
            return {}

    def validate(self,
                simulated_value: float,
                reference_key: str,
                tolerance_sigma: float = 3.0,
                max_error_percent: float = 1.0) -> ValidationResult:
        """
        Validate simulated value against reference data.

        Args:
            simulated_value: Value from simulation
            reference_key: Key in reference database
            tolerance_sigma: Number of standard deviations for pass
            max_error_percent: Maximum allowed percent error

        Returns:
            ValidationResult with status and diagnostics
        """
        if reference_key not in self.reference_db:
            return ValidationResult(
                status=ValidationStatus.UNKNOWN,
                error_percent=np.nan,
                z_score=np.nan,
                simulated_value=simulated_value,
                reference_value=np.nan,
                uncertainty=np.nan,
                message=f"No reference data for '{reference_key}'",
                passed_tests=[],
                failed_tests=["reference_exists"]
            )

        ref = self.reference_db[reference_key]

        # Calculate error metrics
        error = abs(simulated_value - ref.value)
        error_percent = (error / abs(ref.value)) * 100 if ref.value != 0 else np.inf

        # Calculate z-score (how many standard deviations away)
        if ref.uncertainty > 0:
            z_score = error / ref.uncertainty
        else:
            z_score = 0.0 if error == 0 else np.inf

        # Run validation tests
        passed_tests = []
        failed_tests = []

        # Test 1: Within tolerance sigma
        if z_score <= tolerance_sigma:
            passed_tests.append("sigma_tolerance")
        else:
            failed_tests.append("sigma_tolerance")

        # Test 2: Within max percent error
        if error_percent <= max_error_percent:
            passed_tests.append("percent_error")
        else:
            failed_tests.append("percent_error")

        # Determine overall status
        if len(failed_tests) == 0:
            status = ValidationStatus.PASS
            message = f"Validation passed: {error_percent:.4f}% error, {z_score:.2f}σ"
        elif len(passed_tests) > 0:
            status = ValidationStatus.WARN
            message = f"Validation warning: {error_percent:.4f}% error, {z_score:.2f}σ"
        else:
            status = ValidationStatus.FAIL
            message = f"Validation failed: {error_percent:.4f}% error, {z_score:.2f}σ"

        return ValidationResult(
            status=status,
            error_percent=error_percent,
            z_score=z_score,
            simulated_value=simulated_value,
            reference_value=ref.value,
            uncertainty=ref.uncertainty,
            message=message,
            passed_tests=passed_tests,
            failed_tests=failed_tests
        )

    def validate_array(self,
                      simulated_values: np.ndarray,
                      reference_values: np.ndarray,
                      uncertainties: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Validate array of values against reference array.

        Returns:
            Dictionary with statistical metrics
        """
        if uncertainties is None:
            uncertainties = np.zeros_like(reference_values)

        errors = np.abs(simulated_values - reference_values)
        percent_errors = (errors / np.abs(reference_values)) * 100

        # Calculate metrics
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(errors**2))
        max_error = np.max(errors)
        mean_percent_error = np.mean(percent_errors)

        # Correlation
        correlation = np.corrcoef(simulated_values, reference_values)[0, 1]

        # R² score
        ss_res = np.sum((reference_values - simulated_values)**2)
        ss_tot = np.sum((reference_values - np.mean(reference_values))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Chi-squared test (if uncertainties available)
        if np.any(uncertainties > 0):
            chi_squared = np.sum(((simulated_values - reference_values) / uncertainties)**2)
            reduced_chi_squared = chi_squared / len(simulated_values)
        else:
            chi_squared = np.nan
            reduced_chi_squared = np.nan

        return {
            "mae": mae,
            "rmse": rmse,
            "max_error": max_error,
            "mean_percent_error": mean_percent_error,
            "correlation": correlation,
            "r_squared": r_squared,
            "chi_squared": chi_squared,
            "reduced_chi_squared": reduced_chi_squared,
            "n_points": len(simulated_values)
        }

    def benchmark_suite(self, simulator_func, test_keys: List[str]) -> Dict[str, ValidationResult]:
        """
        Run benchmark suite on multiple reference cases.

        Args:
            simulator_func: Function that takes reference_key and returns simulated value
            test_keys: List of reference keys to test

        Returns:
            Dictionary of ValidationResults for each key
        """
        results = {}
        for key in test_keys:
            try:
                simulated_value = simulator_func(key)
                results[key] = self.validate(simulated_value, key)
            except Exception as e:
                results[key] = ValidationResult(
                    status=ValidationStatus.FAIL,
                    error_percent=np.nan,
                    z_score=np.nan,
                    simulated_value=np.nan,
                    reference_value=np.nan,
                    uncertainty=np.nan,
                    message=f"Simulation error: {e}",
                    passed_tests=[],
                    failed_tests=["simulation_execution"]
                )

        return results

    def generate_report(self, results: Dict[str, ValidationResult]) -> str:
        """Generate human-readable validation report."""
        report = []
        report.append("=" * 80)
        report.append("QuLabInfinite Validation Report")
        report.append("=" * 80)
        report.append("")

        # Summary statistics
        total = len(results)
        passed = sum(1 for r in results.values() if r.status == ValidationStatus.PASS)
        warned = sum(1 for r in results.values() if r.status == ValidationStatus.WARN)
        failed = sum(1 for r in results.values() if r.status == ValidationStatus.FAIL)

        report.append(f"Total tests: {total}")
        report.append(f"Passed: {passed} ({passed/total*100:.1f}%)")
        report.append(f"Warnings: {warned} ({warned/total*100:.1f}%)")
        report.append(f"Failed: {failed} ({failed/total*100:.1f}%)")
        report.append("")

        # Individual results
        report.append("Individual Test Results:")
        report.append("-" * 80)
        for key, result in results.items():
            status_symbol = "✓" if result.status == ValidationStatus.PASS else "⚠" if result.status == ValidationStatus.WARN else "✗"
            report.append(f"{status_symbol} {key}:")
            report.append(f"  Status: {result.status.value}")
            report.append(f"  Error: {result.error_percent:.4f}%")
            report.append(f"  Z-score: {result.z_score:.2f}")
            report.append(f"  Simulated: {result.simulated_value:.6e}")
            report.append(f"  Reference: {result.reference_value:.6e} ± {result.uncertainty:.6e}")
            report.append(f"  {result.message}")
            report.append("")

        report.append("=" * 80)
        return "\n".join(report)


if __name__ == "__main__":
    # Demo validation system
    validator = ResultsValidator()

    print("QuLabInfinite Results Validation System")
    print("=" * 80)
    print(f"Loaded {len(validator.reference_db)} reference data points")
    print()

    # Test validation of a physics constant
    print("Test 1: Speed of light")
    result = validator.validate(299792458.0, "speed_of_light")
    print(f"  Status: {result.status.value}")
    print(f"  Error: {result.error_percent:.6f}%")
    print(f"  Message: {result.message}")
    print()

    # Test validation of material property
    print("Test 2: Steel 304 yield strength")
    result = validator.validate(217.5, "steel_304_yield_strength")
    print(f"  Status: {result.status.value}")
    print(f"  Error: {result.error_percent:.4f}%")
    print(f"  Message: {result.message}")
    print()

    # Test array validation
    print("Test 3: Array validation")
    sim_values = np.array([1.0, 2.1, 2.9, 4.2, 5.0])
    ref_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    metrics = validator.validate_array(sim_values, ref_values)
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²: {metrics['r_squared']:.4f}")
    print(f"  Mean % error: {metrics['mean_percent_error']:.4f}%")
