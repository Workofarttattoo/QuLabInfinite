#!/usr/bin/env python3
"""
Experimental Test Results Validation

Validates experimental measurements against database values with:
- Measurement uncertainty propagation
- Statistical significance testing
- Confidence intervals
- Outlier detection

Usage:
    python3 experimental_test_validation.py --material "Al 6061-T6" \
        --test-type tensile --measured-value 305 --uncertainty 5
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from materials_lab.materials_database import MaterialsDatabase, MaterialProperties


class ExperimentalValidator:
    """
    Validates experimental test results against database values
    """

    # Typical measurement uncertainties (%)
    TYPICAL_UNCERTAINTIES = {
        'tensile_strength': 3.0,  # ±3%
        'yield_strength': 3.0,
        'youngs_modulus': 5.0,
        'density': 0.5,
        'thermal_conductivity': 5.0,
        'hardness_vickers': 5.0,
        'elongation_at_break': 10.0,
    }

    def __init__(self, db: MaterialsDatabase):
        self.db = db

    def validate_measurement(
        self,
        material_name: str,
        property_name: str,
        measured_value: float,
        measurement_uncertainty: Optional[float] = None,
        num_samples: int = 1,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Validate experimental measurement against database value

        Args:
            material_name: Name of material tested
            property_name: Property measured (e.g., 'tensile_strength')
            measured_value: Measured value
            measurement_uncertainty: Measurement uncertainty (±)
            num_samples: Number of test samples
            confidence_level: Confidence level for intervals (0-1)

        Returns:
            Validation results dictionary
        """

        # Get material from database
        material = self.db.get_material(material_name)
        if not material:
            return {
                'valid': False,
                'error': f"Material '{material_name}' not found in database"
            }

        # Get database value
        db_value = getattr(material, property_name, None)
        if db_value is None or db_value == 0:
            return {
                'valid': False,
                'error': f"Property '{property_name}' not available for {material_name}"
            }

        # Estimate measurement uncertainty if not provided
        if measurement_uncertainty is None:
            typical_unc_pct = self.TYPICAL_UNCERTAINTIES.get(property_name, 5.0)
            measurement_uncertainty = measured_value * (typical_unc_pct / 100)

        # Calculate database uncertainty (assume ±5% if not specified)
        db_uncertainty = db_value * 0.05

        # Statistical analysis
        deviation = measured_value - db_value
        relative_error = (deviation / db_value) * 100

        # Combined uncertainty (root sum of squares)
        combined_uncertainty = np.sqrt(measurement_uncertainty**2 + db_uncertainty**2)

        # Calculate z-score
        z_score = abs(deviation) / combined_uncertainty

        # Determine statistical significance
        # For 95% confidence: z < 1.96 means not significantly different
        z_critical = stats.norm.ppf((1 + confidence_level) / 2)
        is_significant = z_score > z_critical

        # Calculate confidence interval for measured value
        # Account for sample size
        se = measurement_uncertainty / np.sqrt(num_samples) if num_samples > 1 else measurement_uncertainty
        ci_margin = z_critical * se
        ci_lower = measured_value - ci_margin
        ci_upper = measured_value + ci_margin

        # Check if database value falls within confidence interval
        db_in_ci = ci_lower <= db_value <= ci_upper

        # Overall assessment
        if not is_significant:
            assessment = "✓ PASSED - Measurement agrees with database (not statistically different)"
            status = "pass"
        elif db_in_ci:
            assessment = f"~ ACCEPTABLE - Deviation within expected uncertainty (±{relative_error:.1f}%)"
            status = "acceptable"
        elif abs(relative_error) < 10:
            assessment = f"⚠ WARNING - Moderate deviation ({relative_error:+.1f}%)"
            status = "warning"
        else:
            assessment = f"✗ FAILED - Significant deviation ({relative_error:+.1f}%)"
            status = "fail"

        return {
            'valid': True,
            'material': material_name,
            'property': property_name,
            'measured_value': measured_value,
            'database_value': db_value,
            'deviation': deviation,
            'relative_error_percent': relative_error,
            'measurement_uncertainty': measurement_uncertainty,
            'database_uncertainty': db_uncertainty,
            'combined_uncertainty': combined_uncertainty,
            'z_score': z_score,
            'z_critical': z_critical,
            'statistically_significant': is_significant,
            'confidence_interval': (ci_lower, ci_upper),
            'database_in_ci': db_in_ci,
            'assessment': assessment,
            'status': status,
            'num_samples': num_samples,
            'confidence_level': confidence_level
        }

    def print_validation_result(self, result: Dict):
        """Print formatted validation result"""

        if not result['valid']:
            print(f"\n❌ Error: {result['error']}")
            return

        print("\n" + "=" * 80)
        print("EXPERIMENTAL TEST VALIDATION")
        print("=" * 80)

        print(f"\nMaterial: {result['material']}")
        print(f"Property: {result['property']}")
        print(f"Samples:  {result['num_samples']}")
        print(f"Confidence Level: {result['confidence_level']*100:.0f}%")

        print("\n" + "-" * 80)
        print("MEASURED vs DATABASE")
        print("-" * 80)

        print(f"\nMeasured Value:     {result['measured_value']:.2f} ± {result['measurement_uncertainty']:.2f}")
        print(f"Database Value:     {result['database_value']:.2f} ± {result['database_uncertainty']:.2f}")
        print(f"Deviation:          {result['deviation']:+.2f} ({result['relative_error_percent']:+.1f}%)")

        print("\n" + "-" * 80)
        print("STATISTICAL ANALYSIS")
        print("-" * 80)

        print(f"\nCombined Uncertainty: ±{result['combined_uncertainty']:.2f}")
        print(f"Z-Score:              {result['z_score']:.2f}")
        print(f"Z-Critical (95%):     {result['z_critical']:.2f}")
        print(f"Significant?:         {'Yes' if result['statistically_significant'] else 'No'}")

        ci_low, ci_high = result['confidence_interval']
        print(f"\n{result['confidence_level']*100:.0f}% Confidence Interval:")
        print(f"  [{ci_low:.2f}, {ci_high:.2f}]")
        print(f"  Database value in CI? {'Yes ✓' if result['database_in_ci'] else 'No ✗'}")

        print("\n" + "=" * 80)
        print("ASSESSMENT")
        print("=" * 80)
        print(f"\n{result['assessment']}")

        # Recommendations
        print("\n" + "-" * 80)
        print("RECOMMENDATIONS")
        print("-" * 80)

        if result['status'] == 'pass':
            print("\n✓ Measurement validated successfully")
            print("  - Result is consistent with database value")
            print("  - No further action needed")

        elif result['status'] == 'acceptable':
            print("\n~ Measurement acceptable")
            print("  - Deviation is within expected measurement uncertainty")
            print("  - Consider: More samples for improved precision")

        elif result['status'] == 'warning':
            print("\n⚠ Moderate deviation detected")
            print("  - Recommended actions:")
            print("    1. Verify measurement procedure and calibration")
            print("    2. Increase sample size (currently n={})".format(result['num_samples']))
            print("    3. Check material specification/batch")
            print("    4. Review test conditions (temperature, humidity, etc.)")

        else:  # fail
            print("\n✗ Significant deviation")
            print("  - Required actions:")
            print("    1. CRITICAL: Verify material identity and batch")
            print("    2. Recalibrate measurement equipment")
            print("    3. Repeat test with new samples (n ≥ 3)")
            print("    4. Consider material substitution or database update")

        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Validate experimental test results')
    parser.add_argument('--material', required=True, help='Material name')
    parser.add_argument('--property', required=True, help='Property tested (e.g., tensile_strength)')
    parser.add_argument('--measured', type=float, required=True, help='Measured value')
    parser.add_argument('--uncertainty', type=float, help='Measurement uncertainty (±)')
    parser.add_argument('--samples', type=int, default=1, help='Number of test samples')
    parser.add_argument('--confidence', type=float, default=0.95, help='Confidence level (default 0.95)')

    args = parser.parse_args()

    print("QuLabInfinite - Experimental Test Validation")
    print("=" * 80)

    # Load database
    print("\nLoading materials database...")
    db = MaterialsDatabase()
    print(f"✓ Loaded {db.get_count()} materials")

    # Validate measurement
    validator = ExperimentalValidator(db)

    result = validator.validate_measurement(
        material_name=args.material,
        property_name=args.property,
        measured_value=args.measured,
        measurement_uncertainty=args.uncertainty,
        num_samples=args.samples,
        confidence_level=args.confidence
    )

    validator.print_validation_result(result)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # Demo mode
        print("QuLabInfinite - Experimental Test Validation (DEMO)")
        print("=" * 80)

        db = MaterialsDatabase()
        validator = ExperimentalValidator(db)

        # Example 1: Good agreement
        print("\n\nEXAMPLE 1: Tensile test with good agreement")
        result = validator.validate_measurement(
            material_name="Al 6061-T6",
            property_name="tensile_strength",
            measured_value=305,  # Database: 310 MPa
            measurement_uncertainty=8,
            num_samples=3
        )
        validator.print_validation_result(result)

        # Example 2: Moderate deviation
        print("\n\nEXAMPLE 2: Density measurement with moderate deviation")
        result = validator.validate_measurement(
            material_name="Ti-6Al-4V",
            property_name="density",
            measured_value=4520,  # Database: 4430 kg/m³
            measurement_uncertainty=20,
            num_samples=5
        )
        validator.print_validation_result(result)

        # Example 3: Significant deviation
        print("\n\nEXAMPLE 3: Young's modulus with significant deviation")
        result = validator.validate_measurement(
            material_name="SS 304",
            property_name="youngs_modulus",
            measured_value=220,  # Database: 193 GPa
            measurement_uncertainty=10,
            num_samples=1
        )
        validator.print_validation_result(result)

        print("\n" + "=" * 80)
        print("Run with --help for usage with real test data")
        print("=" * 80)

    else:
        main()
