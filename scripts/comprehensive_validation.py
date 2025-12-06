#!/usr/bin/env python3
"""
Comprehensive Materials Database Validation

Tests database accuracy with:
- Physical consistency checks (property relationships)
- Cross-validation against known references
- Statistical outlier detection
- Uncertainty quantification
- Confidence scoring with ±% error margins

Usage:
    python3 scripts/comprehensive_validation.py [--full] [--save-report]

Options:
    --full          Validate all materials (default: sample of 100)
    --save-report   Save detailed JSON report
    --verbose       Show all validation issues
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from materials_lab.materials_database import MaterialsDatabase, MaterialProperties


@dataclass
class ValidationIssue:
    """Single validation issue with severity and details"""
    severity: str  # 'critical', 'error', 'warning', 'info'
    category: str  # 'consistency', 'outlier', 'missing', 'bounds', 'physics'
    property_name: str
    message: str
    expected_value: Optional[float] = None
    actual_value: Optional[float] = None
    deviation_percent: Optional[float] = None
    confidence_impact: float = 0.0  # How much this reduces confidence (0-1)

    def __str__(self):
        severity_symbols = {
            'critical': '🔴',
            'error': '❌',
            'warning': '⚠️ ',
            'info': 'ℹ️ '
        }
        symbol = severity_symbols.get(self.severity, '•')

        msg = f"{symbol} [{self.severity.upper()}] {self.message}"
        if self.deviation_percent is not None:
            msg += f" (deviation: {self.deviation_percent:.1f}%)"
        return msg


@dataclass
class MaterialScore:
    """Validation score for a material with uncertainty margins"""
    material_name: str
    overall_confidence: float  # 0-100%
    accuracy_score: float  # 0-100%
    completeness_score: float  # 0-100%
    consistency_score: float  # 0-100%
    issues: List[ValidationIssue]
    uncertainty_estimates: Dict[str, float]  # property -> ±% uncertainty

    def grade(self) -> str:
        """Return letter grade"""
        if self.overall_confidence >= 95:
            return 'A+'
        elif self.overall_confidence >= 90:
            return 'A'
        elif self.overall_confidence >= 85:
            return 'A-'
        elif self.overall_confidence >= 80:
            return 'B+'
        elif self.overall_confidence >= 75:
            return 'B'
        elif self.overall_confidence >= 70:
            return 'B-'
        elif self.overall_confidence >= 65:
            return 'C+'
        elif self.overall_confidence >= 60:
            return 'C'
        else:
            return 'D' if self.overall_confidence >= 50 else 'F'


class ComprehensiveValidator:
    """
    Comprehensive database validator with physical consistency checks
    """

    # Known reference values (NIST, CRC Handbook, peer-reviewed literature)
    REFERENCE_VALUES = {
        'Silicon': {'density': (2329, 1), 'youngs_modulus': (130, 5), 'thermal_conductivity': (149, 3), 'melting_point': (1687, 2)},
        'Copper': {'density': (8960, 5), 'youngs_modulus': (130, 3), 'thermal_conductivity': (401, 2), 'melting_point': (1358, 1)},
        'Aluminum': {'density': (2700, 5), 'youngs_modulus': (70, 2), 'thermal_conductivity': (237, 3), 'melting_point': (933, 1)},
        'Iron': {'density': (7874, 5), 'youngs_modulus': (211, 3), 'thermal_conductivity': (80, 2), 'melting_point': (1811, 2)},
        'Titanium': {'density': (4506, 5), 'youngs_modulus': (116, 3), 'thermal_conductivity': (22, 1), 'melting_point': (1941, 2)},
        'Gold': {'density': (19320, 10), 'youngs_modulus': (79, 2), 'thermal_conductivity': (318, 5), 'melting_point': (1337, 1)},
        'Silver': {'density': (10490, 10), 'youngs_modulus': (83, 2), 'thermal_conductivity': (429, 5), 'melting_point': (1235, 1)},
        'Tungsten': {'density': (19250, 20), 'youngs_modulus': (411, 10), 'thermal_conductivity': (173, 5), 'melting_point': (3695, 10)},
        'Graphene': {'density': (2267, 50), 'youngs_modulus': (1000, 50), 'thermal_conductivity': (5000, 500)},
        'Diamond': {'density': (3515, 5), 'youngs_modulus': (1050, 20), 'thermal_conductivity': (2200, 100), 'melting_point': (3823, 50)},
    }

    def __init__(self, db: MaterialsDatabase):
        self.db = db

    def validate_material(self, material: MaterialProperties) -> MaterialScore:
        """Comprehensive validation of a single material"""
        issues = []
        uncertainty = {}

        # 1. Physical bounds checking
        issues.extend(self._check_physical_bounds(material, uncertainty))

        # 2. Property relationship consistency
        issues.extend(self._check_property_relationships(material, uncertainty))

        # 3. Cross-validation against references
        issues.extend(self._cross_validate_references(material, uncertainty))

        # 4. Statistical outlier detection
        issues.extend(self._detect_outliers(material))

        # 5. Completeness check
        completeness_issues, completeness_score = self._check_completeness(material)
        issues.extend(completeness_issues)

        # Calculate scores
        accuracy_score = self._calculate_accuracy_score(issues)
        consistency_score = self._calculate_consistency_score(issues)

        # Overall confidence (weighted average)
        overall_confidence = (
            accuracy_score * 0.4 +
            consistency_score * 0.3 +
            completeness_score * 0.3
        )

        return MaterialScore(
            material_name=material.name,
            overall_confidence=overall_confidence,
            accuracy_score=accuracy_score,
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            issues=issues,
            uncertainty_estimates=uncertainty
        )

    def _check_physical_bounds(self, mat: MaterialProperties, uncertainty: Dict) -> List[ValidationIssue]:
        """Check if values are within physical bounds"""
        issues = []

        # Density: 0.1 kg/m³ (H₂ gas) to 22,590 kg/m³ (Osmium)
        if mat.density > 0:
            if mat.density < 0.05:
                issues.append(ValidationIssue('warning', 'bounds', 'density',
                    f'Density {mat.density:.2f} kg/m³ extremely low (lighter than hydrogen gas)',
                    actual_value=mat.density, confidence_impact=0.1))
            elif mat.density > 23000:
                issues.append(ValidationIssue('error', 'bounds', 'density',
                    f'Density {mat.density:.0f} kg/m³ exceeds osmium (22,590 kg/m³)',
                    actual_value=mat.density, expected_value=22590, confidence_impact=0.2))
            # Estimate uncertainty based on density range
            if mat.density < 1000:
                uncertainty['density'] = 5.0  # ±5% for low density
            else:
                uncertainty['density'] = 2.0  # ±2% for typical materials

        # Poisson's ratio: -1 ≤ ν ≤ 0.5 (theoretical bounds)
        if mat.poissons_ratio != 0:
            if mat.poissons_ratio < -1 or mat.poissons_ratio > 0.5:
                issues.append(ValidationIssue('critical', 'bounds', 'poissons_ratio',
                    f'Poisson ratio {mat.poissons_ratio:.3f} violates thermodynamic bounds [-1, 0.5]',
                    actual_value=mat.poissons_ratio, confidence_impact=0.3))
            elif mat.poissons_ratio < 0:
                issues.append(ValidationIssue('info', 'bounds', 'poissons_ratio',
                    f'Negative Poisson ratio {mat.poissons_ratio:.3f} (auxetic material)',
                    actual_value=mat.poissons_ratio))
            uncertainty['poissons_ratio'] = 5.0  # ±5%

        # Thermal conductivity: must be positive
        if mat.thermal_conductivity < 0:
            issues.append(ValidationIssue('critical', 'bounds', 'thermal_conductivity',
                f'Negative thermal conductivity: {mat.thermal_conductivity:.3f} W/(m·K)',
                actual_value=mat.thermal_conductivity, confidence_impact=0.3))
        elif mat.thermal_conductivity > 0:
            # Diamond has highest at ~2200 W/(m·K), graphene ~5000
            if mat.thermal_conductivity > 5500:
                issues.append(ValidationIssue('warning', 'bounds', 'thermal_conductivity',
                    f'Thermal conductivity {mat.thermal_conductivity:.0f} W/(m·K) exceeds graphene',
                    actual_value=mat.thermal_conductivity, confidence_impact=0.1))
            uncertainty['thermal_conductivity'] = 10.0  # ±10%

        # Melting point: reasonable range
        if mat.melting_point > 0:
            if mat.melting_point < 50:  # -223°C
                issues.append(ValidationIssue('warning', 'bounds', 'melting_point',
                    f'Melting point {mat.melting_point:.0f} K ({mat.melting_point-273:.0f}°C) very low',
                    actual_value=mat.melting_point, confidence_impact=0.05))
            elif mat.melting_point > 4500:  # Above tungsten carbide
                issues.append(ValidationIssue('warning', 'bounds', 'melting_point',
                    f'Melting point {mat.melting_point:.0f} K ({mat.melting_point-273:.0f}°C) exceptionally high',
                    actual_value=mat.melting_point, confidence_impact=0.05))
            uncertainty['melting_point'] = 1.0  # ±1%

        # Young's modulus: carbon nanotubes/graphene ~1 TPa = 1000 GPa
        if mat.youngs_modulus > 0:
            if mat.youngs_modulus > 1100:
                issues.append(ValidationIssue('warning', 'bounds', 'youngs_modulus',
                    f"Young's modulus {mat.youngs_modulus:.0f} GPa exceeds graphene (~1000 GPa)",
                    actual_value=mat.youngs_modulus, confidence_impact=0.1))
            uncertainty['youngs_modulus'] = 5.0  # ±5%

        return issues

    def _check_property_relationships(self, mat: MaterialProperties, uncertainty: Dict) -> List[ValidationIssue]:
        """Check if related properties are consistent"""
        issues = []

        # Relationship 1: E = 2G(1 + ν)
        if mat.youngs_modulus > 0 and mat.shear_modulus > 0 and mat.poissons_ratio > 0:
            E_calc = 2 * mat.shear_modulus * (1 + mat.poissons_ratio)
            deviation = abs(E_calc - mat.youngs_modulus) / mat.youngs_modulus * 100

            if deviation > 15:
                issues.append(ValidationIssue('error' if deviation > 30 else 'warning',
                    'consistency', 'youngs_modulus',
                    f'E = 2G(1+ν) inconsistent: calculated {E_calc:.1f} GPa vs stated {mat.youngs_modulus:.1f} GPa',
                    expected_value=E_calc, actual_value=mat.youngs_modulus,
                    deviation_percent=deviation,
                    confidence_impact=min(0.2, deviation/100)))

        # Relationship 2: K = E / [3(1 - 2ν)]
        if mat.youngs_modulus > 0 and mat.bulk_modulus > 0 and 0 < mat.poissons_ratio < 0.5:
            K_calc = mat.youngs_modulus / (3 * (1 - 2 * mat.poissons_ratio))
            deviation = abs(K_calc - mat.bulk_modulus) / mat.bulk_modulus * 100

            if deviation > 20:
                issues.append(ValidationIssue('warning', 'consistency', 'bulk_modulus',
                    f'K = E/[3(1-2ν)] inconsistent: deviation {deviation:.1f}%',
                    expected_value=K_calc, actual_value=mat.bulk_modulus,
                    deviation_percent=deviation,
                    confidence_impact=min(0.15, deviation/150)))

        # Relationship 3: Yield ≤ Tensile strength
        if mat.yield_strength > 0 and mat.tensile_strength > 0:
            if mat.yield_strength > mat.tensile_strength:
                issues.append(ValidationIssue('error', 'consistency', 'yield_strength',
                    f'Yield strength ({mat.yield_strength:.0f} MPa) exceeds tensile strength ({mat.tensile_strength:.0f} MPa)',
                    expected_value=mat.tensile_strength, actual_value=mat.yield_strength,
                    confidence_impact=0.2))

        # Relationship 4: Service temp < Melting point
        if mat.max_service_temp > 0 and mat.melting_point > 0:
            if mat.max_service_temp > mat.melting_point * 0.95:
                deviation = (mat.max_service_temp / mat.melting_point) * 100
                issues.append(ValidationIssue('warning', 'consistency', 'max_service_temp',
                    f'Service temp ({mat.max_service_temp:.0f} K) too close to melting point ({mat.melting_point:.0f} K)',
                    expected_value=mat.melting_point * 0.8, actual_value=mat.max_service_temp,
                    deviation_percent=deviation - 100,
                    confidence_impact=0.1))

        # Relationship 5: Density consistency (g/cm³ vs kg/m³)
        if mat.density_g_cm3 > 0 and mat.density_kg_m3 > 0:
            expected_kg_m3 = mat.density_g_cm3 * 1000
            deviation = abs(expected_kg_m3 - mat.density_kg_m3) / mat.density_kg_m3 * 100

            if deviation > 1:
                issues.append(ValidationIssue('error', 'consistency', 'density',
                    f'Density unit mismatch: {mat.density_g_cm3} g/cm³ ≠ {mat.density_kg_m3} kg/m³',
                    expected_value=expected_kg_m3, actual_value=mat.density_kg_m3,
                    deviation_percent=deviation,
                    confidence_impact=0.15))

        # Relationship 6: Electrical conductivity = 1 / resistivity
        if mat.electrical_conductivity > 0 and mat.electrical_resistivity > 0:
            expected_conductivity = 1.0 / mat.electrical_resistivity
            deviation = abs(expected_conductivity - mat.electrical_conductivity) / mat.electrical_conductivity * 100

            if deviation > 10:
                issues.append(ValidationIssue('warning', 'consistency', 'electrical_conductivity',
                    f'σ = 1/ρ inconsistent: deviation {deviation:.1f}%',
                    expected_value=expected_conductivity, actual_value=mat.electrical_conductivity,
                    deviation_percent=deviation,
                    confidence_impact=min(0.1, deviation/200)))

        return issues

    def _cross_validate_references(self, mat: MaterialProperties, uncertainty: Dict) -> List[ValidationIssue]:
        """Cross-validate against known reference materials"""
        issues = []

        for ref_name, ref_props in self.REFERENCE_VALUES.items():
            # Check if material name contains reference name
            if ref_name.lower() in mat.name.lower() or mat.name.lower() in ref_name.lower():
                for prop_name, (ref_value, ref_uncertainty) in ref_props.items():
                    actual = getattr(mat, prop_name, 0)

                    if actual > 0:
                        deviation = abs(actual - ref_value) / ref_value * 100

                        # Use 3-sigma rule (99.7% confidence)
                        threshold = (ref_uncertainty / ref_value) * 100 * 3

                        if deviation > threshold:
                            severity = 'error' if deviation > threshold * 2 else 'warning'
                            issues.append(ValidationIssue(severity, 'cross_validation', prop_name,
                                f'{prop_name} deviates from {ref_name} reference: {actual:.2f} vs {ref_value} ± {ref_uncertainty}',
                                expected_value=ref_value, actual_value=actual,
                                deviation_percent=deviation,
                                confidence_impact=min(0.2, deviation/100)))

                        # Update uncertainty estimate
                        uncertainty[prop_name] = (ref_uncertainty / ref_value) * 100

        return issues

    def _detect_outliers(self, mat: MaterialProperties) -> List[ValidationIssue]:
        """Detect statistical outliers based on material category"""
        issues = []

        category_stats = self._get_category_statistics(mat.category)

        for prop_name, (mean, std) in category_stats.items():
            value = getattr(mat, prop_name, 0)

            if value > 0 and std > 0:
                z_score = abs(value - mean) / std

                # Flag if more than 3 standard deviations from mean
                if z_score > 3:
                    issues.append(ValidationIssue('info', 'outlier', prop_name,
                        f'{prop_name} is statistical outlier for {mat.category} (z-score: {z_score:.1f})',
                        actual_value=value, deviation_percent=(z_score - 3) * 10,
                        confidence_impact=min(0.05, z_score/50)))

        return issues

    def _check_completeness(self, mat: MaterialProperties) -> Tuple[List[ValidationIssue], float]:
        """Check data completeness"""
        issues = []

        critical_props = {
            'metal': ['density', 'youngs_modulus', 'thermal_conductivity', 'electrical_resistivity'],
            'ceramic': ['density', 'youngs_modulus', 'melting_point', 'thermal_conductivity'],
            'polymer': ['density', 'tensile_strength', 'thermal_conductivity'],
            'nanomaterial': ['density', 'thermal_conductivity'],
        }

        important_props = {
            'metal': ['tensile_strength', 'melting_point', 'specific_heat'],
            'ceramic': ['hardness_vickers', 'fracture_toughness'],
            'polymer': ['glass_transition_temp', 'melting_point'],
        }

        category = mat.category
        critical = critical_props.get(category, ['density'])
        important = important_props.get(category, [])

        critical_missing = 0
        important_missing = 0

        for prop in critical:
            if getattr(mat, prop, 0) == 0:
                critical_missing += 1
                issues.append(ValidationIssue('warning', 'completeness', prop,
                    f'Missing critical property for {category}: {prop}',
                    confidence_impact=0.1))

        for prop in important:
            if getattr(mat, prop, 0) == 0:
                important_missing += 1
                issues.append(ValidationIssue('info', 'completeness', prop,
                    f'Missing important property: {prop}'))

        # Calculate completeness score
        total_critical = len(critical)
        total_important = len(important)

        if total_critical > 0:
            completeness = 100 * (1 - critical_missing / total_critical * 0.5 - important_missing / max(1, total_important) * 0.2)
        else:
            completeness = 100

        return issues, max(0, completeness)

    def _get_category_statistics(self, category: str) -> Dict[str, Tuple[float, float]]:
        """Get mean and std for properties in a category"""
        stats = {}
        materials = [m for m in self.db.materials.values() if m.category == category]

        if not materials:
            return stats

        for prop in ['density', 'youngs_modulus', 'thermal_conductivity', 'tensile_strength']:
            values = [getattr(m, prop, 0) for m in materials if getattr(m, prop, 0) > 0]
            if len(values) >= 3:  # Need at least 3 samples
                stats[prop] = (np.mean(values), np.std(values))

        return stats

    def _calculate_accuracy_score(self, issues: List[ValidationIssue]) -> float:
        """Calculate accuracy score based on issues"""
        score = 100.0

        for issue in issues:
            if issue.severity in ['critical', 'error']:
                score -= issue.confidence_impact * 100

        return max(0, score)

    def _calculate_consistency_score(self, issues: List[ValidationIssue]) -> float:
        """Calculate consistency score"""
        score = 100.0

        consistency_issues = [i for i in issues if i.category in ['consistency', 'bounds', 'physics']]

        for issue in consistency_issues:
            score -= issue.confidence_impact * 100

        return max(0, score)

    def validate_database(self, limit: Optional[int] = None) -> Dict[str, any]:
        """Validate entire database"""
        materials = list(self.db.materials.values())
        if limit:
            materials = materials[:limit]

        print(f"\nValidating {len(materials)} materials...")
        print("=" * 80)

        scores = []
        all_issues = defaultdict(int)

        for i, material in enumerate(materials, 1):
            if i % 100 == 0:
                print(f"Progress: {i}/{len(materials)}...")

            score = self.validate_material(material)
            scores.append(score)

            for issue in score.issues:
                all_issues[f"{issue.severity}_{issue.category}"] += 1

        # Calculate statistics
        confidences = [s.overall_confidence for s in scores]
        grades = [s.grade() for s in scores]
        grade_counts = {g: grades.count(g) for g in set(grades)}

        print("\n" + "=" * 80)
        print("VALIDATION COMPLETE")
        print("=" * 80)

        report = {
            'total_validated': len(materials),
            'average_confidence': np.mean(confidences),
            'median_confidence': np.median(confidences),
            'std_confidence': np.std(confidences),
            'grade_distribution': grade_counts,
            'top_materials': sorted(scores, key=lambda x: x.overall_confidence, reverse=True)[:10],
            'needs_review': sorted([s for s in scores if s.overall_confidence < 70], key=lambda x: x.overall_confidence)[:10],
            'issue_summary': dict(all_issues),
            'all_scores': scores
        }

        return report

    def print_report(self, report: Dict, verbose: bool = False):
        """Print validation report"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE VALIDATION REPORT")
        print("=" * 80)

        print(f"\nMaterials Validated: {report['total_validated']}")
        print(f"Average Confidence:  {report['average_confidence']:.1f}%")
        print(f"Median Confidence:   {report['median_confidence']:.1f}%")
        print(f"Std Deviation:       {report['std_confidence']:.1f}%")

        print("\n" + "=" * 80)
        print("GRADE DISTRIBUTION")
        print("=" * 80)
        for grade in ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'D', 'F']:
            count = report['grade_distribution'].get(grade, 0)
            pct = count / report['total_validated'] * 100
            bar = '█' * int(pct / 2)
            print(f"{grade:3s}: {bar:50s} {count:4d} ({pct:5.1f}%)")

        print("\n" + "=" * 80)
        print("TOP 10 HIGHEST CONFIDENCE MATERIALS")
        print("=" * 80)
        for score in report['top_materials']:
            uncertainties = ', '.join([f"{k}: ±{v:.1f}%" for k, v in list(score.uncertainty_estimates.items())[:2]])
            print(f"✓ {score.material_name:50s} {score.overall_confidence:>6.1f}% ({score.grade()})")
            if verbose and uncertainties:
                print(f"  Uncertainty: {uncertainties}")

        if report['needs_review']:
            print("\n" + "=" * 80)
            print("MATERIALS NEEDING REVIEW (Confidence < 70%)")
            print("=" * 80)
            for score in report['needs_review']:
                print(f"\n⚠️  {score.material_name} - {score.overall_confidence:.1f}% ({score.grade()})")
                print(f"   Accuracy: {score.accuracy_score:.0f}%, Consistency: {score.consistency_score:.0f}%, Completeness: {score.completeness_score:.0f}%")

                if verbose:
                    for issue in score.issues[:3]:
                        print(f"   {issue}")

        print("\n" + "=" * 80)
        print("ISSUE SUMMARY")
        print("=" * 80)
        for issue_type, count in sorted(report['issue_summary'].items(), key=lambda x: x[1], reverse=True)[:10]:
            severity, category = issue_type.split('_', 1)
            print(f"{severity:10s} / {category:20s}: {count:>5d}")

        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Comprehensive materials database validation')
    parser.add_argument('--full', action='store_true', help='Validate all materials')
    parser.add_argument('--save-report', action='store_true', help='Save JSON report')
    parser.add_argument('--verbose', action='store_true', help='Show detailed issues')
    args = parser.parse_args()

    print("=" * 80)
    print("QuLabInfinite Materials Database - Comprehensive Validation")
    print("=" * 80)

    # Load database
    print("\nLoading materials database...")
    db = MaterialsDatabase()
    print(f"✓ Loaded {db.get_count()} materials")

    # Run validation
    validator = ComprehensiveValidator(db)
    limit = None if args.full else 100

    report = validator.validate_database(limit=limit)
    validator.print_report(report, verbose=args.verbose)

    if args.save_report:
        output_file = 'validation_report_comprehensive.json'
        # Convert scores to dict for JSON
        json_report = {
            'total_validated': report['total_validated'],
            'average_confidence': report['average_confidence'],
            'median_confidence': report['median_confidence'],
            'std_confidence': report['std_confidence'],
            'grade_distribution': report['grade_distribution'],
            'top_materials': [
                {'name': s.material_name, 'confidence': s.overall_confidence, 'grade': s.grade()}
                for s in report['top_materials']
            ],
            'needs_review': [
                {'name': s.material_name, 'confidence': s.overall_confidence, 'issues': len(s.issues)}
                for s in report['needs_review']
            ],
            'issue_summary': report['issue_summary']
        }

        with open(output_file, 'w') as f:
            json.dump(json_report, f, indent=2)
        print(f"\n✓ Report saved to {output_file}")

    print("\n✓ Validation complete!")


if __name__ == '__main__':
    main()
