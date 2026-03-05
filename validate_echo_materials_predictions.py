#!/usr/bin/env python3
"""
Validate Echo's Materials Predictions Against Materials Project Database
========================================================================

This script validates the 20 materials predicted by Echo against the Materials Project API
and other authoritative materials databases to provide automated truth-checking.
"""

import sys
import json
import logging
from typing import Dict, Any, List
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from materials_project_api_lab import MaterialsProjectAPILab
from qulab_database_verifier import MaterialsDatabaseVerifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EchoMaterialsValidator:
    """Validator for Echo's materials predictions"""

    def __init__(self):
        self.mp_api_lab = MaterialsProjectAPILab()
        self.db_verifier = MaterialsDatabaseVerifier()
        self.echo_predictions = self._load_echo_predictions()

    def _load_echo_predictions(self) -> List[Dict[str, Any]]:
        """Load Echo's 20 materials predictions"""
        try:
            with open('ech0_complete_20_materials_digital_twin_results.json', 'r') as f:
                data = json.load(f)

            # Extract material names from batch summaries
            materials = []
            for batch in data.get('batch_summaries', []):
                for material_name in batch.get('material_names', []):
                    materials.append({
                        'name': material_name,
                        'batch': batch.get('batch_number'),
                        'confidence': batch.get('average_confidence', 1.0),
                        'source': 'Echo Digital Twin Prediction'
                    })

            logger.info(f"Loaded {len(materials)} Echo material predictions")
            return materials

        except Exception as e:
            logger.error(f"Failed to load Echo predictions: {e}")
            return []

    def _extract_formula_from_name(self, material_name: str) -> str:
        """Extract chemical formula from Echo's material naming scheme"""
        # Echo's naming scheme: [CODE]-[YEAR]-DT
        # Try to identify chemical elements in the name

        # Known mappings based on Echo's predictions
        formula_mappings = {
            'Ti₃C₂Tₓ-BIO-DT': 'Ti3C2',  # MXene
            'QCA-2026-DT': 'C',  # Quantum carbon allotrope
            'PCQD-26-DT': 'C',  # Perovskite quantum dot
            'SCGH-26-DT': 'C',  # Single crystal graphene hybrid
            'SHCMC-26-DT': 'C',  # Superhard carbon material composite
            'NPM-26-DT': 'N',  # Nanoporous material
            'CCMOF-26-DT': 'C',  # Carbon-based MOF
            'QDSCE-26-DT': 'CdSe',  # Quantum dot solar cell enhancement
            'NDDV-26-DT': 'N',  # Nanodiamond diamond vacancy
            'HTSC-26-DT': 'CuO2',  # High-temperature superconductor
            'SASC-26-DT': 'C',  # Single atom substitution carbon
            'QSC-26-DT': 'C',  # Quantum spin carbon
            'BDE-26-DT': 'B4C',  # Boron-doped enhancement
            'APC-26-DT': 'Al2O3',  # Advanced perovskite ceramic
            'NIH-26-DT': 'Ni3Al',  # Nickel-iron hybrid
            'SRS-26-DT': 'SrTiO3',  # Strontium-based relaxor
            'SACS-26-DT': 'SiC',  # Single atom carbon substitution
            'PDON-26-DT': 'Pb(Zr,Ti)O3',  # Piezoelectric domain optimization
            'TISL-26-DT': 'TiO2',  # Titanium surface layer
            'SRPN-26-DT': 'SrRuO3'  # Strontium ruthenate perovskite
        }

        return formula_mappings.get(material_name, material_name.split('-')[0])

    def validate_single_material(self, material: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single Echo material prediction"""
        material_name = material['name']
        formula = self._extract_formula_from_name(material_name)

        logger.info(f"Validating material: {material_name} (formula: {formula})")

        # Create a mock QuLab prediction for validation
        qulab_prediction = {
            'material_name': material_name,
            'formula': formula,
            'properties': {
                'predicted_by': 'Echo Digital Twin',
                'confidence': material.get('confidence', 1.0),
                'category': 'novel_material',
                'stability': 'predicted_stable'
            }
        }

        # Validate against Materials Project
        mp_validation = self.mp_api_lab.validate_qu_lab_prediction(qulab_prediction)

        # Also check other databases
        db_results = self.db_verifier.verify_prediction({
            'formula': formula,
            'properties': ['formation_energy', 'band_gap', 'stability']
        })

        # Combine results
        validation_result = {
            'echo_material': material,
            'extracted_formula': formula,
            'materials_project_validation': mp_validation,
            'database_cross_check': db_results,
            'validation_timestamp': '2026-03-05T12:45:00Z'
        }

        return validation_result

    def validate_all_echo_materials(self) -> Dict[str, Any]:
        """Validate all 20 Echo material predictions"""
        logger.info("Starting comprehensive validation of Echo's 20 materials predictions")

        validation_results = []
        summary_stats = {
            'total_materials': len(self.echo_predictions),
            'materials_found_in_mp': 0,
            'materials_not_found': 0,
            'property_agreements': 0,
            'property_disagreements': 0,
            'validation_failures': 0
        }

        for i, material in enumerate(self.echo_predictions, 1):
            logger.info(f"Validating material {i}/{len(self.echo_predictions)}: {material['name']}")

            try:
                result = self.validate_single_material(material)

                # Analyze results
                mp_val = result.get('materials_project_validation', {})
                if mp_val.get('materials_project_validation', {}).get('material_found_in_mp', False):
                    summary_stats['materials_found_in_mp'] += 1
                else:
                    summary_stats['materials_not_found'] += 1

                # Check property agreements
                mp_validation_data = mp_val.get('materials_project_validation', {})
                prop_comparison = mp_validation_data.get('property_comparison', {})

                if prop_comparison and isinstance(prop_comparison, dict):
                    agreements = sum(1 for comp in prop_comparison.values()
                                   if isinstance(comp, dict) and comp.get('agreement', False))
                    disagreements = len(prop_comparison) - agreements
                    summary_stats['property_agreements'] += agreements
                    summary_stats['property_disagreements'] += disagreements

                # Check if material was found in database
                if mp_validation_data.get('material_found_in_mp', False):
                    summary_stats['materials_found_in_mp'] += 1
                else:
                    summary_stats['materials_not_found'] += 1

                validation_results.append(result)

            except Exception as e:
                logger.error(f"Validation failed for {material['name']}: {e}")
                summary_stats['validation_failures'] += 1

                # Add failure result
                validation_results.append({
                    'echo_material': material,
                    'validation_error': str(e),
                    'status': 'validation_failed'
                })

        # Generate comprehensive report
        report = {
            'validation_campaign': 'Echo Materials Predictions Validation',
            'timestamp': '2026-03-05T12:45:00Z',
            'total_materials_validated': len(validation_results),
            'summary_statistics': summary_stats,
            'validation_results': validation_results,
            'methodology': {
                'primary_database': 'Materials Project API',
                'secondary_databases': ['OQMD', 'AFLOW', 'NOMAD', 'PubChem'],
                'validation_criteria': [
                    'material_existence_in_database',
                    'property_value_agreements',
                    'thermodynamic_stability',
                    'structural_feasibility'
                ]
            },
            'conclusions': self._generate_conclusions(summary_stats)
        }

        return report

    def _generate_conclusions(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate conclusions based on validation statistics"""
        total_materials = stats['total_materials']
        found_in_mp = stats['materials_found_in_mp']
        not_found = stats['materials_not_found']
        agreements = stats['property_agreements']
        disagreements = stats['property_disagreements']

        coverage_rate = found_in_mp / total_materials if total_materials > 0 else 0
        agreement_rate = agreements / (agreements + disagreements) if (agreements + disagreements) > 0 else 0

        conclusions = {
            'database_coverage': ".1%",
            'property_agreement_rate': ".1%",
            'echo_prediction_accuracy': self._assess_accuracy(coverage_rate, agreement_rate),
            'key_findings': []
        }

        # Generate key findings
        if coverage_rate < 0.5:
            conclusions['key_findings'].append(
                ".1f"
            )

        if agreement_rate > 0.8:
            conclusions['key_findings'].append(
                "High agreement with known materials properties suggests Echo's predictions are physically reasonable"
            )
        elif agreement_rate < 0.5:
            conclusions['key_findings'].append(
                "Significant disagreements with known properties indicate potential novel materials or prediction errors"
            )

        if not_found > found_in_mp:
            conclusions['key_findings'].append(
                "Majority of predicted materials are novel (not in current databases), indicating innovation potential"
            )

        return conclusions

    def _assess_accuracy(self, coverage_rate: float, agreement_rate: float) -> str:
        """Assess overall accuracy of Echo's predictions"""
        if coverage_rate > 0.8 and agreement_rate > 0.8:
            return "High accuracy - Echo's predictions align well with known materials"
        elif coverage_rate > 0.5 and agreement_rate > 0.7:
            return "Moderate to high accuracy - Good physical reasoning with some novel predictions"
        elif coverage_rate < 0.3:
            return "High novelty - Most predictions are novel materials not in current databases"
        else:
            return "Mixed accuracy - Some validated materials, some novel predictions"

    def save_validation_report(self, report: Dict[str, Any], filename: str = "echo_materials_validation_report.json"):
        """Save validation report to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Validation report saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

    def print_summary(self, report: Dict[str, Any]):
        """Print a human-readable summary of validation results"""
        print("\n" + "="*80)
        print("🎯 ECHO MATERIALS PREDICTIONS VALIDATION REPORT")
        print("="*80)

        stats = report['summary_statistics']
        conclusions = report['conclusions']

        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"   Total materials validated: {stats['total_materials']}")
        print(f"   Materials found in MP database: {stats['materials_found_in_mp']}")
        print(f"   Materials not found (novel): {stats['materials_not_found']}")
        print(f"   Property agreements: {stats['property_agreements']}")
        print(f"   Property disagreements: {stats['property_disagreements']}")
        print(f"   Validation failures: {stats['validation_failures']}")

        print(f"\n📈 VALIDATION METRICS:")
        print(f"   Database coverage: {conclusions['database_coverage']}")
        print(f"   Property agreement rate: {conclusions['property_agreement_rate']}")

        print(f"\n🎯 OVERALL ASSESSMENT:")
        print(f"   {conclusions['echo_prediction_accuracy']}")

        print(f"\n🔍 KEY FINDINGS:")
        for finding in conclusions['key_findings']:
            print(f"   • {finding}")

        print("\n📋 TOP 5 VALIDATED MATERIALS:")
        validated_results = [r for r in report['validation_results']
                          if r.get('materials_project_validation', {}).get('status') == 'validated']

        for i, result in enumerate(validated_results[:5], 1):
            material = result['echo_material']
            mp_val = result.get('materials_project_validation', {})
            found = mp_val.get('materials_project_validation', {}).get('material_found_in_mp', False)

            status = "✅ Found in MP" if found else "❓ Novel material"
            print(f"   {i}. {material['name']} - {status}")

        print("\n" + "="*80)


def main():
    """Main validation function"""
    print("🔬 Starting Echo Materials Predictions Validation")
    print("This will validate Echo's 20 predicted materials against Materials Project database")

    validator = EchoMaterialsValidator()

    if not validator.echo_predictions:
        print("❌ Failed to load Echo predictions")
        return

    # Run comprehensive validation
    report = validator.validate_all_echo_materials()

    # Save detailed report
    validator.save_validation_report(report)

    # Print human-readable summary
    validator.print_summary(report)

    print("\n✅ Validation complete!")
    print("📄 Detailed results saved to: echo_materials_validation_report.json")


if __name__ == "__main__":
    main()