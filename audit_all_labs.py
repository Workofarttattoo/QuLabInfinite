#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

QuLabInfinite - Comprehensive Lab Audit Tool
Analyzes all labs for scientific accuracy, pseudoscience, and production readiness
"""

import os
import sys
import json
import time
import importlib
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional

# Physical constants from NIST
NIST_CONSTANTS = {
    'c': 299792458,  # Speed of light (m/s)
    'h': 6.62607015e-34,  # Planck constant (J⋅s)
    'h_bar': 1.054571817e-34,  # Reduced Planck constant (J⋅s)
    'e': 1.602176634e-19,  # Elementary charge (C)
    'k_B': 1.380649e-23,  # Boltzmann constant (J/K)
    'N_A': 6.02214076e23,  # Avogadro constant (1/mol)
    'mu_0': 1.25663706212e-6,  # Vacuum permeability (N/A²)
    'epsilon_0': 8.8541878128e-12,  # Vacuum permittivity (F/m)
    'G': 6.67430e-11,  # Gravitational constant (m³/kg⋅s²)
    'm_e': 9.1093837015e-31,  # Electron mass (kg)
    'm_p': 1.67262192369e-27,  # Proton mass (kg)
    'R': 8.314462618,  # Gas constant (J/mol⋅K)
    'sigma': 5.670374419e-8,  # Stefan-Boltzmann constant (W/m²⋅K⁴)
}

class LabAuditor:
    """Comprehensive lab auditor for QuLabInfinite"""

    def __init__(self):
        self.base_path = Path("/Users/noone/aios/QuLabInfinite")
        self.results = {}
        self.issues = {}

    def scan_all_labs(self) -> Dict[str, Any]:
        """Scan all lab directories"""
        labs = []

        # Find all lab directories
        for path in self.base_path.iterdir():
            if path.is_dir() and (
                path.name.endswith('_lab') or
                path.name.endswith('_engine') or
                path.name.endswith('_sim') or
                path.name == 'biological_quantum'
            ):
                labs.append(path.name)

        labs.sort()

        print(f"\n{'='*80}")
        print(f"QuLabInfinite Comprehensive Lab Audit")
        print(f"{'='*80}")
        print(f"\nFound {len(labs)} labs to audit:\n")

        for i, lab in enumerate(labs, 1):
            print(f"  {i:2d}. {lab}")

        return labs

    def audit_lab(self, lab_name: str) -> Dict[str, Any]:
        """Audit a single lab"""
        lab_path = self.base_path / lab_name

        audit = {
            'name': lab_name,
            'path': str(lab_path),
            'has_demo': False,
            'has_readme': False,
            'has_tests': False,
            'has_init': False,
            'python_files': [],
            'issues': [],
            'warnings': [],
            'pseudoscience': [],
            'hardcoded_values': [],
            'missing_citations': [],
            'score': 0
        }

        # Check for required files
        if (lab_path / 'demo.py').exists():
            audit['has_demo'] = True
            audit['score'] += 20

        if (lab_path / 'README.md').exists():
            audit['has_readme'] = True
            audit['score'] += 20

        if (lab_path / '__init__.py').exists():
            audit['has_init'] = True
            audit['score'] += 10

        if (lab_path / 'tests').exists():
            audit['has_tests'] = True
            audit['score'] += 20

        # Find all Python files
        for py_file in lab_path.glob('**/*.py'):
            if '__pycache__' not in str(py_file):
                audit['python_files'].append(str(py_file.relative_to(lab_path)))

                # Analyze file contents
                try:
                    content = py_file.read_text()
                    self._analyze_python_file(content, py_file.name, audit)
                except Exception as e:
                    audit['issues'].append(f"Could not read {py_file.name}: {e}")

        # Calculate final score
        if not audit['issues']:
            audit['score'] += 10
        if not audit['pseudoscience']:
            audit['score'] += 10
        if not audit['hardcoded_values']:
            audit['score'] += 5
        if not audit['missing_citations']:
            audit['score'] += 5

        return audit

    def _analyze_python_file(self, content: str, filename: str, audit: Dict):
        """Analyze Python file for issues"""
        lines = content.split('\n')

        # Check for copyright header
        if 'Copyright (c) 2025 Joshua Hendricks Cole' not in content[:500]:
            audit['issues'].append(f"{filename}: Missing copyright header")

        # Check for problematic patterns
        problematic_patterns = {
            'print("TODO': 'Incomplete TODO',
            'raise NotImplementedError': 'Unimplemented function',
            '# FIXME': 'FIXME comment',
            '# HACK': 'HACK comment',
            'import random': 'Non-deterministic randomness',
        }

        for pattern, issue in problematic_patterns.items():
            if pattern in content:
                audit['warnings'].append(f"{filename}: {issue}")

        # Check for hardcoded magic numbers
        import re

        # Look for hardcoded values (numbers not in proper constants)
        magic_numbers = re.findall(r'(?<![a-zA-Z_])([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)', content)
        for num in magic_numbers:
            try:
                val = float(num)
                # Ignore small integers and common values
                if val not in [0, 1, 2, 3, 4, 5, 10, 100, 0.5, 0.1, 0.01]:
                    if abs(val) > 10 or (abs(val) < 0.001 and val != 0):
                        # Check if it's defined as a constant
                        if not re.search(rf'{num}\s*#.*(?:m/s|J|K|Pa|N|kg|mol)', content):
                            audit['hardcoded_values'].append(f"{filename}: {num}")
            except:
                pass

        # Check for pseudoscience keywords
        pseudoscience_keywords = [
            'free energy',
            'perpetual motion',
            'overunity',
            'zero point energy extraction',
            'cold fusion',
            'anti-gravity',
            'faster than light',
            'time travel',
        ]

        content_lower = content.lower()
        for keyword in pseudoscience_keywords:
            if keyword in content_lower:
                # Make sure it's not in a comment explaining why it's wrong
                if not re.search(f'#.*not.*{keyword}', content_lower):
                    audit['pseudoscience'].append(f"{filename}: Contains '{keyword}'")

        # Check for proper citations
        if any(word in content_lower for word in ['experiment', 'measured', 'observed']):
            if not any(cite in content for cite in ['doi:', 'DOI:', 'arXiv:', 'Nature', 'Science', 'Phys. Rev.']):
                audit['missing_citations'].append(f"{filename}: Claims experimental data without citations")

    def generate_report(self, labs: List[str]) -> str:
        """Generate comprehensive audit report"""
        report = []
        report.append(f"# QuLabInfinite Lab Audit Report")
        report.append(f"\nCopyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.")
        report.append(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        report.append(f"\n## Summary\n")
        report.append(f"- Total Labs: {len(labs)}")
        report.append(f"- Audit Completed: {len(self.results)}")

        # Calculate overall statistics
        total_score = sum(r['score'] for r in self.results.values())
        avg_score = total_score / len(self.results) if self.results else 0

        perfect_labs = [name for name, r in self.results.items() if r['score'] >= 90]
        good_labs = [name for name, r in self.results.items() if 70 <= r['score'] < 90]
        needs_work = [name for name, r in self.results.items() if r['score'] < 70]

        report.append(f"- Average Score: {avg_score:.1f}/100")
        report.append(f"- Production Ready: {len(perfect_labs)}")
        report.append(f"- Good Condition: {len(good_labs)}")
        report.append(f"- Needs Improvement: {len(needs_work)}")

        # Detailed lab reports
        report.append(f"\n## Detailed Lab Analysis\n")

        for lab_name in sorted(self.results.keys()):
            result = self.results[lab_name]
            report.append(f"\n### {lab_name}")
            report.append(f"**Score: {result['score']}/100**\n")

            # Status indicators
            report.append("**Components:**")
            report.append(f"- Demo: {'✅' if result['has_demo'] else '❌'}")
            report.append(f"- README: {'✅' if result['has_readme'] else '❌'}")
            report.append(f"- Tests: {'✅' if result['has_tests'] else '❌'}")
            report.append(f"- __init__.py: {'✅' if result['has_init'] else '❌'}")

            if result['issues']:
                report.append(f"\n**Critical Issues ({len(result['issues'])}):**")
                for issue in result['issues'][:5]:
                    report.append(f"- {issue}")

            if result['pseudoscience']:
                report.append(f"\n**Pseudoscience Detected ({len(result['pseudoscience'])}):**")
                for ps in result['pseudoscience'][:3]:
                    report.append(f"- {ps}")

            if result['hardcoded_values']:
                report.append(f"\n**Hardcoded Values ({len(result['hardcoded_values'])}):**")
                for hc in result['hardcoded_values'][:3]:
                    report.append(f"- {hc}")

            report.append("")

        # Action items
        report.append(f"\n## Required Actions\n")

        report.append("### High Priority (Production Blockers)")
        for lab_name, result in self.results.items():
            if result['pseudoscience'] or result['score'] < 50:
                report.append(f"- **{lab_name}**: Fix pseudoscience and critical issues")

        report.append("\n### Medium Priority")
        for lab_name, result in self.results.items():
            if not result['has_demo'] or not result['has_readme']:
                report.append(f"- **{lab_name}**: Add demo.py and README.md")

        report.append("\n### Low Priority")
        for lab_name, result in self.results.items():
            if result['hardcoded_values'] and result['score'] >= 70:
                report.append(f"- **{lab_name}**: Replace magic numbers with NIST constants")

        # Footer
        report.append("\n---")
        report.append("Websites: https://aios.is | https://thegavl.com | https://red-team-tools.aios.is")

        return '\n'.join(report)

    def run_full_audit(self):
        """Run complete audit of all labs"""
        labs = self.scan_all_labs()

        print(f"\n{'='*80}")
        print("Starting comprehensive audit...")
        print(f"{'='*80}\n")

        for i, lab_name in enumerate(labs, 1):
            print(f"[{i}/{len(labs)}] Auditing {lab_name}...", end=' ')
            try:
                result = self.audit_lab(lab_name)
                self.results[lab_name] = result

                # Quick status
                if result['score'] >= 90:
                    status = "✅ EXCELLENT"
                elif result['score'] >= 70:
                    status = "⚠️  GOOD"
                else:
                    status = "❌ NEEDS WORK"

                print(f"{status} (Score: {result['score']}/100)")

            except Exception as e:
                print(f"❌ ERROR: {e}")
                self.results[lab_name] = {
                    'name': lab_name,
                    'score': 0,
                    'issues': [f"Audit failed: {str(e)}"],
                    'has_demo': False,
                    'has_readme': False,
                    'has_tests': False,
                    'has_init': False
                }

        # Generate and save report
        report = self.generate_report(labs)

        report_path = self.base_path / "LAB_AUDIT_COMPLETE.md"
        report_path.write_text(report)

        print(f"\n{'='*80}")
        print(f"✅ Audit Complete!")
        print(f"{'='*80}")
        print(f"\nReport saved to: {report_path}")

        # Save detailed JSON results
        json_path = self.base_path / "lab_audit_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Detailed results saved to: {json_path}")

        return report


if __name__ == "__main__":
    auditor = LabAuditor()
    auditor.run_full_audit()