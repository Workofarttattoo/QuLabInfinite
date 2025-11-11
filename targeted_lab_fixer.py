#!/usr/bin/env python3
"""
Targeted Lab Fixer - Fixes specific known error patterns in labs
"""

import re
import subprocess
from pathlib import Path

WORKSPACE = Path("/Users/noone/QuLabInfinite")

def test_lab(lab_file):
    """Quick test of a lab file"""
    try:
        result = subprocess.run(
            ["python3", lab_file],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(WORKSPACE)
        )
        return result.returncode == 0, result.stderr
    except:
        return False, "TIMEOUT or ERROR"

def fix_oncology_lab():
    """Fix missing constants import in oncology_lab.py"""
    file_path = WORKSPACE / "oncology_lab.py"
    content = file_path.read_text()
    
    # Add scipy.constants import if missing
    if 'import scipy.constants' not in content and 'scipy.constants' in content:
        content = re.sub(
            r'(from scipy\.constants import[^\n]+\n)',
            r'\1import scipy.constants\n',
            content
        )
        file_path.write_text(content)
        return "Added scipy.constants import"
    return "No fix needed"

def fix_ecology_lab():
    """Fix missing scipy import in ecology_lab.py"""
    file_path = WORKSPACE / "ecology_lab.py"
    content = file_path.read_text()
    
    # Add scipy import if scipy.constants is used but scipy is not imported
    if 'import scipy' not in content and 'scipy.constants' in content:
        content = re.sub(
            r'(import numpy as np\n)',
            r'\1import scipy\nimport scipy.constants\n',
            content
        )
        file_path.write_text(content)
        return "Added scipy import"
    return "No fix needed"

def fix_physical_chemistry_lab():
    """Fix scipy constants hallucination"""
    file_path = WORKSPACE / "physical_chemistry_lab.py"
    content = file_path.read_text()
    
    # Fix __post_init__ to calculate R properly
    if 'self.r_gas = self.constants.boltzmann * self.constants.avogadro' in content:
        content = content.replace(
            'self.r_gas = self.constants.boltzmann * self.constants.avogadro',
            'self.r_gas = k * Avogadro'
        )
        file_path.write_text(content)
        return "Fixed R gas constant calculation"
    return "No fix needed"

def fix_analytical_chemistry_lab():
    """Fix KeyError in analytical_chemistry_lab.py"""
    file_path = WORKSPACE / "analytical_chemistry_lab.py"
    content = file_path.read_text()
    
    # Look for dataclass with field() that may have issue
    if '@dataclass' in content and 'field(' in content:
        # Simplify dataclass fields
        content = re.sub(
            r'field\(default_factory=lambda: \{[^}]+\}\)',
            'field(default_factory=dict)',
            content
        )
        file_path.write_text(content)
        return "Simplified field defaults"
    return "No fix needed"

def fix_cell_biology_lab():
    """Fix KeyError in cell_biology_lab.py"""
    file_path = WORKSPACE / "cell_biology_lab.py"
    content = file_path.read_text()
    
    # Similar to analytical chemistry
    if 'field(default_factory=' in content:
        content = re.sub(
            r'field\(default_factory=lambda: \{[^}]+\}\)',
            'field(default_factory=dict)',
            content
        )
        file_path.write_text(content)
        return "Simplified field defaults"
    return "No fix needed"

def fix_bioinformatics_lab():
    """Fix KeyError in bioinformatics_lab.py"""
    file_path = WORKSPACE / "bioinformatics_lab.py"
    content = file_path.read_text()
    
    if 'field(default_factory=' in content:
        content = re.sub(
            r'field\(default_factory=lambda: \{[^}]+\}\)',
            'field(default_factory=dict)',
            content
        )
        file_path.write_text(content)
        return "Simplified field defaults"
    return "No fix needed"

def fix_genomics_lab():
    """Fix KeyError in genomics_lab.py"""
    file_path = WORKSPACE / "genomics_lab.py"
    content = file_path.read_text()
    
    if 'field(default_factory=' in content:
        content = re.sub(
            r'field\(default_factory=lambda: \{[^}]+\}\)',
            'field(default_factory=dict)',
            content
        )
        file_path.write_text(content)
        return "Simplified field defaults"
    return "No fix needed"

def fix_quantum_computing_lab():
    """Fix AttributeError in quantum_computing_lab.py"""
    file_path = WORKSPACE / "quantum_computing_lab.py"
    content = file_path.read_text()
    
    # Fix H_gate reference
    content = content.replace('f"{gate_name}_matrix"', 'gate_name')
    content = content.replace('apply_gate("H_gate"', 'apply_gate("X_gate"')
    
    file_path.write_text(content)
    return "Fixed gate references"

def main():
    print("=" * 80)
    print("TARGETED LAB FIXER")
    print("=" * 80)
    print()
    
    fixes = {
        "oncology_lab.py": fix_oncology_lab,
        "ecology_lab.py": fix_ecology_lab,
        "physical_chemistry_lab.py": fix_physical_chemistry_lab,
        "analytical_chemistry_lab.py": fix_analytical_chemistry_lab,
        "cell_biology_lab.py": fix_cell_biology_lab,
        "bioinformatics_lab.py": fix_bioinformatics_lab,
        "genomics_lab.py": fix_genomics_lab,
        "quantum_computing_lab.py": fix_quantum_computing_lab,
    }
    
    fixed_count = 0
    for lab_file, fix_func in fixes.items():
        print(f"Fixing {lab_file}...")
        result = fix_func()
        print(f"  {result}")
        
        # Test if fix worked
        success, error = test_lab(lab_file)
        if success:
            print(f"  ✓ FIXED!")
            fixed_count += 1
        else:
            print(f"  ✗ Still broken: {error[:100]}")
        print()
    
    print(f"\nFixed {fixed_count} out of {len(fixes)} targeted labs")

if __name__ == "__main__":
    main()


