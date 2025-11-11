#!/usr/bin/env python3
"""
Comprehensive Lab Fixer
Systematically fixes all errors in QuLabInfinite lab files
"""

import os
import re
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Define all lab files
LAB_FILES = [
    "quantum_mechanics_lab.py",
    "nuclear_physics_lab.py",
    "particle_physics_lab.py",
    "astrophysics_lab.py",
    "fluid_dynamics_lab.py",
    "thermodynamics_lab.py",
    "electromagnetism_lab.py",
    "optics_and_photonics_lab.py",
    "organic_chemistry_lab.py",
    "inorganic_chemistry_lab.py",
    "physical_chemistry_lab.py",
    "analytical_chemistry_lab.py",
    "electrochemistry_lab.py",
    "catalysis_lab.py",
    "molecular_biology_lab.py",
    "cell_biology_lab.py",
    "ecology_lab.py",
    "evolutionary_biology_lab.py",
    "bioinformatics_lab.py",
    "structural_engineering_lab.py",
    "electrical_engineering_lab.py",
    "mechanical_engineering_lab.py",
    "biomedical_engineering_lab.py",
    "materials_science_lab.py",
    "robotics_lab.py",
    "control_systems_lab.py",
    "signal_processing_lab.py",
    "drug_design_lab.py",
    "pharmacology_lab.py",
    "toxicology_lab.py",
    "medical_imaging_lab.py",
    "genomics_lab.py",
    "proteomics_lab.py",
    "oncology_lab.py",
    "cardiology_lab.py",
    "neurology_lab.py",
    "climate_modeling_lab.py",
    "oceanography_lab.py",
    "hydrology_lab.py",
    "environmental_engineering_lab.py",
    "machine_learning_lab.py",
    "deep_learning_lab.py",
    "neural_networks_lab.py",
    "computer_vision_lab.py",
    "natural_language_processing_lab.py",
    "cryptography_lab.py",
    "quantum_computing_lab.py",
]

def test_lab(lab_file: str) -> Tuple[bool, str]:
    """Test if a lab file runs without errors"""
    try:
        result = subprocess.run(
            ["python3", lab_file],
            capture_output=True,
            text=True,
            timeout=10,
            cwd="/Users/noone/QuLabInfinite"
        )
        if result.returncode == 0:
            return True, "SUCCESS"
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)

def categorize_error(error_msg: str) -> str:
    """Categorize the type of error"""
    if "NameError" in error_msg and "scipy" in error_msg:
        return "MISSING_SCIPY_IMPORT"
    elif "NameError" in error_msg and "constants" in error_msg:
        return "MISSING_CONSTANTS_IMPORT"
    elif "AttributeError" in error_msg and "constants" in error_msg:
        return "SCIPY_CONSTANTS_HALLUCINATION"
    elif "SyntaxError" in error_msg and "dtype=" in error_msg:
        return "TYPE_ANNOTATION_ERROR"
    elif "TypeError" in error_msg and "ndarray()" in error_msg:
        return "NDARRAY_CALL_ERROR"
    elif "KeyError" in error_msg:
        return "KEY_ERROR"
    elif "AttributeError" in error_msg:
        return "ATTRIBUTE_ERROR"
    elif "ZeroDivisionError" in error_msg:
        return "ZERO_DIVISION"
    elif "ValueError" in error_msg:
        return "VALUE_ERROR"
    elif "IndexError" in error_msg:
        return "INDEX_ERROR"
    elif "TypeError" in error_msg:
        return "TYPE_ERROR"
    elif "TIMEOUT" in error_msg:
        return "TIMEOUT"
    else:
        return "RUNTIME_ERROR"

def apply_common_fixes(lab_file: str) -> int:
    """Apply common fixes to a lab file"""
    fixes_applied = 0
    file_path = f"/Users/noone/QuLabInfinite/{lab_file}"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Fix 1: Add missing scipy imports if scipy.constants is used
    if 'scipy.constants.' in content and 'import scipy.constants' not in content:
        # Add import after other imports
        content = re.sub(
            r'(from scipy\.constants import[^\n]+\n)',
            r'\1import scipy.constants\n',
            content
        )
        if content != original_content:
            fixes_applied += 1
            original_content = content
    
    # Fix 2: Add physical_constants import if used
    if 'physical_constants[' in content and 'physical_constants' not in content[:500]:
        content = re.sub(
            r'(from scipy\.constants import[^,\n]+)',
            r'\1, physical_constants',
            content,
            count=1
        )
        if content != original_content:
            fixes_applied += 1
            original_content = content
    
    # Fix 3: Fix np.ndarray(dtype=...) to np.ndarray
    content = re.sub(
        r': np\.ndarray\(dtype=np\.float64\)(\s+#[^\n]*)?(\s+=[^,\n]+)?',
        r': np.ndarray\2',
        content
    )
    if content != original_content:
        fixes_applied += 1
        original_content = content
    
    # Fix 4: Fix default_factory with dtype in wrong place
    content = re.sub(
        r'default_factory=lambda: np\.zeros\((\d+)\)\)',
        r'default_factory=lambda: np.zeros(\1, dtype=np.float64))',
        content
    )
    if content != original_content:
        fixes_applied += 1
        original_content = content
    
    # Fix 5: Replace common scipy.constants hallucinations
    replacements = {
        'constants.boltzmann': 'constants.k',
        'scipy.constants.boltzmann': 'scipy.constants.k',
        'constants.avogadro': 'constants.Avogadro',
        'scipy.constants.avogadro': 'scipy.constants.Avogadro',
    }
    
    for old, new in replacements.items():
        if old in content:
            content = content.replace(old, new)
            fixes_applied += 1
    
    # Write back if changes were made
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
    
    return fixes_applied

def main():
    print("=" * 80)
    print("COMPREHENSIVE LAB FIXER")
    print("=" * 80)
    print()
    
    results = {
        "working": [],
        "fixed": [],
        "still_broken": []
    }
    
    error_categories = {}
    
    # First pass: test all labs and categorize errors
    print("Phase 1: Testing all labs...")
    print("-" * 80)
    
    for lab_file in LAB_FILES:
        if not os.path.exists(f"/Users/noone/QuLabInfinite/{lab_file}"):
            continue
            
        success, error = test_lab(lab_file)
        
        if success:
            results["working"].append(lab_file)
            print(f"✓ {lab_file}")
        else:
            category = categorize_error(error)
            if category not in error_categories:
                error_categories[category] = []
            error_categories[category].append((lab_file, error[:200]))
            print(f"✗ {lab_file}: {category}")
    
    print()
    print("=" * 80)
    print(f"Initial Results: {len(results['working'])} working, {len(LAB_FILES) - len(results['working'])} broken")
    print("=" * 80)
    print()
    
    # Phase 2: Apply common fixes
    print("Phase 2: Applying common fixes...")
    print("-" * 80)
    
    for lab_file in LAB_FILES:
        if lab_file in results["working"]:
            continue
            
        if not os.path.exists(f"/Users/noone/QuLabInfinite/{lab_file}"):
            continue
            
        fixes = apply_common_fixes(lab_file)
        if fixes > 0:
            print(f"  Applied {fixes} fixes to {lab_file}")
    
    print()
    
    # Phase 3: Re-test all labs
    print("Phase 3: Re-testing all labs...")
    print("-" * 80)
    
    for lab_file in LAB_FILES:
        if lab_file in results["working"]:
            continue
            
        if not os.path.exists(f"/Users/noone/QuLabInfinite/{lab_file}"):
            continue
            
        success, error = test_lab(lab_file)
        
        if success:
            results["fixed"].append(lab_file)
            print(f"✓ {lab_file} - FIXED!")
        else:
            results["still_broken"].append((lab_file, categorize_error(error), error[:300]))
            print(f"✗ {lab_file}: {categorize_error(error)}")
    
    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Working from start: {len(results['working'])}")
    print(f"Fixed this session: {len(results['fixed'])}")
    print(f"Still broken: {len(results['still_broken'])}")
    print(f"Total working: {len(results['working']) + len(results['fixed'])}")
    print()
    
    # Save detailed report
    report = {
        "summary": {
            "working_from_start": len(results['working']),
            "fixed_this_session": len(results['fixed']),
            "still_broken": len(results['still_broken']),
            "total_working": len(results['working']) + len(results['fixed'])
        },
        "working_labs": results['working'],
        "fixed_labs": results['fixed'],
        "still_broken_labs": [
            {"file": f, "category": c, "error": e} 
            for f, c, e in results['still_broken']
        ]
    }
    
    with open('/Users/noone/QuLabInfinite/comprehensive_fix_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("Detailed report saved to: comprehensive_fix_report.json")
    print()
    
    # Print still broken labs by category
    if results['still_broken']:
        print("Still Broken Labs by Category:")
        print("-" * 80)
        broken_by_category = {}
        for lab_file, category, error in results['still_broken']:
            if category not in broken_by_category:
                broken_by_category[category] = []
            broken_by_category[category].append(lab_file)
        
        for category, labs in sorted(broken_by_category.items()):
            print(f"\n{category}: ({len(labs)} labs)")
            for lab in labs:
                print(f"  - {lab}")

if __name__ == "__main__":
    main()


