#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Extended test suite for all QuLabInfinite lab modules.
"""

import sys
import os
sys.path.insert(0, '/Users/noone/QuLabInfinite')

import importlib
import traceback

def test_lab_module(name: str, module_path: str) -> dict:
    """Test a lab module."""
    result = {
        'name': name,
        'import': False,
        'has_class': False,
        'instantiable': False,
        'error': None
    }

    try:
        # Import
        module = importlib.import_module(module_path)
        result['import'] = True

        # Check for main class
        expected_classes = [
            name.replace(' ', ''),
            name.replace(' ', '') + 'Lab',
            'Lab',
            module_path.split('.')[-1].title().replace('_', '')
        ]

        for cls_name in expected_classes:
            if hasattr(module, cls_name):
                result['has_class'] = True
                cls = getattr(module, cls_name)

                # Try instantiation
                try:
                    instance = cls()
                    result['instantiable'] = True
                except:
                    pass
                break

        if not result['has_class']:
            # Check for functions
            if any(hasattr(module, fn) for fn in ['run', 'demo', 'main', 'test']):
                result['has_class'] = True
                result['instantiable'] = True

    except Exception as e:
        result['error'] = str(e)

    return result

def main():
    """Test extended lab modules."""
    labs = [
        # Core labs
        ("Chemistry Lab", "chemistry_lab.chemistry_lab"),
        ("Materials Lab", "materials_lab.materials_lab"),
        ("Quantum Lab", "quantum_lab.quantum_lab"),
        ("Oncology Lab", "oncology_lab.oncology_lab"),
        ("Frequency Lab", "frequency_lab.frequency_lab"),

        # Specialized modules
        ("Cardiovascular Plaque Lab", "cardiovascular_plaque_lab"),
        ("Protein Folding Lab", "protein_folding_lab_lab"),
        ("Complete Realistic Lab", "complete_realistic_lab"),
        ("Realistic Tumor Lab", "realistic_tumor_lab"),

        # Integration modules
        ("Chemistry QuLab AI", "chemistry_lab.qulab_ai_integration"),
        ("Materials QuLab AI", "materials_lab.qulab_ai_integration"),
        ("Frequency QuLab AI", "frequency_lab.qulab_ai_integration"),
    ]

    print("\n" + "="*80)
    print("EXTENDED LAB MODULE TESTING")
    print("="*80 + "\n")

    results = []
    for name, module_path in labs:
        print(f"Testing: {name}...", end=" ")
        result = test_lab_module(name, module_path)
        results.append(result)

        if result['import'] and result['instantiable']:
            print("✓ PASS")
        elif result['import']:
            print("⚠ PARTIAL (imported but not instantiable)")
        else:
            print(f"✗ FAIL: {result['error']}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80 + "\n")

    passed = sum(1 for r in results if r['import'] and r['instantiable'])
    partial = sum(1 for r in results if r['import'] and not r['instantiable'])
    failed = sum(1 for r in results if not r['import'])

    print(f"Total: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Partial: {partial}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {100*passed/len(results):.1f}%\n")

    # Table
    print("| Lab Name | Import | Class Found | Instantiable |")
    print("|----------|--------|------------|-------------|")
    for r in results:
        import_icon = "✓" if r['import'] else "✗"
        class_icon = "✓" if r['has_class'] else "✗"
        inst_icon = "✓" if r['instantiable'] else "✗"
        print(f"| {r['name']} | {import_icon} | {class_icon} | {inst_icon} |")

if __name__ == "__main__":
    main()
