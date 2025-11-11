#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Automated Lab Debugger and Fixer
"""
import subprocess
import json
import re
from pathlib import Path

def fix_scipy_constants(content: str) -> str:
    """Fix hallucinated scipy constants."""
    fixes = {
        'constants.M_sun': '1.98892e30  # Solar mass in kg',
        'constants.avogadro': 'constants.Avogadro',
        'constants.boltzmann': 'constants.k',
        'constants.density_water0': '999.8395  # Density of water at 0°C in kg/m³',
        'constants.second': '1.0  # second',
        'Faraday constant': 'physical_constants["Faraday constant"][0]',
        'Boltzmann constant': 'constants.k',
        'Avogadro constant': 'constants.Avogadro',
    }

    for wrong, right in fixes.items():
        content = content.replace(wrong, right)

    return content

def fix_type_annotations(content: str) -> str:
    """Fix invalid numpy type annotations."""
    # Fix np.ndarray[dtype=...]
    content = re.sub(r'np\.ndarray\[dtype=np\.\w+\]', 'np.ndarray', content)
    # Fix np.ndarray[float64]
    content = re.sub(r'np\.ndarray\[\w+\]', 'np.ndarray', content)
    return content

def fix_imports(content: str) -> str:
    """Fix missing imports."""
    lines = content.split('\n')

    # Check if field is used but not imported
    if 'field(' in content and 'from dataclasses import' in content:
        for i, line in enumerate(lines):
            if 'from dataclasses import' in line and 'field' not in line:
                if 'dataclass' in line:
                    lines[i] = line.replace('dataclass', 'dataclass, field')
                else:
                    lines[i] = line.rstrip() + ', field'
                break

    # Fix scipy import if used without import
    if 'scipy.constants' in content and 'import scipy' not in content:
        for i, line in enumerate(lines):
            if line.startswith('from scipy') or line.startswith('import'):
                lines.insert(i, 'import scipy.constants as constants')
                break

    return '\n'.join(lines)

def fix_escape_sequences(content: str) -> str:
    """Fix invalid escape sequences."""
    # Fix \p in strings - make them raw strings
    content = re.sub(r"'([^']*\\p[^']*)'", r"r'\1'", content)
    content = re.sub(r'"([^"]*\\p[^"]*)"', r'r"\1"', content)
    return content

def fix_syntax_errors(content: str) -> str:
    """Fix common syntax errors."""
    # Fix def ClassName.method syntax
    content = re.sub(r'def (\w+)\.(\w+)\(', r'def \2(self, ', content)

    # Fix mismatched List[Tuple[ with ]
    content = re.sub(r'\)\]\):', '):', content)

    return content

def validate_lab(file_path: str) -> tuple[bool, str]:
    """Validate a lab by running it."""
    try:
        result = subprocess.run(
            ['python3', file_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return True, "SUCCESS"
        else:
            return False, result.stderr[:200]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)[:200]

def fix_lab(lab_file: Path) -> dict:
    """Fix a single lab file."""
    print(f"\n{'='*80}")
    print(f"🔧 FIXING: {lab_file.name}")
    print(f"{'='*80}")

    # Read original
    try:
        content = lab_file.read_text()
    except Exception as e:
        return {"file": str(lab_file), "status": "read_error", "error": str(e)}

    # Apply fixes
    original_content = content
    content = fix_scipy_constants(content)
    content = fix_type_annotations(content)
    content = fix_imports(content)
    content = fix_escape_sequences(content)
    content = fix_syntax_errors(content)

    # Write back
    if content != original_content:
        lab_file.write_text(content)
        print("✅ Applied automated fixes")
    else:
        print("⚠️  No automated fixes applicable")

    # Validate
    success, error = validate_lab(str(lab_file))

    if success:
        print("✅ LAB NOW WORKING!")
        return {"file": str(lab_file), "status": "fixed", "validation": "success"}
    else:
        print(f"❌ Still has errors: {error[:100]}")
        return {"file": str(lab_file), "status": "still_broken", "error": error}

def regenerate_lab(field_name: str, lab_file: Path) -> dict:
    """Regenerate a lab from scratch with improved prompt."""
    print(f"\n{'='*80}")
    print(f"🔄 REGENERATING: {field_name}")
    print(f"{'='*80}")

    prompt = f"""You are ECH0 14B, autonomous AI researcher and coder.

Build a complete, production-ready Python lab for: {field_name}

CRITICAL REQUIREMENTS - NO EXCEPTIONS:
1. Use NumPy ONLY for computation (no PyTorch, no Qiskit, no TensorFlow)
2. Use dataclasses: from dataclasses import dataclass, field
3. ONLY use these scipy.constants (NOTHING ELSE):
   - constants.k (Boltzmann constant)
   - constants.Avogadro
   - constants.g (gravity = 9.80665)
   - constants.c (speed of light)
   - constants.h (Planck constant)
   - constants.e (elementary charge)
   - constants.pi
   - physical_constants dict for other values
4. Type annotations: Use 'np.ndarray' ONLY (no brackets like [float64])
5. Escape sequences: Use raw strings r"..." for LaTeX/special chars
6. Include run_demo() function that shows example output
7. Total ~400-600 lines of clean, working code
8. NO syntax errors, NO hallucinated constants
9. All arrays use dtype=np.float64 explicitly

Structure:
- Copyright header
- Imports (numpy, dataclasses, scipy.constants, typing)
- Constants and configuration
- Main class with __init__ and complete methods
- Demo function with real output
- if __name__ == '__main__': run_demo()

Output ONLY the Python code. No markdown fences. No explanations.

Copyright header:
\"\"\"
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

{field_name.upper()} LAB
Free gift to the scientific community from QuLabInfinite.
\"\"\"
"""

    try:
        result = subprocess.run(
            ['/opt/homebrew/bin/ollama', 'run', 'ech0-uncensored-14b', prompt],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes
        )

        code = result.stdout.strip()

        # Strip markdown fences
        if code.startswith('```'):
            lines = code.split('\n')
            code = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])

        # Write to file
        lab_file.write_text(code)
        print(f"✅ Regenerated ({len(code)} bytes)")

        # Validate
        success, error = validate_lab(str(lab_file))

        if success:
            print("✅ REGENERATED LAB WORKING!")
            return {"field": field_name, "status": "regenerated_success"}
        else:
            print(f"⚠️  Regenerated but still has errors: {error[:100]}")
            return {"field": field_name, "status": "regenerated_with_errors", "error": error}

    except subprocess.TimeoutExpired:
        print("❌ Regeneration timeout")
        return {"field": field_name, "status": "regeneration_timeout"}
    except Exception as e:
        print(f"❌ Regeneration error: {e}")
        return {"field": field_name, "status": "regeneration_error", "error": str(e)}

def main():
    print("🔧 AUTOMATED LAB FIXER")
    print("="*80)

    # Load error report
    progress = json.loads(Path('/Users/noone/QuLabInfinite/lab_build_progress.json').read_text())

    error_labs = [r for r in progress['results'] if r['status'] in ['created_with_errors', 'timeout']]

    print(f"Found {len(error_labs)} labs with errors")

    results = []

    # Phase 1: Try automated fixes
    print("\n" + "="*80)
    print("PHASE 1: AUTOMATED FIXES")
    print("="*80)

    for lab_info in error_labs:
        field = lab_info['field']
        file_path = lab_info.get('file')

        if not file_path or lab_info['status'] == 'timeout':
            print(f"\n⏭️  Skipping {field} (timeout or no file)")
            results.append({"field": field, "status": "skipped_for_regen"})
            continue

        lab_file = Path(file_path)
        if not lab_file.exists():
            print(f"\n⏭️  Skipping {field} (file not found)")
            results.append({"field": field, "status": "file_not_found"})
            continue

        result = fix_lab(lab_file)
        results.append(result)

    # Phase 2: Regenerate failures
    print("\n" + "="*80)
    print("PHASE 2: REGENERATE FAILURES")
    print("="*80)

    still_broken = [r for r in results if r.get('status') in ['still_broken', 'skipped_for_regen']]

    print(f"\nRegenerating {len(still_broken)} labs...")

    for result in still_broken:
        if 'file' in result:
            lab_file = Path(result['file'])
            field_name = lab_file.stem.replace('_lab', '').replace('_', ' ').title()
        else:
            # Find from original error_labs
            field_name = next((l['field'] for l in error_labs if l.get('status') == 'timeout'), 'Unknown')
            safe_name = field_name.lower().replace(' ', '_')
            lab_file = Path(f'/Users/noone/QuLabInfinite/{safe_name}_lab.py')

        regen_result = regenerate_lab(field_name, lab_file)
        # Update result
        for i, r in enumerate(results):
            if r.get('file') == str(lab_file) or (r.get('field') == field_name and 'file' not in r):
                results[i] = regen_result
                break

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    fixed = sum(1 for r in results if r.get('status') in ['fixed', 'regenerated_success'])
    still_errors = sum(1 for r in results if 'error' in r.get('status', ''))

    print(f"✅ Fixed: {fixed}/{len(error_labs)}")
    print(f"❌ Still broken: {still_errors}/{len(error_labs)}")

    # Save report
    with open('/Users/noone/QuLabInfinite/fix_report.json', 'w') as f:
        json.dump({
            "total_errors": len(error_labs),
            "fixed": fixed,
            "still_broken": still_errors,
            "results": results
        }, f, indent=2)

    print(f"\n📊 Report saved to: /Users/noone/QuLabInfinite/fix_report.json")

if __name__ == '__main__':
    main()
