#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Quick fix for missing imports and type annotations
"""
import re
from pathlib import Path

# Labs that need import fixes
IMPORT_FIXES = [
    'quantum_computing_lab.py',
    'inorganic_chemistry_lab.py',
    'electrochemistry_lab.py',
]

# Labs that need type annotation fixes
TYPE_FIXES = [
    'particle_physics_lab.py',
    'catalysis_lab.py',
    'computer_vision_lab.py',
]

def fix_imports(file_path: Path) -> bool:
    """Add missing scipy imports."""
    content = file_path.read_text()

    # Check if scipy.constants is used but not imported
    if 'scipy.constants' in content and 'import scipy' not in content:
        # Find the import section
        lines = content.split('\n')
        import_idx = None
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                import_idx = i

        if import_idx is not None:
            # Insert after last import
            lines.insert(import_idx + 1, 'import scipy.constants')
            content = '\n'.join(lines)
            file_path.write_text(content)
            return True

    # Check if constants.X is used but constants not imported
    if re.search(r'\bconstants\.\w+', content) and 'from scipy import constants' not in content and 'import scipy.constants as constants' not in content:
        lines = content.split('\n')
        import_idx = None
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                import_idx = i

        if import_idx is not None:
            lines.insert(import_idx + 1, 'from scipy import constants')
            content = '\n'.join(lines)
            file_path.write_text(content)
            return True

    return False

def fix_type_annotations(file_path: Path) -> bool:
    """Fix invalid type annotations."""
    content = file_path.read_text()
    original = content

    # Fix np.ndarray[dtype=...] -> np.ndarray
    content = re.sub(r'np\.ndarray\[dtype=np\.\w+\]', 'np.ndarray', content)

    # Fix np.ndarray[float64] -> np.ndarray
    content = re.sub(r'np\.ndarray\[\w+\]', 'np.ndarray', content)

    if content != original:
        file_path.write_text(content)
        return True

    return False

def test_lab(file_path: Path) -> tuple[bool, str]:
    """Test if lab runs."""
    import subprocess
    try:
        result = subprocess.run(
            ['python3', str(file_path)],
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

def main():
    base_path = Path('/Users/noone/QuLabInfinite')

    print("🔧 QUICK IMPORT & TYPE ANNOTATION FIXER")
    print("="*80)

    fixed_count = 0
    working_count = 0

    # Fix imports
    print("\n📦 FIXING IMPORTS...")
    for lab_file in IMPORT_FIXES:
        file_path = base_path / lab_file
        if not file_path.exists():
            print(f"⚠️  {lab_file}: File not found")
            continue

        print(f"\n🔧 {lab_file}")
        if fix_imports(file_path):
            print("  ✅ Added missing imports")
            success, msg = test_lab(file_path)
            if success:
                print("  ✅ LAB NOW WORKING!")
                fixed_count += 1
                working_count += 1
            else:
                print(f"  ⚠️  Still has errors: {msg[:100]}")
                fixed_count += 1
        else:
            print("  ℹ️  No import fixes needed")

    # Fix type annotations
    print("\n🏷️  FIXING TYPE ANNOTATIONS...")
    for lab_file in TYPE_FIXES:
        file_path = base_path / lab_file
        if not file_path.exists():
            print(f"⚠️  {lab_file}: File not found")
            continue

        print(f"\n🔧 {lab_file}")
        if fix_type_annotations(file_path):
            print("  ✅ Fixed type annotations")
            success, msg = test_lab(file_path)
            if success:
                print("  ✅ LAB NOW WORKING!")
                fixed_count += 1
                working_count += 1
            else:
                print(f"  ⚠️  Still has errors: {msg[:100]}")
                fixed_count += 1
        else:
            print("  ℹ️  No type annotation fixes needed")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✅ Files fixed: {fixed_count}")
    print(f"✅ Labs now working: {working_count}")
    print(f"⚠️  Labs still broken: {fixed_count - working_count}")

if __name__ == '__main__':
    main()
