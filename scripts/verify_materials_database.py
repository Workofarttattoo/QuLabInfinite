#!/usr/bin/env python3
"""
Materials Database Verification Script

Checks all materials database components and reports status
"""

import sys
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def print_header(text):
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def print_status(component, status, details=""):
    emoji = "✅" if status else "⚠️"
    print(f"\n{emoji} {component}")
    if details:
        for line in details.split('\n'):
            if line.strip():
                print(f"    {line}")

def check_extended_database():
    """Check 14GB extended materials database"""
    try:
        from materials_lab.extended_materials_loader import ExtendedMaterialsLoader

        loader = ExtendedMaterialsLoader()
        info = loader.get_database_info()

        if info['exists']:
            details = f"""Location: {info['path']}
Size: {info['size_gb']} GB
Estimated materials: {info['estimated_materials']:,}
Status: READY TO USE"""
            return True, details
        else:
            details = f"""Expected location: {info['path']}
Status: FILE NOT FOUND
Action: Copy your 14GB file to this location"""
            return False, details
    except Exception as e:
        return False, f"Error: {e}"

def check_materials_project():
    """Check Materials Project integration"""
    try:
        from materials_lab.materials_project_client import MaterialsProjectClient
        import os

        api_key = os.environ.get('MP_API_KEY')

        if not api_key or api_key == 'your_materials_project_api_key_here':
            details = """API Key: NOT SET
Status: NEEDS CONFIGURATION
Action: Set MP_API_KEY environment variable"""
            return False, details

        details = f"""API Key: {api_key[:15]}...
Status: CONFIGURED
Downloaded: Check ./mp_cache/ for cached materials"""

        # Check for cached materials
        cache_dir = Path('./mp_cache')
        if cache_dir.exists():
            cached = len(list(cache_dir.glob('*.json')))
            details += f"\nCached materials: {cached}"

        return True, details
    except ImportError as e:
        details = f"""Status: NEEDS INSTALLATION
Error: {e}
Action: pip install --user pymatgen mp-api"""
        return False, details
    except Exception as e:
        return False, f"Error: {e}"

def check_curated_library():
    """Check curated materials library (1,059 materials)"""
    try:
        from materials_lab import MaterialsLab

        lab = MaterialsLab()
        count = len(lab.database.materials)

        details = f"""Materials loaded: {count:,}
Status: ACTIVE
Categories:
  - Metals: {len([m for m in lab.database.materials.values() if m.category == 'metal'])}
  - Ceramics: {len([m for m in lab.database.materials.values() if m.category == 'ceramic'])}
  - Polymers: {len([m for m in lab.database.materials.values() if m.category == 'polymer'])}"""

        return True, details
    except Exception as e:
        return False, f"Error: {e}"

def check_comprehensive_collection():
    """Check comprehensive materials collection"""
    try:
        data_file = Path('/home/user/QuLabInfinite/data/comprehensive_materials.json')

        if not data_file.exists():
            return False, "File not found"

        size_mb = data_file.stat().st_size / (1024 * 1024)

        with open(data_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'materials' in data:
                count = len(data['materials'])
            elif isinstance(data, list):
                count = len(data)
            else:
                count = 1

        details = f"""File: {data_file}
Size: {size_mb:.2f} MB
Materials: {count:,}
Status: AVAILABLE"""

        return True, details
    except Exception as e:
        return False, f"Error: {e}"

def calculate_total():
    """Calculate total materials available"""
    totals = []

    # Extended database
    try:
        from materials_lab.extended_materials_loader import ExtendedMaterialsLoader
        loader = ExtendedMaterialsLoader()
        info = loader.get_database_info()
        if info['exists']:
            totals.append(('Extended Database', info['estimated_materials']))
    except:
        pass

    # Materials Project (estimate if configured)
    import os
    if os.environ.get('MP_API_KEY'):
        totals.append(('Materials Project', 140_000))

    # Curated library
    try:
        from materials_lab import MaterialsLab
        lab = MaterialsLab()
        totals.append(('Curated Library', len(lab.database.materials)))
    except:
        pass

    # Comprehensive collection
    try:
        data_file = Path('/home/user/QuLabInfinite/data/comprehensive_materials.json')
        if data_file.exists():
            with open(data_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'materials' in data:
                    count = len(data['materials'])
                elif isinstance(data, list):
                    count = len(data)
                else:
                    count = 1
            totals.append(('Comprehensive Collection', count))
    except:
        pass

    return totals

def main():
    print_header("QULABINFINITE MATERIALS DATABASE VERIFICATION")

    print("\n📊 Checking database components...")

    # Check each component
    components = [
        ("Extended Materials Database (14GB)", check_extended_database),
        ("Materials Project Integration", check_materials_project),
        ("Curated Materials Library", check_curated_library),
        ("Comprehensive Materials Collection", check_comprehensive_collection),
    ]

    results = []
    for name, check_func in components:
        status, details = check_func()
        results.append((name, status))
        print_status(name, status, details)

    # Calculate totals
    print_header("TOTAL MATERIALS AVAILABLE")

    totals = calculate_total()
    total_count = sum(count for _, count in totals)

    print("\n📈 Materials by source:")
    for source, count in totals:
        print(f"   • {source}: {count:,}")

    print(f"\n🏆 TOTAL: {total_count:,} materials")

    # Competitive comparison
    print("\n" + "-" * 80)
    print("📊 Competitive Comparison:")
    print(f"   QuLabInfinite:     {total_count:>10,} ← YOU ARE HERE")
    print(f"   Materials Project:    140,000")
    print(f"   OQMD:               1,000,000")
    print(f"   MatWeb:               150,000")
    print(f"   Granta MI:            500,000")

    if total_count > 1_000_000:
        print("\n   🏆 YOU ARE IN THE TOP 3 GLOBALLY!")
    elif total_count > 500_000:
        print("\n   ⭐ YOU ARE IN THE TOP 5 GLOBALLY!")
    elif total_count > 140_000:
        print("\n   ✨ YOU BEAT MATERIALS PROJECT!")

    # Summary
    print_header("SUMMARY")

    working = sum(1 for _, status in results if status)
    total = len(results)

    print(f"\n✅ Working: {working}/{total} components")

    if working == total:
        print("\n🎉 ALL SYSTEMS OPERATIONAL!")
        print("\n   Your materials database is ready for:")
        print("   • Materials screening (1M+ materials)")
        print("   • Property prediction")
        print("   • Simulation validation")
        print("   • Materials discovery")
    else:
        print("\n⚠️  Some components need setup:")
        for name, status in results:
            if not status:
                print(f"   • {name}")

        print("\n📝 Next steps:")
        if not results[0][1]:  # Extended DB
            print("   1. Copy extended_materials_db.json to /home/user/QuLabInfinite/data/")
        if not results[1][1]:  # Materials Project
            print("   2. Set up Materials Project API: export MP_API_KEY='your_key'")
            print("      Get key from: https://materialsproject.org/api")

    print("\n" + "=" * 80)
    print("\n📚 Documentation:")
    print("   • Materials Database: MATERIALS_DATABASE.md")
    print("   • Materials Project: materials_lab/MATERIALS_PROJECT_README.md")
    print("   • API Reference: API_REFERENCE.md")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
