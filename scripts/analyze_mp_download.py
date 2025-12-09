#!/usr/bin/env python3
"""
Analyze Materials Project Download

Run this script locally to analyze your MP data at:
/noone/visualizer/qulab-infinite

Usage:
    python3 analyze_mp_download.py /path/to/mp/data

This will:
1. Count total materials
2. Show file sizes
3. Sample data structure
4. Generate statistics
5. Create import script for QuLabInfinite
"""

import os
import sys
import json
import gzip
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def format_size(bytes_size):
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def analyze_directory(path: Path):
    """Analyze directory for MP data"""
    print("=" * 80)
    print(f"ANALYZING: {path}")
    print("=" * 80)

    if not path.exists():
        print(f"\n❌ Error: Path does not exist: {path}")
        print("\nTrying common variations...")

        # Try variations
        variations = [
            Path("/noone/visualizer/qulab-infinite"),
            Path.home() / "visualizer" / "qulab-infinite",
            Path("/mnt/visualizer/qulab-infinite"),
            Path.cwd() / "qulab-infinite",
        ]

        for var_path in variations:
            if var_path.exists():
                print(f"✓ Found: {var_path}")
                path = var_path
                break
        else:
            print("\n❌ Could not find the directory")
            print("\nPlease provide the correct path as argument:")
            print("  python3 analyze_mp_download.py /correct/path")
            return None

    # Find all data files
    print(f"\n📂 Scanning directory...")

    json_files = list(path.rglob("*.json"))
    jsonl_files = list(path.rglob("*.jsonl"))
    gz_files = list(path.rglob("*.gz"))
    all_files = json_files + jsonl_files + gz_files

    print(f"\nFound:")
    print(f"  - JSON files: {len(json_files)}")
    print(f"  - JSONL files: {len(jsonl_files)}")
    print(f"  - GZ files: {len(gz_files)}")
    print(f"  - Total: {len(all_files)}")

    if not all_files:
        print("\n⚠️  No data files found!")
        return None

    # Analyze each file
    print("\n" + "=" * 80)
    print("FILE ANALYSIS")
    print("=" * 80)

    total_materials = 0
    file_stats = []

    for file_path in sorted(all_files, key=lambda x: x.stat().st_size, reverse=True)[:20]:
        size = file_path.stat().st_size

        # Try to count entries
        count = count_materials_in_file(file_path)
        total_materials += count

        file_stats.append({
            'path': str(file_path.relative_to(path)),
            'size': size,
            'count': count
        })

        print(f"\n{file_path.name}")
        print(f"  Size: {format_size(size)}")
        print(f"  Materials: {count:,}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTotal Materials Found: {total_materials:,}")
    print(f"Total Files: {len(all_files)}")

    total_size = sum(f.stat().st_size for f in all_files)
    print(f"Total Size: {format_size(total_size)}")

    # Sample data
    if file_stats:
        print("\n" + "=" * 80)
        print("SAMPLE DATA STRUCTURE")
        print("=" * 80)

        largest_file = Path(path) / file_stats[0]['path']
        sample = get_sample_material(largest_file)

        if sample:
            print(f"\nSample from: {file_stats[0]['path']}")
            print(json.dumps(sample, indent=2)[:1000] + "...")

    return {
        'path': str(path),
        'total_materials': total_materials,
        'total_files': len(all_files),
        'total_size': total_size,
        'file_stats': file_stats
    }


def count_materials_in_file(file_path: Path) -> int:
    """Count materials in a file"""
    try:
        # Handle gzipped files
        if file_path.suffix == '.gz':
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                return count_from_file_object(f, file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return count_from_file_object(f, file_path)
    except Exception as e:
        print(f"    ⚠️  Error reading: {e}")
        return 0


def count_from_file_object(f, file_path: Path) -> int:
    """Count from file object"""
    # JSONL format (one JSON per line)
    if file_path.suffix in ['.jsonl', '.ndjson'] or file_path.name.endswith('.jsonl.gz'):
        count = 0
        for line in f:
            if line.strip():
                count += 1
        return count

    # JSON format
    else:
        try:
            data = json.load(f)

            # If it's a list
            if isinstance(data, list):
                return len(data)

            # If it's a dict with materials array
            elif isinstance(data, dict):
                if 'materials' in data:
                    return len(data['materials'])
                elif 'data' in data:
                    return len(data['data'])
                else:
                    # Count top-level keys that look like material entries
                    return len([k for k in data.keys() if not k.startswith('_')])

            return 0
        except json.JSONDecodeError:
            # Try counting lines
            f.seek(0)
            return sum(1 for line in f if line.strip() and not line.strip().startswith('#'))


def get_sample_material(file_path: Path) -> Dict:
    """Get a sample material from file"""
    try:
        if file_path.suffix == '.gz':
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                return get_sample_from_file(f, file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return get_sample_from_file(f, file_path)
    except Exception as e:
        print(f"    ⚠️  Error getting sample: {e}")
        return {}


def get_sample_from_file(f, file_path: Path) -> Dict:
    """Get sample from file object"""
    # JSONL
    if file_path.suffix in ['.jsonl', '.ndjson'] or file_path.name.endswith('.jsonl.gz'):
        line = f.readline()
        if line.strip():
            return json.loads(line)

    # JSON
    else:
        data = json.load(f)
        if isinstance(data, list) and data:
            return data[0]
        elif isinstance(data, dict):
            if 'materials' in data and data['materials']:
                return data['materials'][0]
            elif 'data' in data and data['data']:
                return data['data'][0]
            else:
                # Return first non-metadata key
                for key, value in data.items():
                    if not key.startswith('_') and isinstance(value, dict):
                        return value

    return {}


def create_import_script(analysis: Dict, output_file: str = "import_mp_data.py"):
    """Create a script to import this data into QuLabInfinite"""

    script = f'''#!/usr/bin/env python3
"""
Import Materials Project Data to QuLabInfinite

Generated script to import {analysis['total_materials']:,} materials
from {analysis['path']}

Usage:
    python3 {output_file}
"""

import json
import gzip
import sys
from pathlib import Path
from typing import Dict, List

# Update this path to your QuLabInfinite installation
QULAB_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(QULAB_PATH))

from materials_lab.materials_database import MaterialsDatabase, MaterialProperties

def load_mp_materials(data_path: str) -> List[Dict]:
    """Load all MP materials from data directory"""
    materials = []

    data_dir = Path(data_path)

    # Process all JSON/JSONL files
    for file_path in data_dir.rglob("*.json*"):
        print(f"Loading {{file_path.name}}...")

        try:
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rt') as f:
                    materials.extend(load_from_file(f, file_path))
            else:
                with open(file_path, 'r') as f:
                    materials.extend(load_from_file(f, file_path))
        except Exception as e:
            print(f"  ⚠️  Error: {{e}}")

    return materials

def load_from_file(f, file_path):
    """Load materials from file object"""
    materials = []

    # JSONL format
    if '.jsonl' in file_path.name:
        for line in f:
            if line.strip():
                materials.append(json.loads(line))

    # JSON format
    else:
        data = json.load(f)
        if isinstance(data, list):
            materials = data
        elif isinstance(data, dict):
            if 'materials' in data:
                materials = data['materials']
            elif 'data' in data:
                materials = data['data']

    return materials

def convert_to_qulab_format(mp_material: Dict) -> MaterialProperties:
    """Convert MP material to QuLabInfinite format"""

    # Extract properties (adjust based on your MP data structure)
    return MaterialProperties(
        name=mp_material.get('formula_pretty', mp_material.get('formula')),
        category='unknown',  # Infer from properties
        subcategory='mp_material',
        density=mp_material.get('density', 0) * 1000,  # g/cm³ to kg/m³
        youngs_modulus=mp_material.get('elasticity', {{}}).get('K_VRH', 0),
        band_gap_ev=mp_material.get('band_gap', 0),
        notes=f"Materials Project ID: {{mp_material.get('material_id', 'unknown')}}"
    )

def main():
    print("Materials Project Data Import")
    print("=" * 80)

    # Load QuLabInfinite database
    print("\\nLoading QuLabInfinite database...")
    db = MaterialsDatabase()
    initial_count = db.get_count()
    print(f"Current materials: {{initial_count:,}}")

    # Load MP data
    print("\\nLoading MP materials...")
    mp_data = load_mp_materials("{analysis['path']}")
    print(f"Loaded {{len(mp_data):,}} materials from MP")

    # Convert and add to database
    print("\\nConverting to QuLabInfinite format...")
    added = 0
    for mp_mat in mp_data:
        try:
            qulab_mat = convert_to_qulab_format(mp_mat)
            db.materials[qulab_mat.name] = qulab_mat
            added += 1
        except Exception as e:
            print(f"  ⚠️  Error converting {{mp_mat.get('material_id')}}: {{e}}")

    print(f"\\n✓ Added {{added:,}} materials to database")

    # Save
    print("\\nSaving database...")
    db.save()

    final_count = db.get_count()
    print(f"\\n✓ Database now has {{final_count:,}} materials (was {{initial_count:,}})")

if __name__ == "__main__":
    main()
'''

    with open(output_file, 'w') as f:
        f.write(script)

    print(f"\n✓ Created import script: {output_file}")
    print(f"\nTo use it:")
    print(f"  1. Copy {output_file} to your QuLabInfinite directory")
    print(f"  2. Run: python3 {output_file}")


def main():
    """Main entry point"""

    # Get path from command line or use default
    if len(sys.argv) > 1:
        data_path = Path(sys.argv[1])
    else:
        data_path = Path("/noone/visualizer/qulab-infinite")

    print("\n" + "=" * 80)
    print("MATERIALS PROJECT DOWNLOAD ANALYZER")
    print("=" * 80)

    # Analyze
    analysis = analyze_directory(data_path)

    if analysis:
        # Create import script
        print("\n" + "=" * 80)
        print("GENERATING IMPORT SCRIPT")
        print("=" * 80)

        create_import_script(analysis)

        print("\n" + "=" * 80)
        print("✓ ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nYour Materials Project download contains:")
        print(f"  🔬 {analysis['total_materials']:,} materials")
        print(f"  📁 {analysis['total_files']} files")
        print(f"  💾 {format_size(analysis['total_size'])}")

        if analysis['total_materials'] > 1000000:
            print(f"\n🎉 YES! You have over 1 million materials!")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
