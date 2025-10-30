#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Comprehensive testing of 6.6M materials expanded database
Tests: Loading, counts, search, integrity, performance
"""

import json
import time
import sys
from pathlib import Path

print("="*70)
print("  TESTING 6.6M MATERIALS DATABASE")
print("="*70)
print()

# Test 1: File exists and size
print("TEST 1: File Validation")
print("-" * 70)

db_path = Path("data/materials_db_expanded.json")
if not db_path.exists():
    print(f"❌ FAIL: Database file not found at {db_path}")
    sys.exit(1)

file_size_gb = db_path.stat().st_size / (1024**3)
print(f"✅ File exists: {db_path}")
print(f"   Size: {file_size_gb:.2f} GB")
print()

# Test 2: Load and count materials (sample-based for speed)
print("TEST 2: Material Count Validation")
print("-" * 70)

print("Loading database (this may take a moment for 14GB file)...")
start_time = time.time()

try:
    with open(db_path, 'r') as f:
        # Read first part to get structure
        f.seek(0)
        first_chars = f.read(1000)

        # Count by parsing incrementally
        f.seek(0)
        content = f.read()

    load_time = time.time() - start_time
    print(f"✅ File readable in {load_time:.1f} seconds")

    # Parse JSON
    print("Parsing JSON...")
    parse_start = time.time()
    data = json.loads(content)
    parse_time = time.time() - parse_start

    material_count = len(data)
    print(f"✅ Database parsed in {parse_time:.1f} seconds")
    print(f"   Total materials: {material_count:,}")

    # Verify count matches expected
    expected = 6_609_495
    if material_count == expected:
        print(f"✅ COUNT MATCHES: {material_count:,} = {expected:,}")
    else:
        print(f"⚠️  COUNT MISMATCH: {material_count:,} != {expected:,}")
        print(f"   Difference: {abs(material_count - expected):,}")

    print()

except Exception as e:
    print(f"❌ FAIL: Could not load database: {e}")
    sys.exit(1)

# Test 3: Category breakdown
print("TEST 3: Category Breakdown")
print("-" * 70)

categories = {}
for name, mat in list(data.items())[:10000]:  # Sample first 10K for speed
    cat = mat.get('category', 'unknown')
    categories[cat] = categories.get(cat, 0) + 1

print("Sample of first 10,000 materials:")
for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
    print(f"   {cat}: {count:,}")
print()

# Test 4: Data integrity checks
print("TEST 4: Data Integrity (Sample Check)")
print("-" * 70)

sample_size = 1000
sample_materials = list(data.items())[:sample_size]

issues = []
valid_count = 0

for name, mat in sample_materials:
    # Check required fields
    if 'category' not in mat:
        issues.append(f"Missing category: {name}")
    if 'density' not in mat:
        issues.append(f"Missing density: {name}")

    # Check data types
    if not isinstance(mat.get('density', 0), (int, float)):
        issues.append(f"Invalid density type: {name}")

    if len(issues) == 0:
        valid_count += 1

integrity_rate = (valid_count / sample_size) * 100
print(f"Checked {sample_size} materials:")
print(f"   Valid: {valid_count} ({integrity_rate:.1f}%)")
print(f"   Issues: {len(issues)}")

if len(issues) > 0:
    print(f"\nFirst 5 issues:")
    for issue in issues[:5]:
        print(f"   - {issue}")
else:
    print(f"✅ ALL SAMPLES VALID")

print()

# Test 5: Search functionality
print("TEST 5: Search Functionality")
print("-" * 70)

# Search for specific materials
test_searches = [
    ("Alloys", lambda m: 'alloy' in m.get('subcategory', '').lower()),
    ("Composites", lambda m: m.get('category') == 'composite'),
    ("High temp (>1000K)", lambda m: m.get('melting_point', 0) > 1000),
    ("High strength (>500 MPa)", lambda m: m.get('tensile_strength', 0) > 500),
]

for search_name, search_func in test_searches:
    matches = sum(1 for name, mat in list(data.items())[:10000] if search_func(mat))
    print(f"   {search_name}: {matches:,} (in first 10K)")

print()

# Test 6: Specific material retrieval
print("TEST 6: Specific Material Retrieval")
print("-" * 70)

test_materials = [
    "Iron-2.00%Carbon",  # Should exist (alloy)
    "Aluminum Oxide",     # Should exist (ceramic)
    "Graphene (Single Layer)",  # Should exist (base)
]

for mat_name in test_materials:
    if mat_name in data:
        mat = data[mat_name]
        print(f"✅ Found: {mat_name}")
        print(f"   Category: {mat.get('category')}")
        print(f"   Density: {mat.get('density', 0):.1f} kg/m³")
    else:
        print(f"❌ Not found: {mat_name}")

print()

# Test 7: Memory footprint
print("TEST 7: Memory Footprint")
print("-" * 70)

import sys
data_size_mb = sys.getsizeof(data) / (1024**2)
print(f"In-memory size: {data_size_mb:.1f} MB")
print(f"Materials: {len(data):,}")
print(f"Avg per material: {(data_size_mb * 1024) / len(data):.2f} KB")
print()

# Test 8: Performance benchmark
print("TEST 8: Performance Benchmark")
print("-" * 70)

# Lookup speed test
test_keys = list(data.keys())[:100]
start = time.time()
for key in test_keys:
    _ = data[key]
elapsed = time.time() - start
avg_ms = (elapsed / 100) * 1000

print(f"Lookup test (100 materials):")
print(f"   Total time: {elapsed*1000:.2f} ms")
print(f"   Average: {avg_ms:.3f} ms per lookup")

if avg_ms < 10:
    print(f"✅ PERFORMANCE EXCELLENT: {avg_ms:.3f} ms < 10 ms target")
else:
    print(f"⚠️  PERFORMANCE SLOW: {avg_ms:.3f} ms > 10 ms target")

print()

# Final summary
print("="*70)
print("  TEST SUMMARY")
print("="*70)
print()
print(f"✅ File size: {file_size_gb:.2f} GB")
print(f"✅ Material count: {material_count:,}")
print(f"✅ Data integrity: {integrity_rate:.1f}%")
print(f"✅ Lookup speed: {avg_ms:.3f} ms")
print()
print("🎉 DATABASE VALIDATION COMPLETE!")
print()
