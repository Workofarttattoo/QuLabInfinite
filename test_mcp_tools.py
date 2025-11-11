#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Quick MCP Tools Test
====================
Tests that QuLab MCP server tools are working for live demo.

Usage:
    python3 test_mcp_tools.py
"""

import sys
sys.path.insert(0, '/Users/noone/QuLabInfinite')

def test_materials_database():
    """Test materials database access."""
    print("\n" + "="*70)
    print("TEST 1: Materials Database")
    print("="*70)

    try:
        from materials_lab.qulab_ai_integration import get_materials_database_info

        db_info = get_materials_database_info()
        print("✅ Materials database accessible")
        print(f"   Info: {db_info}")
        return True
    except Exception as e:
        print(f"❌ Materials database failed: {e}")
        return False

def test_element_validation():
    """Test element properties lookup."""
    print("\n" + "="*70)
    print("TEST 2: Element Validation")
    print("="*70)

    try:
        from physics_engine.thermodynamics import get_element_properties

        elements = ["Si", "C", "O", "H"]
        for elem in elements:
            props = get_element_properties(elem)
            print(f"✅ {elem}: {props}")
        return True
    except Exception as e:
        print(f"❌ Element validation failed: {e}")
        return False

def test_chemistry_tools():
    """Test chemistry validation."""
    print("\n" + "="*70)
    print("TEST 3: Chemistry Tools")
    print("="*70)

    try:
        from chemistry_lab.qulab_ai_integration import validate_smiles

        # Test water molecule
        result = validate_smiles("O")
        print(f"✅ SMILES validation: {result}")
        return True
    except Exception as e:
        print(f"❌ Chemistry tools failed: {e}")
        return False

def test_ai_calc():
    """Test AI calculator."""
    print("\n" + "="*70)
    print("TEST 4: AI Calculator")
    print("="*70)

    try:
        from qulab_ai.tools import calc

        result = calc("2 + 2")
        print(f"✅ Calculator: 2 + 2 = {result}")

        result2 = calc("sqrt(16)")
        print(f"✅ Calculator: sqrt(16) = {result2}")
        return True
    except Exception as e:
        print(f"❌ AI calc failed: {e}")
        return False

def main():
    """Run all MCP tool tests."""

    print("\n" + "="*70)
    print("QULAB MCP TOOLS TEST SUITE")
    print("="*70)
    print("\nTesting core MCP server functionality...\n")

    results = []

    # Run tests
    results.append(("Materials Database", test_materials_database()))
    results.append(("Element Validation", test_element_validation()))
    results.append(("Chemistry Tools", test_chemistry_tools()))
    results.append(("AI Calculator", test_ai_calc()))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! MCP server ready for demo.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check imports and dependencies.")
        return 1

if __name__ == "__main__":
    exit(main())
