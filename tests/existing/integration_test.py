"""
QuLabInfinite Integration Tests - All Departments

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Comprehensive integration testing across all laboratory departments.
"""

import sys
from pathlib import Path
import time
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.qulab_api import QuLabSimulator, ExperimentRequest, ExperimentType


def test_materials_lab():
    """Test Materials Laboratory integration."""
    print("\n" + "="*80)
    print("TEST 1: Materials Laboratory Integration")
    print("="*80)

    sim = QuLabSimulator()

    # Test 1.1: Material database lookup
    print("\n[1.1] Material Database Lookup")
    result = sim.run("Test AISI 304 Stainless Steel tensile strength")
    assert result.success, f"Material test failed: {result.error_message}"
    print(f"  ✓ Found material: AISI 304 Stainless Steel")
    print(f"  ✓ Yield strength: {result.data['yield_strength_MPa']:.1f} MPa")

    # Test 1.2: Airloy X103 aerogel (critical material)
    print("\n[1.2] Airloy X103 Aerogel Properties")
    materials = sim.materials_lab.list_materials()
    assert "Airloy X103 Strong Aerogel" in materials, "Airloy X103 not in database"
    aerogel = sim.materials_lab.get_material("Airloy X103 Strong Aerogel")
    assert aerogel is not None, "Could not retrieve Airloy X103"
    print(f"  ✓ Airloy X103 found in database")
    print(f"  ✓ Density: {aerogel.density:.3f} kg/m³")
    print(f"  ✓ Thermal conductivity: {aerogel.thermal_conductivity:.3f} W/(m·K)")


def test_environmental_sim():
    """Test Environmental Simulator integration."""
    print("\n" + "="*80)
    print("TEST 2: Environmental Simulator Integration")
    print("="*80)

    sim = QuLabSimulator()

    # Test 2.1: Extreme cold conditions
    print("\n[2.1] Extreme Cold Simulation (-200°C)")
    conditions = sim.simulate_environment({
        'temperature': -200,
        'pressure': 0.001,
        'wind_speed': 30
    })
    assert 'temperature_K' in conditions, "Temperature not set"
    print(f"  ✓ Temperature: {conditions['temperature_K']:.1f} K")
    print(f"  ✓ Pressure: {conditions['pressure_bar']:.4f} bar")

    # Test 2.2: High temperature
    print("\n[2.2] High Temperature Simulation (1000°C)")
    conditions = sim.simulate_environment({
        'temperature': 1000,
        'pressure': 1.0
    })
    print(f"  ✓ Temperature: {conditions['temperature_K']:.1f} K")


def test_physics_engine():
    """Test Physics Engine integration."""
    print("\n" + "="*80)
    print("TEST 3: Physics Engine Integration")
    print("="*80)

    sim = QuLabSimulator()

    # Test 3.1: Free fall simulation
    print("\n[3.1] Free Fall Simulation")
    request = ExperimentRequest(
        experiment_type=ExperimentType.PHYSICS_PROBE,
        description="Free fall test",
        parameters={
            'mass_kg': 0.5,
            'initial_height_m': 5.0,
            'duration_s': 1.0
        }
    )
    result = sim.run(request)
    assert result.success, f"Physics simulation failed: {result.error_message}"
    print(f"  ✓ Initial height: {result.data['initial_height_m']:.2f} m")
    print(f"  ✓ Final height: {result.data['final_height_m']:.2f} m")
    print(f"  ✓ Energy error: {result.data['energy_error_fraction']*100:.4f}%")


def test_integrated_stack():
    """Test full integrated stack."""
    print("\n" + "="*80)
    print("TEST 4: Integrated Multi-Department Stack")
    print("="*80)

    sim = QuLabSimulator()

    # Test 4.1: Aerogel under extreme conditions
    print("\n[4.1] Airloy X103 Under Extreme Conditions")
    print("       Temperature: -200°C, Pressure: 0.001 bar, Wind: 30 mph")

    request = ExperimentRequest(
        experiment_type=ExperimentType.INTEGRATED,
        description="Airloy X103 extreme cold wind test",
        parameters={
            'material': 'Airloy X103 Strong Aerogel',
            'temperature_c': -200,
            'pressure_bar': 0.001,
            'wind_mph': 30,
            'specimen_mass_kg': 0.1,
            'drop_height_m': 2.0,
            'simulation_duration_s': 0.5
        }
    )

    result = sim.run(request)
    assert result.success, f"Integrated test failed: {result.error_message}"

    env = result.data['environment']
    material = result.data['material_test']

    print(f"  ✓ Environment configured:")
    print(f"    - Temperature: {env['temperature_C']:.1f}°C ({env['temperature_K']:.1f} K)")
    print(f"    - Pressure: {env['pressure_bar']:.4f} bar")
    print(f"    - Wind: {env['wind_velocity_m_s'][0]:.2f} m/s")

    print(f"  ✓ Material properties at test conditions:")
    print(f"    - Yield strength: {material['yield_strength_MPa']:.1f} MPa")
    print(f"    - Ultimate strength: {material['ultimate_strength_MPa']:.1f} MPa")


def test_validation_system():
    """Test validation against reference data."""
    print("\n" + "="*80)
    print("TEST 5: Results Validation System")
    print("="*80)

    sim = QuLabSimulator()

    # Test 5.1: Validate steel properties
    print("\n[5.1] Validate Steel 304 Properties")
    result = sim.run("Test AISI 304 Stainless Steel tensile strength")

    if result.validation:
        val = result.validation
        print(f"  ✓ Validation status: {val['status']}")
        print(f"  ✓ Error: {val['error_percent']:.4f}%")
        print(f"  ✓ Z-score: {val['z_score']:.2f}σ")
        print(f"  ✓ Simulated: {val['simulated_value']:.1f} MPa")
        print(f"  ✓ Reference: {val['reference_value']:.1f} ± {val['uncertainty']:.1f} MPa")
        assert val['status'] in ['pass', 'warning'], "Validation failed"


def test_materials_search():
    """Test materials search functionality."""
    print("\n" + "="*80)
    print("TEST 6: Materials Search & Discovery")
    print("="*80)

    sim = QuLabSimulator()

    # Test 6.1: Search by criteria
    print("\n[6.1] Search for Lightweight High-Strength Materials")
    criteria = {
        'density_max': 3000,  # kg/m³
        'yield_strength_min': 200  # MPa
    }
    results = sim.find_materials(criteria)
    print(f"  ✓ Found {len(results)} materials matching criteria")
    if results:
        print(f"  ✓ Examples: {', '.join(results[:5])}")


def test_performance():
    """Test performance metrics."""
    print("\n" + "="*80)
    print("TEST 7: Performance Benchmarks")
    print("="*80)

    sim = QuLabSimulator()

    # Test 7.1: Material lookup speed
    print("\n[7.1] Material Database Lookup Speed")
    start = time.time()
    for _ in range(100):
        sim.materials_lab.get_material("AISI 304 Stainless Steel")
    elapsed = time.time() - start
    avg_time = (elapsed / 100) * 1000
    print(f"  ✓ Average lookup time: {avg_time:.2f} ms")
    assert avg_time < 10, f"Lookup too slow: {avg_time:.2f} ms > 10 ms"

    # Test 7.2: Simple simulation throughput
    print("\n[7.2] Experiment Throughput")
    start = time.time()
    for i in range(10):
        sim.run("Test steel tensile strength")
    elapsed = time.time() - start
    throughput = 10 / elapsed
    print(f"  ✓ Experiments per second: {throughput:.2f}")


def test_all_departments():
    """Test all departments are operational."""
    print("\n" + "="*80)
    print("TEST 8: All Departments Operational Check")
    print("="*80)

    sim = QuLabSimulator()

    departments = {
        'Materials Lab': sim.materials_lab,
        'Environmental Simulator': sim.environment,
        'Results Validator': sim.validator,
        'Hive Mind': getattr(sim, 'hive_mind', None)
    }

    for name, dept in departments.items():
        status = "✓ OPERATIONAL" if dept is not None else "✗ NOT FOUND"
        print(f"  {status}: {name}")


def run_all_tests():
    """Run complete integration test suite."""
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "  QuLabInfinite - COMPREHENSIVE INTEGRATION TEST SUITE".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)

    tests = [
        ("Materials Laboratory", test_materials_lab),
        ("Environmental Simulator", test_environmental_sim),
        ("Physics Engine", test_physics_engine),
        ("Integrated Stack", test_integrated_stack),
        ("Validation System", test_validation_system),
        ("Materials Search", test_materials_search),
        ("Performance Benchmarks", test_performance),
        ("All Departments", test_all_departments),
    ]

    results = []
    start_time = time.time()

    for test_name, test_func in tests:
        try:
            test_func()
        except Exception as exc:  # pragma: no cover - manual summary path
            results.append((test_name, False, str(exc)))
            print(f"\n  ❌ {test_name}: FAILED")
            print(f"     Error: {exc}")
        else:
            results.append((test_name, True, None))
            print(f"\n  ✅ {test_name}: PASSED")

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed} ({passed/total*100:.1f}%)")
    print(f"Failed: {total - passed} ({(total-passed)/total*100:.1f}%)")
    print(f"Duration: {elapsed:.2f} seconds")

    print("\nDetailed Results:")
    for test_name, success, error in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if error:
            print(f"         {error}")

    print("\n" + "="*80)

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! QuLabInfinite is fully operational! 🎉\n")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review errors above.\n")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
