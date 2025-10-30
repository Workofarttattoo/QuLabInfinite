#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

QuLabInfinite Materials Testing Runner
Runs comprehensive tests on curated materials list
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from materials_test_list import MaterialsTestList
from materials_lab.materials_database import MaterialsDatabase
from materials_lab.material_testing import (
    TensileTest, CompressionTest, FatigueTest, ImpactTest,
    HardnessTest, ThermalTest, CorrosionTest, EnvironmentalTest,
    TestResult
)


class MaterialsTestRunner:
    """Comprehensive materials testing runner for QuLabInfinite"""

    def __init__(self, iterations: int = 1):
        self.test_list = MaterialsTestList()
        self.db = MaterialsDatabase()
        self.iterations = iterations
        self.results = {}
        self.errors = {}

    def run_single_material_test(self, material_name: str) -> Dict[str, Any]:
        """Run comprehensive test suite on a single material"""
        material = self.db.get_material(material_name)
        if material is None:
            return {"error": f"Material not found: {material_name}"}

        test_results = {
            "material": material_name,
            "category": material.category,
            "tests": {}
        }

        # Run tensile test
        try:
            tensile = TensileTest(material)
            result = tensile.run()
            test_results["tests"]["tensile"] = {
                "success": result.success,
                "yield_strength": result.data.get("yield_strength", 0),
                "ultimate_strength": result.data.get("ultimate_strength", 0),
                "elongation": result.data.get("elongation_at_break", 0)
            }
        except Exception as e:
            test_results["tests"]["tensile"] = {"error": str(e)}

        # Run compression test
        try:
            compression = CompressionTest(material)
            result = compression.run()
            test_results["tests"]["compression"] = {
                "success": result.success,
                "compressive_strength": result.data.get("compressive_strength", 0)
            }
        except Exception as e:
            test_results["tests"]["compression"] = {"error": str(e)}

        # Run hardness test
        try:
            hardness = HardnessTest(material)
            result = hardness.run()
            test_results["tests"]["hardness"] = {
                "success": result.success,
                "vickers": result.data.get("vickers", 0),
                "rockwell_c": result.data.get("rockwell_c", 0)
            }
        except Exception as e:
            test_results["tests"]["hardness"] = {"error": str(e)}

        # Run thermal conductivity test
        try:
            thermal = ThermalTest(material)
            result = thermal.run_thermal_conductivity()
            test_results["tests"]["thermal"] = {
                "success": result.success,
                "thermal_conductivity": result.data.get("thermal_conductivity", 0)
            }
        except Exception as e:
            test_results["tests"]["thermal"] = {"error": str(e)}

        # For aerogels, run extreme cold test
        if "aerogel" in material.subcategory.lower():
            try:
                env_test = EnvironmentalTest(material)
                result = env_test.run_extreme_cold(
                    temperature=73,  # -200°C
                    wind_speed=13.4,  # 30 mph
                    duration_hours=24
                )
                test_results["tests"]["extreme_cold"] = {
                    "success": result.success,
                    "status": result.data.get("status", "unknown"),
                    "performance_factor": result.data.get("performance_factor", 0)
                }
            except Exception as e:
                test_results["tests"]["extreme_cold"] = {"error": str(e)}

        return test_results

    def run_category_tests(self, category: str, max_materials: int = None):
        """Run tests on all materials in a category"""
        materials = self.test_list.get_by_category(category)
        if max_materials:
            materials = materials[:max_materials]

        print(f"\n{'='*80}")
        print(f"Testing Category: {category.upper()}")
        print(f"Materials: {len(materials)}")
        print(f"{'='*80}")

        category_results = []
        for i, mat_name in enumerate(materials, 1):
            print(f"\n[{i}/{len(materials)}] Testing {mat_name}...", end=" ", flush=True)

            try:
                result = self.run_single_material_test(mat_name)
                if "error" in result:
                    print(f"❌ {result['error']}")
                    self.errors[mat_name] = result["error"]
                else:
                    # Count successful tests
                    success_count = sum(
                        1 for test_data in result["tests"].values()
                        if test_data.get("success", False)
                    )
                    print(f"✓ {success_count}/{len(result['tests'])} tests passed")
                    category_results.append(result)
                    self.results[mat_name] = result
            except Exception as e:
                print(f"❌ Error - {e}")
                self.errors[mat_name] = str(e)

        return category_results

    def run_all_tests(self, max_per_category: int = None):
        """Run tests on all categories"""
        print("\n" + "█"*80)
        print("█" + " "*78 + "█")
        print("█" + "  QuLabInfinite Materials Testing Suite".center(78) + "█")
        print("█" + f"  Total Materials: {len(self.test_list.get_all_test_materials())}".center(78) + "█")
        print("█" + f"  Iterations: {self.iterations}".center(78) + "█")
        print("█" + " "*78 + "█")
        print("█"*80)

        start_time = time.time()

        for iteration in range(1, self.iterations + 1):
            if self.iterations > 1:
                print(f"\n{'🔄 Iteration %d/%d' % (iteration, self.iterations)}")
                print("="*80)

            for category in self.test_list.test_materials.keys():
                self.run_category_tests(category, max_materials=max_per_category)

        elapsed = time.time() - start_time

        # Generate summary
        self.print_summary(elapsed)

    def print_summary(self, elapsed: float):
        """Print test summary"""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)

        total_materials = len(self.test_list.get_all_test_materials())
        tested = len(self.results)
        failed = len(self.errors)

        print(f"\nMaterials tested: {tested}/{total_materials}")
        print(f"Successful: {tested - failed} ({(tested-failed)/total_materials*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total_materials*100 if total_materials > 0 else 0:.1f}%)")
        print(f"Duration: {elapsed:.2f} seconds")

        if self.errors:
            print(f"\nFailed Materials:")
            for mat_name, error in self.errors.items():
                print(f"  ❌ {mat_name}: {error}")

        # Test type statistics
        test_stats = {}
        for mat_name, result in self.results.items():
            for test_type, test_data in result.get("tests", {}).items():
                if test_type not in test_stats:
                    test_stats[test_type] = {"success": 0, "total": 0}
                test_stats[test_type]["total"] += 1
                if test_data.get("success", False):
                    test_stats[test_type]["success"] += 1

        if test_stats:
            print(f"\nTest Type Statistics:")
            for test_type, stats in sorted(test_stats.items()):
                success_rate = stats["success"] / stats["total"] * 100 if stats["total"] > 0 else 0
                print(f"  {test_type}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")

        print("\n" + "="*80)

    def export_results(self, output_path: str = "materials_test_results.json"):
        """Export test results to JSON"""
        results_data = {
            "manifest_version": "1.0",
            "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "iterations": self.iterations,
            "total_materials": len(self.test_list.get_all_test_materials()),
            "tested": len(self.results),
            "failed": len(self.errors),
            "results": self.results,
            "errors": self.errors
        }

        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"\n[info] Results exported to: {output_path}")
        return output_path


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Run QuLabInfinite materials tests")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Number of iterations to run (default: 1)")
    parser.add_argument("--category", type=str, default=None,
                        help="Test only specific category (e.g., 'aerogels')")
    parser.add_argument("--max-per-category", type=int, default=None,
                        help="Maximum materials to test per category")
    parser.add_argument("--export", type=str, default="materials_test_results.json",
                        help="Export results to JSON file")
    args = parser.parse_args()

    runner = MaterialsTestRunner(iterations=args.iterations)

    if args.category:
        runner.run_category_tests(args.category, max_materials=args.max_per_category)
    else:
        runner.run_all_tests(max_per_category=args.max_per_category)

    if args.export:
        runner.export_results(args.export)

    # Return exit code based on results
    if runner.errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
