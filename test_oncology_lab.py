"""
Basic smoke test for the QuLabInfinite oncology lab prototype.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from oncology_lab import (
    OncologyLaboratory,
    OncologyLabConfig,
    TumorType,
    CancerStage,
    TumorGrowthModel
)
from oncology_lab.ten_field_controller import create_ech0_three_stage_protocol

print("=" * 80)
print("QuLabInfinite Oncology Lab - Quick Test")
print("=" * 80)

# Create lab
print("\n[Test 1] Creating laboratory...")
config = OncologyLabConfig(
    tumor_type=TumorType.BREAST_CANCER,
    stage=CancerStage.STAGE_II,
    initial_tumor_cells=50,
    growth_model=TumorGrowthModel.GOMPERTZIAN,
)

lab = OncologyLaboratory(config)
print("✓ Laboratory created successfully")

# Test untreated tumor growth
print("\n[Test 2] Simulating 7 days of untreated tumor growth...")
lab.run_experiment(duration_days=7, report_interval_hours=24*7)
print("✓ Untreated simulation complete")

# Test drug administration
print("\n[Test 3] Administering cisplatin...")
lab.administer_drug("cisplatin", 135.0)
print("✓ Drug administered")

# Test field protocol
print("\n[Test 4] Applying ECH0 three-stage protocol...")
protocol = create_ech0_three_stage_protocol()
lab.apply_intervention_protocol(protocol)
lab.run_experiment(duration_days=7, report_interval_hours=24*7)
print("✓ Protocol applied and tested")

# Get final results
print("\n[Test 5] Retrieving results...")
results = lab.get_results()
print(f"✓ Results retrieved: {len(results['time_hours'])} time points")

# Print summary
print("\n" + "=" * 80)
lab.print_summary()

print("\n" + "=" * 80)
print("ALL TESTS PASSED ✓")
print("=" * 80)
print("\nPrototype smoke test completed.")
print("Remember: these heuristics are for experimentation, not clinical use.\n")
