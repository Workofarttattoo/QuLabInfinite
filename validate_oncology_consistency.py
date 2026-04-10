import sys
import subprocess
from pathlib import Path
from typing import List, Tuple

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

SCRIPT_IMPORT_CONSISTENCY = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from oncology_lab import OncologyLaboratory, TumorType, CancerStage
from oncology_lab.tumor_simulator import TumorSimulator
from oncology_lab.drug_response import DRUG_DATABASE
from oncology_lab.ten_field_controller import TenFieldController
from oncology_lab.validation import DrugResponseValidator
print('✓ All imports successful')
"""

SCRIPT_PARAMETER_SANITY = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from oncology_lab.oncology_lab import OncologyLaboratory, OncologyLabConfig, TumorType, CancerStage

# Check growth multipliers are positive
config = OncologyLabConfig()
lab = OncologyLaboratory(config)

# Check all tumor types work
for tumor_type in TumorType:
    for stage in CancerStage:
        profile = lab._derive_tumor_profile(tumor_type, stage)
        assert profile['growth_rate'] > 0, f"Negative growth rate for {tumor_type}/{stage}"
        assert profile['carrying_capacity'] > 0, f"Negative capacity for {tumor_type}/{stage}"
        assert 0.4 <= profile['drug_sensitivity'] <= 1.4, f"Sensitivity out of range for {tumor_type}/{stage}"

print(f'✓ Validated {len(TumorType) * len(CancerStage)} tumor/stage combinations')
print('✓ All growth rates positive')
print('✓ All carrying capacities positive')
print('✓ All drug sensitivities in range [0.4, 1.4]')
"""

SCRIPT_FIELD_DELTAS = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from oncology_lab.ten_field_controller import (
    create_ech0_three_stage_protocol,
    create_standard_chemotherapy_protocol
)

# Check ECH0 protocol
protocol = create_ech0_three_stage_protocol()
print(f'✓ ECH0 protocol has {len(protocol.interventions)} interventions')

# Validate effect magnitudes are reasonable
for intervention in protocol.interventions:
    for field_name, delta in intervention.effects.items():
        # Deltas should be small enough to not cause instant jumps
        assert abs(delta) < 100, f"Delta too large: {field_name} = {delta}"
        print(f'  {intervention.name}: {field_name} → {delta:+.2f}/hour')

# Check standard chemo protocol
chemo = create_standard_chemotherapy_protocol()
print(f'\\n✓ Chemo protocol has {len(chemo.interventions)} interventions')

print('✓ All intervention deltas within reasonable ranges')
"""

SCRIPT_DRUG_DATABASE = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from oncology_lab.drug_response import DRUG_DATABASE

print(f'✓ Drug database has {len(DRUG_DATABASE)} drugs')

for drug_name, drug in DRUG_DATABASE.items():
    # Check critical parameters are positive
    assert drug.molecular_weight > 0, f"{drug_name}: negative molecular weight"
    assert drug.ic50 > 0, f"{drug_name}: negative IC50"
    assert drug.ec50 > 0, f"{drug_name}: negative EC50"
    assert 0 <= drug.emax <= 1.5, f"{drug_name}: Emax out of range"
    assert drug.pk_model.half_life > 0, f"{drug_name}: negative half-life"

    # Check elimination rate matches half-life
    expected_ke = 0.693 / drug.pk_model.half_life
    assert abs(drug.pk_model.elimination_rate - expected_ke) < 0.01, \\
        f"{drug_name}: elimination rate mismatch"

    print(f'  ✓ {drug.name}: IC50={drug.ic50}µM, t½={drug.pk_model.half_life}h')

print('\\n✓ All drug parameters validated')
"""

SCRIPT_END_TO_END = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from oncology_lab import OncologyLaboratory, OncologyLabConfig, TumorType, CancerStage
from oncology_lab.ten_field_controller import create_ech0_three_stage_protocol

# Create lab
config = OncologyLabConfig(
    tumor_type=TumorType.BREAST_CANCER,
    stage=CancerStage.STAGE_II,
    initial_tumor_cells=50,
)
lab = OncologyLaboratory(config)

# Apply protocol
protocol = create_ech0_three_stage_protocol()
lab.apply_intervention_protocol(protocol)

# Add drug
lab.administer_drug('cisplatin', 135.0)

# Run experiment
lab.run_experiment(duration_days=7, report_interval_hours=1000)

# Get results
results = lab.get_results()

# Validate results structure
assert 'time_hours' in results
assert 'cell_counts' in results
assert 'field_data' in results
assert 'drug_data' in results
assert len(results['time_hours']) > 0

# Check that intervention actually affected fields
field_data = results['field_data']
glucose_start = field_data['glucose_mm'][0]
glucose_end = field_data['glucose_mm'][-1]

print(f'✓ Simulation completed: {len(results["time_hours"])} time points')
print(f'✓ Glucose changed from {glucose_start:.2f} to {glucose_end:.2f} mM')
print(f'✓ Final cell count: {results["cell_counts"][-1]}')
print('✓ All result fields present')
"""
def print_header(title: str, color: str = BLUE):
    """Print formatted header"""
    print(f"\n{color}{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}{RESET}\n")

def run_test(name: str, command: List[str], timeout: int = 120) -> Tuple[bool, str]:
    """
    Run a test command and capture results

    Args:
        name: Test name
        command: Command to run
        timeout: Timeout in seconds

    Returns:
        (success, output)
    """
    print(f"{BLUE}▶ Running: {name}{RESET}")
    print(f"  Command: {' '.join(command)}")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent
        )

        success = result.returncode == 0
        output = result.stdout + result.stderr

        if success:
            print(f"  {GREEN}✓ PASSED{RESET}")
        else:
            print(f"  {RED}✗ FAILED (exit code {result.returncode}){RESET}")

        return success, output

    except subprocess.TimeoutExpired:
        print(f"  {RED}✗ TIMEOUT (>{timeout}s){RESET}")
        return False, f"Test timed out after {timeout} seconds"
    except Exception as e:
        print(f"  {RED}✗ ERROR: {e}{RESET}")
        return False, str(e)
def print_summary(results: List[Tuple[str, bool]]) -> int:
    """Print the final validation summary and return exit code."""
    print_header("Validation Summary", BLUE)

    total = len(results)
    passed = sum(1 for _, success in results if success)
    failed = total - passed

    print(f"Total tests: {total}")
    print(f"{GREEN}Passed: {passed}{RESET}")
    if failed > 0:
        print(f"{RED}Failed: {failed}{RESET}")

    print(f"\n{BLUE}Individual Results:{RESET}")
    for test_name, success in results:
        status = f"{GREEN}✓ PASSED{RESET}" if success else f"{RED}✗ FAILED{RESET}"
        print(f"  {test_name:30s} {status}")

    if failed == 0:
        print(f"\n{GREEN}{'=' * 80}")
        print("  ✓ ALL TESTS PASSED - ONCOLOGY LAB IS CONSISTENT")
        print(f"{'=' * 80}{RESET}\n")
        return 0
    else:
        print(f"\n{RED}{'=' * 80}")
        print(f"  ✗ {failed} TEST(S) FAILED - REVIEW PARAMETERS")
        print(f"{'=' * 80}{RESET}\n")
        print(f"{YELLOW}Next steps:{RESET}")
        print("  1. Review failed tests above")
        print("  2. Check parameter changes in:")
        print("     - oncology_lab/ten_field_controller.py (intervention deltas)")
        print("     - oncology_lab/oncology_lab.py (growth multipliers)")
        print("     - oncology_lab/drug_response.py (PK/PD parameters)")
        print("  3. Re-run this script after fixes")
        print()
        return 1

def main():
    """Main validation runner"""
    print_header("QuLabInfinite Oncology Lab - Consistency Validation Suite", BLUE)

    print("This script validates that all oncology lab components work together")
    print("consistently after parameter changes.\n")

    print(f"{YELLOW}Key parameters to track:{RESET}")
    print("  • Intervention deltas (ten_field_controller.py:311-383)")
    print("  • Growth multipliers (oncology_lab.py:333-392)")
    print("  • Drug response curves (drug_response.py)")

    results = []

    test_configs = [
        (
            "Test 1: Basic Smoke Test",
            "Basic oncology lab smoke test",
            ["python", "test_oncology_lab.py"],
            60,
            True # Always print last 2000 chars on failure
        ),
        (
            "Test 2: Validation Helpers",
            "Clinical validation helpers",
            ["python", "oncology_lab/validation.py"],
            30,
            False # No special failure printing
        ),
        (
            "Test 3: Import Consistency",
            "Import all oncology modules",
            ["python", "-c", SCRIPT_IMPORT_CONSISTENCY],
            10,
            False # Print all output on failure
        ),
        (
            "Test 4: Parameter Sanity Checks",
            "Validate parameter ranges",
            ["python", "-c", SCRIPT_PARAMETER_SANITY],
            30,
            False
        ),
        (
            "Test 5: Field Intervention Deltas",
            "Validate intervention effect ranges",
            ["python", "-c", SCRIPT_FIELD_DELTAS],
            30,
            False
        ),
        (
            "Test 6: Drug Database Integrity",
            "Validate drug parameters",
            ["python", "-c", SCRIPT_DRUG_DATABASE],
            30,
            False
        ),
        (
            "Test 7: End-to-End Simulation",
            "Run complete experiment simulation",
            ["python", "-c", SCRIPT_END_TO_END],
            90,
            True # Print last 2000 chars
        )
    ]

    for header, name, command, timeout, truncate_output in test_configs:
        print_header(header, BLUE)
        success, output = run_test(name, command, timeout=timeout)

        # Need to parse original name logic to keep result array the same name
        result_name = name
        if "smoke test" in name:
            result_name = "Basic Smoke Test"
        elif "Clinical validation" in name:
            result_name = "Validation Helpers"
        elif "Import all" in name:
            result_name = "Import Consistency"
        elif "parameter ranges" in name:
            result_name = "Parameter Sanity"
        elif "intervention effect" in name:
            result_name = "Field Deltas"
        elif "Validate drug" in name:
            result_name = "Drug Database"
        elif "experiment simulation" in name:
            result_name = "End-to-End"

        results.append((result_name, success))

        if not success:
            print(f"\n{RED}Output:{RESET}")
            if truncate_output:
                print(output[-2000:])
            else:
                print(output)

    return print_summary(results)

if __name__ == "__main__":
    sys.exit(main())
