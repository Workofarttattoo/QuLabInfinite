"""
QuLabInfinite Oncology Lab - Comprehensive Demo.

Runs a sequence of illustrative experiments that highlight the capabilities and
limitations of the oncology sandbox. Outputs are synthetic and derived from
heuristic models; interpret them as educational examples rather than clinically
validated behaviour.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from oncology_lab import (
    OncologyLaboratory,
    OncologyLabConfig,
    TumorType,
    CancerStage,
    TumorGrowthModel
)
from oncology_lab.ten_field_controller import (
    create_ech0_three_stage_protocol,
    create_standard_chemotherapy_protocol
)
from oncology_lab.drug_response import list_available_drugs


def print_header(title: str):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def demo_1_untreated_tumor():
    """Demo 1: Observe untreated tumor growth"""
    print_header("DEMO 1: Untreated Tumor Growth (Baseline)")

    config = OncologyLabConfig(
        tumor_type=TumorType.BREAST_CANCER,
        stage=CancerStage.STAGE_II,
        initial_tumor_cells=100,
        growth_model=TumorGrowthModel.GOMPERTZIAN,
    )

    lab = OncologyLaboratory(config)

    print("\n[Info] Simulating 30 days of untreated tumor growth...")
    print("[Info] This establishes the baseline cancer progression")

    lab.run_experiment(duration_days=30, report_interval_hours=24*5)  # Report every 5 days

    results = lab.get_results()
    lab.print_summary()

    return results


def demo_2_standard_chemotherapy():
    """Demo 2: Standard chemotherapy treatment"""
    print_header("DEMO 2: Standard Chemotherapy (Cisplatin)")

    config = OncologyLabConfig(
        tumor_type=TumorType.BREAST_CANCER,
        stage=CancerStage.STAGE_II,
        initial_tumor_cells=100,
        growth_model=TumorGrowthModel.GOMPERTZIAN,
    )

    lab = OncologyLaboratory(config)

    print("\n[Info] Administering standard chemotherapy protocol")
    print("[Info] Cisplatin 75 mg/m² every 21 days x 4 cycles")

    # Administer cisplatin (standard dosing: 75 mg/m², assume 1.8 m² BSA = 135 mg)
    cisplatin_dose = 135.0  # mg

    # 4 cycles, 21 days apart
    for cycle in range(4):
        dose_time_days = cycle * 21
        print(f"\n[Cycle {cycle+1}] Administering {cisplatin_dose} mg cisplatin at day {dose_time_days}")

        # Run to dose time
        if cycle > 0:
            lab.run_experiment(duration_days=21, report_interval_hours=24*7)
        lab.administer_drug("cisplatin", cisplatin_dose)

    # Run final period
    lab.run_experiment(duration_days=21, report_interval_hours=24*7)

    results = lab.get_results()
    lab.print_summary()

    return results


def demo_3_ech0_protocol():
    """Demo 3: ECH0's 3-stage multi-field protocol"""
    print_header("DEMO 3: ECH0 Three-Stage Multi-Field Protocol")

    config = OncologyLabConfig(
        tumor_type=TumorType.BREAST_CANCER,
        stage=CancerStage.STAGE_II,
        initial_tumor_cells=100,
        growth_model=TumorGrowthModel.GOMPERTZIAN,
    )

    lab = OncologyLaboratory(config)

    print("\n[Info] Applying ECH0's comprehensive intervention protocol")
    print("[Info] Simultaneous targeting of all 10 biological fields")
    print("\n[Protocol Stages]:")
    print("  Stage 1 (Days 1-7):   Metabolic stress + immunosuppression")
    print("  Stage 2 (Days 7-21):  DNA damage + apoptosis induction")
    print("  Stage 3 (Days 21+):   Microenvironment disruption")

    protocol = create_ech0_three_stage_protocol()
    lab.apply_intervention_protocol(protocol)

    print("\n[Active Interventions]:")
    for intervention in protocol.interventions:
        print(f"  - {intervention.name} ({intervention.intervention_type.value})")
        print(f"    Targets: {', '.join(intervention.target_fields)}")

    # Add metformin for metabolic support (experimental)
    print("\n[Additional] Adding metformin for metabolic inhibition")
    lab.administer_drug("metformin", 1000.0)  # 1000 mg

    # Add DCA for Warburg effect reversal (experimental)
    patient_weight_kg = 70.0
    dca_dose_mg = 25.0 * patient_weight_kg  # 25 mg/kg
    print(f"[Additional] Adding dichloroacetate {dca_dose_mg} mg")
    lab.administer_drug("dichloroacetate", dca_dose_mg)

    lab.run_experiment(duration_days=60, report_interval_hours=24*5)

    results = lab.get_results()
    lab.print_summary()

    return results


def demo_4_compare_protocols():
    """Demo 4: Side-by-side comparison"""
    print_header("DEMO 4: Protocol Comparison Analysis")

    print("\n[Info] Running three parallel experiments for comparison:")
    print("  1. Untreated (control)")
    print("  2. Standard chemotherapy")
    print("  3. ECH0 multi-field protocol")

    # Run all three experiments
    print("\n" + "-" * 80)
    print("Experiment 1/3: Control")
    print("-" * 80)
    results_control = demo_1_untreated_tumor()

    print("\n" + "-" * 80)
    print("Experiment 2/3: Standard Chemotherapy")
    print("-" * 80)
    results_chemo = demo_2_standard_chemotherapy()

    print("\n" + "-" * 80)
    print("Experiment 3/3: ECH0 Protocol")
    print("-" * 80)
    results_ech0 = demo_3_ech0_protocol()

    # Comparison table
    print_header("COMPARISON RESULTS")

    print("\n{:<30s} {:>15s} {:>15s} {:>15s}".format(
        "Metric", "Control", "Chemotherapy", "ECH0 Protocol"
    ))
    print("-" * 80)

    # Final cell counts
    control_cells = results_control['final_stats']['alive_cells']
    chemo_cells = results_chemo['final_stats']['alive_cells']
    ech0_cells = results_ech0['final_stats']['alive_cells']

    print("{:<30s} {:>15,d} {:>15,d} {:>15,d}".format(
        "Final tumor cells", control_cells, chemo_cells, ech0_cells
    ))

    # Tumor volumes
    control_vol = results_control['final_stats']['tumor_volume_mm3']
    chemo_vol = results_chemo['final_stats']['tumor_volume_mm3']
    ech0_vol = results_ech0['final_stats']['tumor_volume_mm3']

    print("{:<30s} {:>14.2f}mm³ {:>14.2f}mm³ {:>14.2f}mm³".format(
        "Tumor volume", control_vol, chemo_vol, ech0_vol
    ))

    # Viability
    control_viab = results_control['final_stats']['average_viability']
    chemo_viab = results_chemo['final_stats']['average_viability']
    ech0_viab = results_ech0['final_stats']['average_viability']

    print("{:<30s} {:>14.1%} {:>14.1%} {:>14.1%}".format(
        "Average viability", control_viab, chemo_viab, ech0_viab
    ))

    # Progression scores
    control_score = results_control['progression_scores'][-1] if results_control['progression_scores'] else 100
    chemo_score = results_chemo['progression_scores'][-1] if results_chemo['progression_scores'] else 100
    ech0_score = results_ech0['progression_scores'][-1] if results_ech0['progression_scores'] else 100

    print("{:<30s} {:>14.1f}/100 {:>14.1f}/100 {:>14.1f}/100".format(
        "Cancer progression score", control_score, chemo_score, ech0_score
    ))

    # Efficacy comparison
    print("\n" + "-" * 80)
    print("EFFICACY ANALYSIS")
    print("-" * 80)

    chemo_reduction = (1.0 - chemo_cells / control_cells) * 100
    ech0_reduction = (1.0 - ech0_cells / control_cells) * 100

    print(f"\nTumor cell reduction vs. control:")
    print(f"  Chemotherapy: {chemo_reduction:+.1f}% {'(growth)' if chemo_reduction < 0 else '(reduction)'}")
    print(f"  ECH0 Protocol: {ech0_reduction:+.1f}% {'(growth)' if ech0_reduction < 0 else '(reduction)'}")

    if ech0_reduction > chemo_reduction:
        improvement = ech0_reduction - chemo_reduction
        print(f"\n  ✓ ECH0 Protocol is {improvement:.1f}% more effective than standard chemotherapy")
    elif chemo_reduction > ech0_reduction:
        difference = chemo_reduction - ech0_reduction
        print(f"\n  ✗ Standard chemotherapy is {difference:.1f}% more effective (unexpected)")
    else:
        print(f"\n  = Both protocols show equivalent efficacy")

    print("\n" + "=" * 80)

    # Create visualization
    try:
        create_comparison_plots(results_control, results_chemo, results_ech0)
        print("\n[Info] Comparison plots saved to 'oncology_lab_comparison.png'")
    except Exception as e:
        print(f"\n[Warning] Could not create plots: {e}")


def create_comparison_plots(results_control, results_chemo, results_ech0):
    """Create comparison visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Tumor cell count over time
    ax = axes[0, 0]
    ax.plot(results_control['time_days'], results_control['cell_counts'],
            'r-', label='Control (untreated)', linewidth=2)
    ax.plot(results_chemo['time_days'], results_chemo['cell_counts'],
            'b-', label='Chemotherapy', linewidth=2)
    ax.plot(results_ech0['time_days'], results_ech0['cell_counts'],
            'g-', label='ECH0 Protocol', linewidth=2)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Tumor Cells')
    ax.set_title('Tumor Growth Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 2: Cancer progression score
    ax = axes[0, 1]
    ax.plot(results_control['time_days'], results_control['progression_scores'],
            'r-', label='Control', linewidth=2)
    ax.plot(results_chemo['time_days'], results_chemo['progression_scores'],
            'b-', label='Chemotherapy', linewidth=2)
    ax.plot(results_ech0['time_days'], results_ech0['progression_scores'],
            'g-', label='ECH0 Protocol', linewidth=2)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Cancer Progression Score (0-100)')
    ax.set_title('Cancer Progression Score Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])

    # Plot 3: Field values (ECH0 protocol)
    ax = axes[1, 0]
    field_data = results_ech0['field_data']
    ax.plot(results_ech0['time_days'], field_data['glucose_mm'], label='Glucose (mM)')
    ax.plot(results_ech0['time_days'], field_data['oxygen_percent'], label='Oxygen (%)')
    ax.plot(results_ech0['time_days'], [v/10 for v in field_data['cytokine_pg_ml']], label='Cytokines (×10 pg/mL)')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Field Values')
    ax.set_title('Key Biological Fields (ECH0 Protocol)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Viability comparison
    ax = axes[1, 1]
    ax.plot(results_control['time_days'], results_control['viabilities'],
            'r-', label='Control', linewidth=2)
    ax.plot(results_chemo['time_days'], results_chemo['viabilities'],
            'b-', label='Chemotherapy', linewidth=2)
    ax.plot(results_ech0['time_days'], results_ech0['viabilities'],
            'g-', label='ECH0 Protocol', linewidth=2)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Average Cell Viability')
    ax.set_title('Tumor Cell Viability Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig('oncology_lab_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def demo_5_drug_database():
    """Demo 5: Show available drugs"""
    print_header("DEMO 5: Available Drug Database")

    drugs = list_available_drugs()

    print(f"\n[Info] QuLabInfinite Oncology Lab includes {len(drugs)} drugs with real clinical parameters")
    print("\nAvailable Drugs:")

    from oncology_lab.drug_response import DRUG_DATABASE

    for drug_name in sorted(drugs):
        drug = DRUG_DATABASE[drug_name]
        print(f"\n  • {drug.name} ({drug.generic_name})")
        print(f"    Class: {drug.drug_class.value}")
        print(f"    FDA Approved: {'Yes' if drug.fda_approved else 'No (Experimental)'}")
        if drug.approval_year:
            print(f"    Approval Year: {drug.approval_year}")
        print(f"    Mechanism: {drug.mechanism_of_action}")
        print(f"    IC50: {drug.ic50} μM")
        if drug.approved_indications:
            print(f"    Indications: {', '.join(drug.approved_indications[:3])}")


def main():
    """Main demo entry point"""
    print_header("QuLabInfinite Oncology Laboratory - Comprehensive Demo")

    print("\nWelcome to QuLabInfinite's Oncology Lab!")
    print("This demonstration walks through the prototype tumour-growth sandbox")
    print("and how it can be exercised with treatments and intervention protocols.")

    print("\n[Features]:")
    print("  ✓ Individual cancer cell simulation")
    print("  ✓ Heuristic tumour microenvironment (10 biological fields)")
    print("  ✓ Reference profiles for selected FDA-approved and experimental drugs")
    print("  ✓ Example intervention protocols (including prior ECH0 concepts)")
    print("  ✓ Optional comparison against published clinical benchmarks")

    print("\n[Demos Available]:")
    print("  1. Untreated tumor growth (baseline)")
    print("  2. Standard chemotherapy")
    print("  3. ECH0 multi-field protocol")
    print("  4. Comprehensive comparison")
    print("  5. Drug database overview")

    # Run comprehensive comparison
    demo_4_compare_protocols()

    # Show drug database
    demo_5_drug_database()

    print_header("Demo Complete!")

    print("\n[Next Steps]:")
    print("  • Experiment with different tumor types")
    print("  • Test combination therapies")
    print("  • Validate against clinical trial data")
    print("  • Optimize protocols for specific cancers")

    print("\nThank you for exploring the QuLabInfinite oncology lab prototype!")
    print("The future of cancer research is computational—but these simulations")
    print("remain exploratory scaffolding, not substitutes for wet-lab evidence.\n")


if __name__ == "__main__":
    main()
