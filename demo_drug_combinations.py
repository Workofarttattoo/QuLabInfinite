#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

QuLabInfinite Oncology Lab - Drug Combination Protocol Demo

Demonstrates how to experiment with various drug combinations including:
- FDA-approved chemotherapy
- Targeted therapies
- Immunotherapy
- Natural compounds & vitamins
- Experimental/off-label drugs
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
from oncology_lab.drug_response import list_available_drugs


def demo_combo_1_standard_plus_metabolic():
    """Demo 1: Standard chemo + metabolic support"""
    print("\n" + "=" * 80)
    print("  DEMO 1: Standard Chemotherapy + Metabolic Support")
    print("=" * 80)
    print("\nProtocol: Cisplatin + Metformin + Vitamin D3")
    print("Rationale: Metabolic inhibitors may sensitize tumors to chemotherapy\n")

    config = OncologyLabConfig(
        tumor_type=TumorType.BREAST_CANCER,
        stage=CancerStage.STAGE_III,
        initial_tumor_cells=100,
    )
    lab = OncologyLaboratory(config)

    # Apply ECH0 protocol
    protocol = create_ech0_three_stage_protocol()
    lab.apply_intervention_protocol(protocol)

    # Add standard chemo
    lab.administer_drug("cisplatin", 135.0)  # 75 mg/m² × 1.8 m²

    # Add metabolic support
    lab.administer_drug("metformin", 1000.0)  # 1000 mg oral
    lab.administer_drug("vitamin_d3", 5000.0)  # 5000 IU

    # Run experiment
    lab.run_experiment(duration_days=21, report_interval_hours=24*7)
    lab.print_summary()

    return lab.get_results()


def demo_combo_2_natural_cocktail():
    """Demo 2: Natural compound combination"""
    print("\n" + "=" * 80)
    print("  DEMO 2: Natural Compound Cocktail (Joe Tippens Protocol-Inspired)")
    print("=" * 80)
    print("\nProtocol: Fenbendazole + Curcumin + Vitamin C + Quercetin + EGCG")
    print("Rationale: Synergistic natural compounds with low toxicity\n")

    config = OncologyLabConfig(
        tumor_type=TumorType.COLORECTAL_CANCER,
        stage=CancerStage.STAGE_II,
        initial_tumor_cells=100,
    )
    lab = OncologyLaboratory(config)

    # Apply ECH0 protocol
    protocol = create_ech0_three_stage_protocol()
    lab.apply_intervention_protocol(protocol)

    # Natural compound stack
    lab.administer_drug("fenbendazole", 222.0)  # 222 mg
    lab.administer_drug("curcumin", 1000.0)  # 1g with piperine
    lab.administer_drug("vitamin_c", 5000.0)  # High dose oral
    lab.administer_drug("quercetin", 500.0)  # 500 mg
    lab.administer_drug("egcg", 400.0)  # Green tea extract

    # Run experiment
    lab.run_experiment(duration_days=30, report_interval_hours=24*7)
    lab.print_summary()

    return lab.get_results()


def demo_combo_3_targeted_plus_immune():
    """Demo 3: Targeted therapy + immunotherapy"""
    print("\n" + "=" * 80)
    print("  DEMO 3: Targeted Therapy + Immunotherapy")
    print("=" * 80)
    print("\nProtocol: Pembrolizumab + Vemurafenib (for BRAF+ Melanoma)")
    print("Rationale: Checkpoint inhibitors + targeted therapy combination\n")

    config = OncologyLabConfig(
        tumor_type=TumorType.MELANOMA,
        stage=CancerStage.STAGE_IV,
        initial_tumor_cells=100,
    )
    lab = OncologyLaboratory(config)

    # Apply ECH0 protocol
    protocol = create_ech0_three_stage_protocol()
    lab.apply_intervention_protocol(protocol)

    # Immunotherapy
    lab.administer_drug("pembrolizumab", 200.0)  # 200 mg IV

    # Targeted therapy (continuous oral)
    lab.administer_drug("vemurafenib", 960.0)  # 960 mg BID

    # Run experiment
    lab.run_experiment(duration_days=21, report_interval_hours=24*7)
    lab.print_summary()

    return lab.get_results()


def demo_combo_4_repurposed_drugs():
    """Demo 4: Repurposed drugs + DCA"""
    print("\n" + "=" * 80)
    print("  DEMO 4: Repurposed Drugs (Metabolic Approach)")
    print("=" * 80)
    print("\nProtocol: DCA + Ivermectin + Mebendazole + Hydroxychloroquine")
    print("Rationale: Multi-pathway metabolic targeting with repurposed drugs\n")

    config = OncologyLabConfig(
        tumor_type=TumorType.PANCREATIC_CANCER,
        stage=CancerStage.STAGE_III,
        initial_tumor_cells=100,
    )
    lab = OncologyLaboratory(config)

    # Apply ECH0 protocol
    protocol = create_ech0_three_stage_protocol()
    lab.apply_intervention_protocol(protocol)

    # Repurposed drug cocktail
    patient_weight_kg = 70.0
    lab.administer_drug("dichloroacetate", 25.0 * patient_weight_kg)  # 25 mg/kg
    lab.administer_drug("ivermectin", 0.2 * patient_weight_kg)  # 0.2 mg/kg
    lab.administer_drug("mebendazole", 100.0)  # 100 mg BID
    lab.administer_drug("hydroxychloroquine", 400.0)  # 400 mg

    # Add aspirin for COX-2 inhibition
    lab.administer_drug("aspirin", 325.0)  # 325 mg

    # Run experiment
    lab.run_experiment(duration_days=30, report_interval_hours=24*7)
    lab.print_summary()

    return lab.get_results()


def demo_combo_5_maximal_stack():
    """Demo 5: Kitchen sink approach (everything)"""
    print("\n" + "=" * 80)
    print("  DEMO 5: Maximal Multi-Modal Stack")
    print("=" * 80)
    print("\nProtocol: Chemo + Targeted + Immune + Natural + Repurposed")
    print("Warning: Hypothetical only - toxicity would be severe in reality!\n")

    config = OncologyLabConfig(
        tumor_type=TumorType.LUNG_CANCER,
        stage=CancerStage.STAGE_IV,
        initial_tumor_cells=100,
    )
    lab = OncologyLaboratory(config)

    # Apply ECH0 protocol
    protocol = create_ech0_three_stage_protocol()
    lab.apply_intervention_protocol(protocol)

    # Chemotherapy
    lab.administer_drug("carboplatin", 400.0)
    lab.administer_drug("paclitaxel", 175.0)

    # Targeted therapy
    lab.administer_drug("erlotinib", 150.0)

    # Immunotherapy
    lab.administer_drug("nivolumab", 240.0)

    # Metabolic inhibitors
    lab.administer_drug("metformin", 1000.0)
    lab.administer_drug("dichloroacetate", 25.0 * 70.0)

    # Natural compounds
    lab.administer_drug("curcumin", 1000.0)
    lab.administer_drug("quercetin", 500.0)
    lab.administer_drug("vitamin_d3", 5000.0)
    lab.administer_drug("vitamin_c", 5000.0)

    # Repurposed
    lab.administer_drug("ivermectin", 0.2 * 70.0)
    lab.administer_drug("fenbendazole", 222.0)

    # Anti-angiogenic
    lab.administer_drug("bevacizumab", 5.0 * 70.0)

    print("\n[Warning] This combination includes 15 drugs simultaneously!")
    print("[Note] Real-world toxicity would be prohibitive - simulation only\n")

    # Run experiment
    lab.run_experiment(duration_days=21, report_interval_hours=24*7)
    lab.print_summary()

    return lab.get_results()


def main():
    """Main demo runner"""
    print("\n" + "=" * 80)
    print("  QuLabInfinite - Drug Combination Protocol Demonstrations")
    print("=" * 80)

    print(f"\nAvailable drugs: {len(list_available_drugs())}")
    print("\nThis demo shows 5 different combination approaches:")
    print("  1. Standard chemo + metabolic support")
    print("  2. Natural compound cocktail")
    print("  3. Targeted + immunotherapy")
    print("  4. Repurposed drugs (metabolic)")
    print("  5. Maximal multi-modal stack")

    # Run all demos
    results = {}

    results['combo1'] = demo_combo_1_standard_plus_metabolic()
    results['combo2'] = demo_combo_2_natural_cocktail()
    results['combo3'] = demo_combo_3_targeted_plus_immune()
    results['combo4'] = demo_combo_4_repurposed_drugs()
    results['combo5'] = demo_combo_5_maximal_stack()

    # Summary comparison
    print("\n" + "=" * 80)
    print("  SUMMARY COMPARISON")
    print("=" * 80)

    print(f"\n{'Protocol':<40s} {'Final Cells':>15s} {'Viability':>12s}")
    print("-" * 80)

    for name, result in results.items():
        final_cells = result['final_stats']['alive_cells']
        viability = result['final_stats']['average_viability']
        print(f"{name:<40s} {final_cells:>15,d} {viability:>11.1%}")

    print("\n" + "=" * 80)
    print("Demo complete! Experiment with your own combinations.")
    print("Remember: These are heuristic simulations, not clinical predictions.\n")


if __name__ == "__main__":
    main()
