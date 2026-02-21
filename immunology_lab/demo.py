"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

IMMUNOLOGY LAB - Production Demo
"""

import numpy as np
from .immunology_lab import ImmunologyLab


def run_demo():
    """Demonstrate comprehensive immunology capabilities."""
    print("=" * 80)
    print("IMMUNOLOGY LAB - Production Demo")
    print("=" * 80)

    lab = ImmunologyLab()

    print("\n1. ANTIBODY REPERTOIRE GENERATION")
    print("-" * 40)
    repertoire = lab.generate_antibody_repertoire(diversity=100)
    print(f"Generated {len(repertoire)} unique antibodies")
    print(f"Example CDR3 length: {len(repertoire[0])} amino acids")

    print("\n2. ANTIBODY-ANTIGEN BINDING")
    print("-" * 40)
    antigen = np.random.randint(0, 20, 15)
    antibody = repertoire[0]
    affinity = lab.antibody_antigen_affinity(antibody, antigen)
    print(f"Binding affinity: {affinity:.4f}")
    print(f"Interpretation: {'High' if affinity > 0.5 else 'Low'} affinity")

    print("\n3. AFFINITY MATURATION")
    print("-" * 40)
    mature_ab, history = lab.affinity_maturation(antibody, antigen, generations=20)
    print(f"Initial affinity: {history[0]:.4f}")
    print(f"Final affinity: {history[-1]:.4f}")
    print(f"Improvement: {history[-1]/history[0]:.2f}x")

    print("\n4. T CELL ACTIVATION")
    print("-" * 40)
    signal1 = 0.7  # Strong TCR signal
    signal2 = 0.8  # Good costimulation
    signal3 = {'IL-2': 0.5, 'IFN-γ': 0.3}
    activation = lab.t_cell_activation(signal1, signal2, signal3)
    print(f"T cell activation level: {activation:.2%}")

    print("\n5. INFECTION RESPONSE")
    print("-" * 40)
    response = lab.simulate_infection_response(pathogen_load=1e6, duration_days=14)
    print(f"Peak pathogen load: {np.max(response['pathogen']):.2e}")
    print(f"Time to clearance: {np.argmax(response['pathogen'] < 1) / 24:.1f} days")
    print(f"Peak antibody level: {np.max(response['antibodies']):.0f}")
    print(f"Peak CD8 T cells: {np.max(response['cd8_t_cells']):.0f}")

    print("\n6. VACCINATION RESPONSE")
    print("-" * 40)
    for vaccine_type in ['protein', 'mRNA']:
        vacc = lab.simulate_vaccination(vaccine_type, doses=[0, 28], duration_days=180)
        print(f"\n{vaccine_type.upper()} Vaccine:")
        print(f"  Peak antibody titer: {np.max(vacc['antibody_titer']):.0f}")
        print(f"  Duration of protection (>50%): {np.sum(vacc['protection'] > 0.5)} days")
        print(f"  Memory B cells at day 180: {vacc['memory_b_cells'][-1]:.0f}")

    print("\n7. CYTOKINE NETWORK")
    print("-" * 40)
    initial = {'IL-2': 10, 'TNF-α': 5}
    cytokines = lab.model_cytokine_network(initial, time_steps=50)
    print(f"Initial stimulus: {initial}")
    print(f"Peak IL-2: {np.max(cytokines[:, lab.CYTOKINES.index('IL-2')]):.1f}")
    print(f"Peak IFN-γ: {np.max(cytokines[:, lab.CYTOKINES.index('IFN-γ')]):.1f}")

    print("\n8. MHC-PEPTIDE BINDING")
    print("-" * 40)
    peptide = np.array([11, 11, 5, 7, 9, 12, 14, 18, 17])  # LLFILVPWV-like
    binding = lab.mhc_peptide_binding(peptide, 'HLA-A*02:01')
    print(f"Peptide length: {len(peptide)} aa")
    print(f"Predicted binding affinity: {binding:.1f} nM")
    print(f"Classification: {'Strong' if binding < 50 else 'Weak' if binding < 500 else 'Non'}-binder")

    print("\n9. TCR REPERTOIRE DIVERSITY")
    print("-" * 40)
    tcr_analysis = lab.tcr_repertoire_diversity(repertoire_size=1000)
    print(f"Total TCRs: {tcr_analysis['total_sequences']}")
    print(f"Unique sequences: {tcr_analysis['unique_sequences']}")
    print(f"Shannon entropy: {tcr_analysis['shannon_entropy']:.2f}")
    print(f"Simpson diversity: {tcr_analysis['simpson_diversity']:.4f}")
    print(f"Clonality: {tcr_analysis['clonality']:.4f}")

    print("\n10. AUTOIMMUNE RESPONSE")
    print("-" * 40)
    self_antigen = np.random.randint(0, 20, 15)
    autoimmune = lab.simulate_autoimmune_response(self_antigen, tolerance_threshold=0.8)
    print(f"Autoreactive T cells: {autoimmune['autoreactive_cells']}")
    print(f"Regulatory response: {autoimmune['regulatory_response']:.2%}")
    print(f"Tissue damage: {autoimmune['tissue_damage']:.1f}%")
    print(f"Autoantibodies generated: {len(autoimmune['autoantibodies'])}")

    print("\n11. IMMUNOSENESCENCE")
    print("-" * 40)
    for age in [25, 50, 75]:
        aging = lab.model_immunosenescence(age)
        print(f"\nAge {age} years:")
        print(f"  Thymic output: {aging['thymic_output']:.1f}%")
        print(f"  Naive T cells: {aging['naive_t_cells']:.1f}%")
        print(f"  B cell diversity: {aging['b_cell_diversity']:.1f}%")
        print(f"  Inflammation: {aging['inflammation']:.1f}%")
        print(f"  Vaccine response: {aging['vaccine_response']:.1f}%")

    print("\n12. CANCER IMMUNOTHERAPY")
    print("-" * 40)
    for therapy in ['checkpoint', 'car_t', 'vaccine']:
        result = lab.simulate_immunotherapy(therapy, tumor_burden=1000, duration_weeks=12)
        print(f"\n{therapy.upper()} Therapy:")
        print(f"  Initial tumor: 1000 cells")
        print(f"  Final tumor: {result['tumor_size'][-1]:.0f} cells")
        print(f"  Peak CD8 infiltration: {np.max(result['cd8_infiltration']):.0f} cells")
        print(f"  Response: {result['response'].replace('_', ' ').title()}")

    print("\n13. COMPREHENSIVE ANALYSIS")
    print("-" * 40)
    comprehensive = lab.run_comprehensive_analysis()
    print("\nAnalysis Results:")
    for key, value in comprehensive.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 80)
    print("Immunology Lab demonstration complete!")
    print("=" * 80)


if __name__ == '__main__':
    run_demo()
