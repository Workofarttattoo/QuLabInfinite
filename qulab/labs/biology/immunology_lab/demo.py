"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Immunology Laboratory Demo
"""

from immunology_lab import ImmunologyLaboratory
import numpy as np


def main():
    """Run immunology lab demonstration"""
    print("QuLabInfinite Immunology Laboratory - Demo")
    print("=" * 70)

    lab = ImmunologyLaboratory(seed=42)

    # Demo 1: Antibody-antigen binding kinetics
    print("\n1. Antibody-Antigen Binding Kinetics")
    print("-" * 70)

    affinities = ['high_affinity', 'medium_affinity', 'low_affinity']
    for affinity in affinities:
        binding = lab.simulate_antibody_antigen_binding(
            antibody_conc_nM=100,
            antigen_conc_nM=50,
            affinity=affinity,
            duration_s=3600
        )

        print(f"\n{affinity.replace('_', ' ').title()}:")
        print(f"  KD = {binding.KD*1e9:.2f} nM")
        print(f"  ka = {binding.ka:.2e} M^-1 s^-1")
        print(f"  kd = {binding.kd:.2e} s^-1")
        print(f"  Max binding: {np.max(binding.binding_curve)*100:.1f}%")

    # Demo 2: Immune response to infection
    print("\n2. Immune Response to Pathogen")
    print("-" * 70)

    doses = [1e4, 1e6, 1e8]
    for dose in doses:
        response = lab.simulate_immune_response(
            pathogen_dose=dose,
            pathogen_growth_rate=2.0,
            duration_days=21
        )

        peak_pathogen = np.max(response.pathogen_count)
        clearance_idx = np.where(response.pathogen_count < dose * 0.01)[0]
        clearance_day = response.time_days[clearance_idx[0]] if len(clearance_idx) > 0 else 21

        print(f"\nInitial dose: {dose:.0e} pathogens")
        print(f"  Peak pathogen: {peak_pathogen:.2e}")
        print(f"  Clearance time: {clearance_day:.1f} days")
        print(f"  Peak antibody: {np.max(response.antibody_titer):.2f} nM")
        print(f"  Peak T cells: {np.max(response.t_cell_count):.2e}")

    # Demo 3: Vaccine comparison
    print("\n3. Vaccine Efficacy Comparison")
    print("-" * 70)

    vaccines = ['mRNA', 'protein', 'inactivated', 'live_attenuated']

    for vaccine_type in vaccines:
        # Single dose
        single = lab.calculate_vaccine_efficacy(
            vaccine_type=vaccine_type,
            dose_schedule=[0],
            adjuvant=False
        )

        # Two doses with adjuvant
        double = lab.calculate_vaccine_efficacy(
            vaccine_type=vaccine_type,
            dose_schedule=[0, 28],
            adjuvant=True
        )

        print(f"\n{vaccine_type.replace('_', ' ').title()}:")
        print(f"  Single dose: {single.protection_rate*100:.1f}% protection, "
              f"{single.seroconversion_rate*100:.1f}% seroconversion")
        print(f"  Two doses + adj: {double.protection_rate*100:.1f}% protection, "
              f"{double.seroconversion_rate*100:.1f}% seroconversion")
        print(f"  GMT: {double.geometric_mean_titer:.0f}")
        print(f"  Duration: {double.duration_months:.0f} months")
        print(f"  Adverse events: {double.adverse_events_rate*100:.1f}%")

    # Demo 4: Autoimmune disease progression
    print("\n4. Autoimmune Disease Progression")
    print("-" * 70)

    diseases = ['rheumatoid_arthritis', 'lupus', 'multiple_sclerosis', 'type1_diabetes']
    treatments = [None, 'corticosteroid', 'dmard', 'biologic']

    for disease in diseases[:2]:  # Show first 2 diseases
        print(f"\n{disease.replace('_', ' ').title()}:")

        for treatment in treatments:
            sim = lab.simulate_autoimmune_disease(
                disease=disease,
                duration_months=12,
                treatment=treatment
            )

            treatment_name = treatment if treatment else "No treatment"
            print(f"  {treatment_name}:")
            print(f"    Avg severity: {sim['average_severity']:.1f}/100")
            print(f"    Flares: {sim['flare_count']}")
            print(f"    Remission: {sim['remission_months']}/12 months")

    # Demo 5: Cytokine storm simulation
    print("\n5. Immune Response Kinetics Detail")
    print("-" * 70)

    response = lab.simulate_immune_response(
        pathogen_dose=1e7,
        pathogen_growth_rate=1.5,
        duration_days=14
    )

    # Find peak cytokine day
    il6_levels = response.cytokine_levels['IL6']
    peak_day = response.time_days[np.argmax(il6_levels)]

    print(f"High-dose infection (1e7 pathogens):")
    print(f"  Peak IL-6 day: {peak_day:.1f}")
    print(f"  Peak IL-6 level: {np.max(il6_levels):.1f} pg/mL")
    print(f"  Peak IL-2 level: {np.max(response.cytokine_levels['IL2']):.1f} pg/mL")
    print(f"  Peak IFN-γ level: {np.max(response.cytokine_levels['IFNG']):.1f} pg/mL")

    # Demo 6: Dose-response relationship
    print("\n6. Antibody Dose-Response")
    print("-" * 70)

    antibody_concs = [1, 10, 100, 1000]  # nM
    print("Antigen concentration: 50 nM")

    for ab_conc in antibody_concs:
        binding = lab.simulate_antibody_antigen_binding(
            antibody_conc_nM=ab_conc,
            antigen_conc_nM=50,
            affinity='medium_affinity'
        )

        equilibrium_binding = binding.binding_curve[-1]
        print(f"  [{ab_conc:4d} nM Ab] → {equilibrium_binding*100:5.1f}% antigen bound")

    print("\n" + "=" * 70)
    print("Demo complete!")


if __name__ == "__main__":
    main()
