"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Protein Engineering Laboratory Demo
"""

from protein_engineering_lab import ProteinEngineeringLaboratory, SecondaryStructure
import numpy as np


def main():
    """Run protein engineering lab demonstration"""
    print("QuLabInfinite Protein Engineering Laboratory - Demo")
    print("=" * 70)

    lab = ProteinEngineeringLaboratory(seed=42)

    # Demo 1: Protein structure prediction
    print("\n1. Protein Structure Prediction (AlphaFold-style)")
    print("-" * 70)

    sequences = {
        'Insulin B-chain': "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
        'Short peptide': "AAALLLAAALL",
        'Mixed sequence': "KKEEKKEEAALLFFAA"
    }

    for name, seq in sequences.items():
        structure = lab.predict_protein_folding(seq, iterations=50)

        helix_pct = sum(1 for s in structure.secondary_structure
                       if s == SecondaryStructure.HELIX) / len(seq) * 100
        sheet_pct = sum(1 for s in structure.secondary_structure
                       if s == SecondaryStructure.SHEET) / len(seq) * 100

        print(f"\n{name} ({len(seq)} residues):")
        print(f"  Sequence: {seq}")
        print(f"  Energy: {structure.energy:.1f} kcal/mol")
        print(f"  Confidence: {structure.confidence:.3f}")
        print(f"  α-helix: {helix_pct:.1f}%")
        print(f"  β-sheet: {sheet_pct:.1f}%")

    # Demo 2: Enzyme kinetics comparison
    print("\n2. Enzyme Kinetics (Michaelis-Menten)")
    print("-" * 70)

    enzymes = [
        ("Super enzyme (low Km)", 100, 0.1),
        ("Efficient enzyme", 100, 1),
        ("Normal enzyme", 100, 10),
        ("Poor enzyme (high Km)", 100, 100)
    ]

    print(f"\n{'Enzyme':<25} {'Vmax':<12} {'Km':<12} {'kcat':<12} {'Efficiency'}")
    print("-" * 85)

    for name, vmax, km in enzymes:
        kinetics = lab.simulate_enzyme_kinetics(name, vmax, km)
        print(f"{name:<25} {kinetics.vmax:>10.1f} μM/s {kinetics.km:>10.2f} μM "
              f"{kinetics.kcat:>10.1f} s⁻¹ {kinetics.kcat_km:>12.2e} M⁻¹s⁻¹")

    # Demo 3: Protein-drug binding
    print("\n3. Protein-Drug Binding Affinity")
    print("-" * 70)

    protein_seq = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGG"

    binding_sites = [
        ("Hydrophobic pocket", [10, 11, 12, 13, 14, 15]),  # HLCGSH
        ("Charged pocket", [30, 31, 32, 33, 34]),  # ERGFF
        ("Mixed pocket", [20, 25, 30, 35, 40])
    ]

    for site_name, residues in binding_sites:
        binding = lab.calculate_binding_affinity(protein_seq, "Drug_X", residues)

        print(f"\n{site_name}:")
        print(f"  Kd: {binding.kd:.2f} nM " +
              ("(strong)" if binding.kd < 10 else "(moderate)" if binding.kd < 100 else "(weak)"))
        print(f"  ΔG: {binding.delta_g:.2f} kcal/mol")
        print(f"  kon: {binding.kon:.2e} M⁻¹s⁻¹")
        print(f"  koff: {binding.koff:.2e} s⁻¹")
        print(f"  Binding energy: {binding.binding_energy:.2f} kcal/mol")

    # Demo 4: Mutation effects
    print("\n4. Mutation Effect Prediction")
    print("-" * 70)

    test_seq = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGG"

    mutations = [
        (20, 'A', "Conservative (small→small)"),
        (20, 'W', "Large substitution"),
        (20, 'D', "Charge introduction"),
        (20, 'P', "Proline substitution"),
        (30, 'G', "Conservative"),
        (30, 'R', "Charge introduction")
    ]

    print(f"\n{'Mutation':<12} {'ΔΔG':<12} {'Function':<15} {'Impact':<12} {'Description'}")
    print("-" * 85)

    for pos, new_aa, description in mutations:
        if pos < len(test_seq):
            mutation = lab.predict_mutation_effect(test_seq, pos, new_aa)
            orig = mutation.original

            print(f"{orig}{pos+1}{new_aa:<10} "
                  f"{mutation.stability_change:>10.2f} kcal/mol "
                  f"{mutation.function_change*100:>12.0f}% "
                  f"{mutation.structural_impact:<12} {description}")

    # Demo 5: Stabilizing mutation design
    print("\n5. Rational Protein Engineering")
    print("-" * 70)

    print("\nDesigning stabilizing mutations for exposed hydrophobic residues...")

    stabilizing = lab.design_stabilizing_mutations(test_seq, n_mutations=5)

    if stabilizing:
        print(f"\n{'Mutation':<12} {'ΔΔG':<15} {'Function Retained':<20} {'Impact'}")
        print("-" * 70)

        for mut in stabilizing:
            print(f"{mut.original}{mut.position+1}{mut.mutated:<10} "
                  f"{mut.stability_change:>13.2f} kcal/mol "
                  f"{mut.function_change*100:>17.0f}% "
                  f"{mut.structural_impact}")
    else:
        print("No suitable stabilizing mutations found.")

    # Demo 6: Enzyme optimization
    print("\n6. Enzyme Engineering for Improved Catalysis")
    print("-" * 70)

    print("\nOptimizing enzyme by reducing Km (improving substrate binding):\n")

    original_km = 50  # μM
    vmax = 100  # μM/s

    km_values = [50, 25, 10, 5, 1]

    for km in km_values:
        kinetics = lab.simulate_enzyme_kinetics("optimized", vmax, km)

        improvement = (original_km / km - 1) * 100 if km < original_km else 0

        print(f"Km = {km:>5.1f} μM: "
              f"Efficiency = {kinetics.kcat_km:.2e} M⁻¹s⁻¹ "
              f"({improvement:>5.0f}% improvement)" if improvement > 0 else
              f"Km = {km:>5.1f} μM: "
              f"Efficiency = {kinetics.kcat_km:.2e} M⁻¹s⁻¹")

    # Demo 7: Binding affinity vs pocket composition
    print("\n7. Binding Pocket Composition Analysis")
    print("-" * 70)

    pocket_compositions = [
        ("All hydrophobic", "LLLLLL"),
        ("Mixed hydrophobic/polar", "LLSSLL"),
        ("All charged", "KKKKDD"),
        ("Balanced", "LKSKED")
    ]

    print(f"\n{'Pocket Type':<25} {'Kd (nM)':<12} {'ΔG (kcal/mol)':<15} {'Binding Strength'}")
    print("-" * 75)

    for pocket_name, pocket_seq in pocket_compositions:
        # Create sequence with this pocket
        full_seq = "AAA" + pocket_seq + "AAA" * 10
        binding = lab.calculate_binding_affinity(
            full_seq, "Ligand", [3, 4, 5, 6, 7, 8]
        )

        strength = "Very Strong" if binding.kd < 1 else \
                  "Strong" if binding.kd < 10 else \
                  "Moderate" if binding.kd < 100 else "Weak"

        print(f"{pocket_name:<25} {binding.kd:>10.2f} {binding.delta_g:>13.2f} {strength}")

    print("\n" + "=" * 70)
    print("Demo complete!")


if __name__ == "__main__":
    main()
