"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Genomics Laboratory Demo
"""

from genomics_lab import GenomicsLaboratory
import json


def main():
    """Run genomics lab demonstration"""
    print("QuLabInfinite Genomics Laboratory - Demo")
    print("=" * 70)

    lab = GenomicsLaboratory(seed=42)

    # Demo 1: Generate and sequence DNA
    print("\n1. DNA Sequencing")
    print("-" * 70)
    sequence = lab.generate_random_sequence(5000, gc_content=0.48)
    print(f"Generated sequence: {len(sequence)} bp")
    print(f"First 100 bp: {sequence[:100]}")

    seq_results = lab.sequence_dna(sequence[:1000], coverage=50)
    print(f"Sequencing coverage: {seq_results['coverage_mean']:.1f}x")
    print(f"Average quality score: Q{seq_results['average_quality']:.1f}")
    print(f"Number of reads: {seq_results['num_reads']}")

    # Demo 2: Gene expression analysis
    print("\n2. Gene Expression Analysis")
    print("-" * 70)
    genes = ['TP53', 'BRCA1', 'EGFR', 'MYC', 'KRAS']
    tissues = ['brain', 'liver', 'lung', 'breast']

    for gene in genes[:3]:
        for tissue in tissues[:2]:
            expr = lab.analyze_gene_expression(gene, tissue)
            print(f"{gene} in {tissue}: {expr['tpm']:.2f} TPM ({expr['expression_category']})")

    # Demo 3: CRISPR guide design
    print("\n3. CRISPR-Cas9 Guide Design")
    print("-" * 70)
    target_seq = lab.generate_random_sequence(100, gc_content=0.5)
    crispr = lab.design_crispr_guide(target_seq, position=50)

    if crispr.sequence:
        print(f"Guide sequence: 5'-{crispr.sequence}-3'")
        print(f"PAM site: {crispr.pam_site}")
        print(f"GC content: {crispr.gc_content*100:.1f}%")
        print(f"On-target score: {crispr.on_target_score:.3f}")
        print(f"Predicted efficiency: {crispr.efficiency*100:.1f}%")
        print(f"Off-target sites: {crispr.off_target_sites}")
    else:
        print("No suitable PAM site found in target sequence")

    # Demo 4: Mutation effect prediction
    print("\n4. Mutation Effect Prediction")
    print("-" * 70)
    test_positions = [10, 50, 90]
    mutations_to_test = ['A', 'T', 'C', 'G']

    for pos in test_positions[:2]:
        original = sequence[pos]
        for mut_base in mutations_to_test:
            if mut_base != original:
                mutation = lab.predict_mutation_effect(sequence, pos, mut_base)
                print(f"Position {pos}: {mutation.original}>{mutation.mutated} - "
                      f"Pathogenicity: {mutation.pathogenicity_score:.3f} ({mutation.functional_impact})")
                break

    # Demo 5: Mutation accumulation simulation
    print("\n5. Mutation Accumulation Over Generations")
    print("-" * 70)
    short_seq = lab.generate_random_sequence(100000, gc_content=0.5)
    mutations = lab.simulate_mutation_accumulation(short_seq, generations=5000)

    deleterious = [m for m in mutations if m.functional_impact == "deleterious"]
    benign = [m for m in mutations if m.functional_impact == "benign"]

    print(f"Total mutations: {len(mutations)}")
    if mutations:
        print(f"Deleterious: {len(deleterious)} ({len(deleterious)/len(mutations)*100:.1f}%)")
        print(f"Benign: {len(benign)} ({len(benign)/len(mutations)*100:.1f}%)")
        print(f"Average pathogenicity: {sum(m.pathogenicity_score for m in mutations)/len(mutations):.3f}")
    else:
        print("No mutations accumulated.")

    # Demo 6: RNA-Seq experiment
    print("\n6. RNA-Seq Differential Expression")
    print("-" * 70)
    for gene in genes[:3]:
        rna = lab.rna_sequencing(gene, tissue='liver')
        print(f"{gene}: {rna['read_count']} reads, "
              f"log2FC: {rna['log2_fold_change']:.2f}, "
              f"p-value: {rna['p_value']:.2e}, "
              f"Significant: {rna['significant']}")

    print("\n" + "=" * 70)
    print("Demo complete!")


if __name__ == "__main__":
    main()
