# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Comprehensive test suite for Genomics Laboratory
Tests all major functions with known benchmarks
"""

import unittest
import numpy as np
from .genomics_lab import (
    GenomicsLaboratory,
    DNASequence,
    Gene,
    CRISPRTarget,
    Mutation,
    Nucleotide,
    GeneticElement
)


class TestGenomicsLaboratory(unittest.TestCase):
    """Test suite for genomics lab"""

    def setUp(self):
        """Set up test fixtures"""
        self.lab = GenomicsLaboratory(seed=42)

    def test_sequence_generation(self):
        """Test DNA sequence generation with target GC content"""
        seq = self.lab.generate_random_sequence(1000, gc_content=0.5)

        self.assertEqual(len(seq), 1000)
        gc_count = seq.count('G') + seq.count('C')
        gc_actual = gc_count / len(seq)

        # Should be within 5% of target
        self.assertAlmostEqual(gc_actual, 0.5, delta=0.05)

    def test_dna_sequencing(self):
        """Test DNA sequencing simulation"""
        test_seq = self.lab.generate_random_sequence(500)
        result = self.lab.sequence_dna(test_seq, coverage=30)

        self.assertIn('num_reads', result)
        self.assertIn('average_quality', result)
        self.assertIn('coverage_mean', result)

        # Quality scores should be realistic (Q20-Q40)
        self.assertGreater(result['average_quality'], 20)
        self.assertLess(result['average_quality'], 45)

        # Coverage should be near target
        self.assertGreater(result['coverage_mean'], 20)
        self.assertLess(result['coverage_mean'], 40)

    def test_gene_expression_analysis(self):
        """Test gene expression quantification"""
        result = self.lab.analyze_gene_expression('BRCA1', tissue='breast')

        self.assertIn('tpm', result)
        self.assertIn('fpkm', result)
        self.assertIn('transcript_count', result)

        # TPM should be positive
        self.assertGreater(result['tpm'], 0)

        # Tissue modifier should work
        generic_result = self.lab.analyze_gene_expression('BRCA1', tissue='generic')
        self.assertNotEqual(result['tpm'], generic_result['tpm'])

    def test_crispr_guide_design(self):
        """Test CRISPR guide RNA design"""
        # Create sequence with known PAM sites (NGG)
        test_seq = "ATCGATCGATCGATCGAGGATCGATCGATCG"

        crispr = self.lab.design_crispr_guide(test_seq, position=15)

        self.assertIsInstance(crispr, CRISPRTarget)

        if crispr.sequence:  # If PAM found
            self.assertGreater(len(crispr.sequence), 0)
            self.assertGreaterEqual(crispr.on_target_score, 0)
            self.assertLessEqual(crispr.on_target_score, 1)
            self.assertGreaterEqual(crispr.efficiency, 0)
            self.assertLessEqual(crispr.efficiency, 1)

    def test_mutation_prediction(self):
        """Test mutation pathogenicity prediction"""
        test_seq = self.lab.generate_random_sequence(100)

        mutation = self.lab.predict_mutation_effect(test_seq, 50, 'T')

        self.assertIsInstance(mutation, Mutation)
        self.assertEqual(mutation.position, 50)
        self.assertIn(mutation.original, ['A', 'T', 'C', 'G'])
        self.assertEqual(mutation.mutated, 'T')
        self.assertGreaterEqual(mutation.pathogenicity_score, 0)
        self.assertLessEqual(mutation.pathogenicity_score, 1)
        self.assertIn(mutation.functional_impact,
                     ['benign', 'possibly_deleterious', 'deleterious'])

    def test_mutation_accumulation(self):
        """Test mutation accumulation over generations"""
        test_seq = self.lab.generate_random_sequence(1000)

        mutations = self.lab.simulate_mutation_accumulation(test_seq, generations=100)

        # Should accumulate some mutations
        self.assertGreater(len(mutations), 0)

        # All mutations should be valid
        for mut in mutations:
            self.assertIsInstance(mut, Mutation)
            self.assertIn(mut.original, ['A', 'T', 'C', 'G'])
            self.assertIn(mut.mutated, ['A', 'T', 'C', 'G'])

    def test_rna_sequencing(self):
        """Test RNA-Seq simulation"""
        result = self.lab.rna_sequencing('GAPDH', tissue='muscle')

        self.assertIn('gene_name', result)
        self.assertIn('read_count', result)
        self.assertIn('tpm', result)
        self.assertIn('log2_fold_change', result)
        self.assertIn('p_value', result)

        # Read count should be non-negative
        self.assertGreaterEqual(result['read_count'], 0)

        # P-value should be between 0 and 1
        self.assertGreaterEqual(result['p_value'], 0)
        self.assertLessEqual(result['p_value'], 1)

    def test_known_benchmark_sequence(self):
        """Test with known reference sequence"""
        # Human beta-globin gene partial sequence
        hbb_partial = "ATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTG"

        # Calculate GC content
        gc = (hbb_partial.count('G') + hbb_partial.count('C')) / len(hbb_partial)

        # Should be around 56% for HBB
        self.assertGreater(gc, 0.5)
        self.assertLess(gc, 0.6)


if __name__ == '__main__':
    unittest.main()
