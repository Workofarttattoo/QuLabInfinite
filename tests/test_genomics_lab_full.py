import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add root directory to path to import genomics_lab_full
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestGenomicsLabFull(unittest.TestCase):

    def setUp(self):
        # We need to patch sys.modules inside the test context to avoid polluting
        # the global test runner state, which would break other tests that need real numpy/scipy.
        self.patcher_numpy = patch.dict('sys.modules', {'numpy': MagicMock()})
        self.patcher_scipy = patch.dict('sys.modules', {'scipy': MagicMock()})
        self.patcher_scipy_stats = patch.dict('sys.modules', {'scipy.stats': MagicMock()})

        self.patcher_numpy.start()
        self.patcher_scipy.start()
        self.patcher_scipy_stats.start()

        # We must import after starting the sys.modules patches
        # However, to be totally safe that other tests don't get the patched version
        # if they run later, we might need to be careful. But let's just import here.
        # Given it's a test for genomics_lab_full, this is the safest way to mock it without global pollution.
        if 'genomics_lab_full' in sys.modules:
            del sys.modules['genomics_lab_full']

    def tearDown(self):
        self.patcher_numpy.stop()
        self.patcher_scipy.stop()
        self.patcher_scipy_stats.stop()
        if 'genomics_lab_full' in sys.modules:
            del sys.modules['genomics_lab_full']

    @patch('genomics_lab_full.GenomicsLaboratory')
    def test_run_comprehensive_test(self, MockGenomicsLaboratory):
        # Import the module here so the sys.modules patch is active
        from genomics_lab_full import run_comprehensive_test

        # Set up the mock instance
        mock_lab_instance = MagicMock()
        MockGenomicsLaboratory.return_value = mock_lab_instance

        # Configure mock returns for lab methods
        mock_lab_instance.generate_random_sequence.return_value = "ATGC" * 250  # Length 1000
        mock_lab_instance.sequence_dna.return_value = {
            'num_reads': 15000,
            'average_quality': 38.5,
            'coverage_mean': 30.5
        }

        mock_lab_instance.analyze_gene_expression.return_value = {
            'gene': 'BRCA1',
            'expression_level': 100.5,
            'status': 'upregulated'
        }

        # Setup mock for CRISPR Design
        mock_crispr = MagicMock()
        mock_crispr.sequence = "ATCG" * 5 # Length 20
        mock_crispr.pam_site = "NGG"
        mock_crispr.on_target_score = 0.85
        mock_crispr.efficiency = 0.90
        mock_crispr.off_target_sites = 2
        mock_lab_instance.design_crispr_guide.return_value = mock_crispr

        # Setup mock for Mutation Prediction
        mock_mutation = MagicMock()
        mock_mutation.position = 100
        mock_mutation.original = "A"
        mock_mutation.mutated = "T"
        mock_mutation.pathogenicity_score = 0.75
        mock_mutation.functional_impact = "damaging"
        mock_lab_instance.predict_mutation_effect.return_value = mock_mutation

        # Setup mock for RNA-Seq
        mock_lab_instance.rna_sequencing.return_value = {
            'gene_name': 'BRCA1',
            'tissue': 'breast',
            'read_count': 1200,
            'tpm': 45.6,
            'log2_fold_change': 2.1,
            'p_value': 0.001,
            'significant': True
        }

        # Execute function
        results = run_comprehensive_test()

        # Verify initialization
        MockGenomicsLaboratory.assert_called_once_with(seed=42)

        # Verify method calls
        mock_lab_instance.generate_random_sequence.assert_called_once_with(1000, gc_content=0.5)
        mock_lab_instance.sequence_dna.assert_called_once_with("ATGC" * 250, coverage=30)
        mock_lab_instance.analyze_gene_expression.assert_called_once_with('BRCA1', tissue='breast')
        mock_lab_instance.design_crispr_guide.assert_called_once_with("ATGC" * 250, position=500)
        mock_lab_instance.predict_mutation_effect.assert_called_once_with("ATGC" * 250, 100, 'T')
        mock_lab_instance.rna_sequencing.assert_called_once_with('BRCA1', tissue='breast')

        # Verify result structure and content
        self.assertIn('sequencing', results)
        self.assertEqual(results['sequencing']['sequence_length'], 1000)
        self.assertEqual(results['sequencing']['num_reads'], 15000)
        self.assertEqual(results['sequencing']['avg_quality'], 38.5)
        self.assertEqual(results['sequencing']['coverage'], 30.5)

        self.assertIn('expression', results)
        self.assertEqual(results['expression']['gene'], 'BRCA1')
        self.assertEqual(results['expression']['expression_level'], 100.5)
        self.assertEqual(results['expression']['status'], 'upregulated')

        self.assertIn('crispr', results)
        self.assertEqual(results['crispr']['guide_length'], 20)
        self.assertEqual(results['crispr']['pam_site'], "NGG")
        self.assertEqual(results['crispr']['on_target_score'], 0.85)
        self.assertEqual(results['crispr']['efficiency'], 0.90)
        self.assertEqual(results['crispr']['off_targets'], 2)

        self.assertIn('mutation', results)
        self.assertEqual(results['mutation']['position'], 100)
        self.assertEqual(results['mutation']['change'], "A>T")
        self.assertEqual(results['mutation']['pathogenicity'], 0.75)
        self.assertEqual(results['mutation']['impact'], "damaging")

        self.assertIn('rna_seq', results)
        self.assertEqual(results['rna_seq']['gene_name'], 'BRCA1')
        self.assertEqual(results['rna_seq']['tissue'], 'breast')
        self.assertEqual(results['rna_seq']['read_count'], 1200)
        self.assertEqual(results['rna_seq']['tpm'], 45.6)
        self.assertEqual(results['rna_seq']['log2_fold_change'], 2.1)
        self.assertEqual(results['rna_seq']['p_value'], 0.001)
        self.assertTrue(results['rna_seq']['significant'])

if __name__ == '__main__':
    unittest.main()
