
import sys
from unittest.mock import MagicMock
import unittest

# Mock numpy and scipy before importing bioinformatics_lab
sys.modules['numpy'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['scipy.spatial'] = MagicMock()
sys.modules['scipy.spatial.distance'] = MagicMock()
sys.modules['scipy.cluster'] = MagicMock()
sys.modules['scipy.cluster.hierarchy'] = MagicMock()

from bioinformatics_lab import BioinformaticsLab

class TestBioinformaticsLab(unittest.TestCase):
    def setUp(self):
        self.lab = BioinformaticsLab()

    def test_find_motifs_basic(self):
        sequence = "ATGCGATCG"
        motif = "ATGC"
        positions = self.lab.find_motifs(sequence, motif)
        self.assertEqual(positions, [0])

    def test_find_motifs_multiple(self):
        sequence = "ATGCGATCGATGC"
        motif = "ATGC"
        positions = self.lab.find_motifs(sequence, motif)
        self.assertEqual(positions, [0, 9])

    def test_find_motifs_wildcard(self):
        # The implementation is: re.compile(motif_pattern.replace('N', '.'))
        # So "ATNNA" becomes "AT..A".

        sequence = "ATGGAATGGA"
        motif = "ATNNA"
        positions = self.lab.find_motifs(sequence, motif)
        self.assertEqual(positions, [0, 5])

    def test_find_motifs_no_match(self):
        sequence = "AAAAAAA"
        motif = "TGC"
        positions = self.lab.find_motifs(sequence, motif)
        self.assertEqual(positions, [])

    def test_find_motifs_empty_sequence(self):
        sequence = ""
        motif = "ATGC"
        positions = self.lab.find_motifs(sequence, motif)
        self.assertEqual(positions, [])

    def test_find_motifs_empty_motif(self):
        # Empty motif behavior depends on regex. Usually matches empty string at every position.
        sequence = "ATG"
        motif = ""
        positions = self.lab.find_motifs(sequence, motif)
        # Regex "" matches at 0, 1, 2, 3 (empty strings between chars and at ends)
        self.assertEqual(positions, [0, 1, 2, 3])

if __name__ == '__main__':
    unittest.main()
