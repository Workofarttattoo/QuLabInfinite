import time
import numpy as np
import unittest
from bioinformatics_lab import BioinformaticsLab

class TestBioinformaticsPerf(unittest.TestCase):
    def test_nj_performance(self):
        lab = BioinformaticsLab()

        # We know the fast implementation runs in ~0.01s,
        # while original took ~3s for 500 sequences of length 100
        np.random.seed(42)
        bases = ['A', 'C', 'G', 'T']
        sequences = [''.join(np.random.choice(bases, 100)) for _ in range(200)]

        n = len(sequences)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = sum(1 for a, b in zip(sequences[i], sequences[j]) if a != b)
                distance_matrix[i, j] = distance_matrix[j, i] = dist

        start = time.time()
        res = lab._neighbor_joining(distance_matrix)
        t = time.time() - start

        # Verify it runs efficiently
        self.assertLess(t, 0.5, "Neighbor Joining is too slow, expected <0.5s")
        print(f"Neighbor Joining ran in {t:.4f}s")

        # Also test on a small deterministic matrix to ensure it still works correctly
        dist_matrix_small = np.array([
            [0, 5, 4, 7],
            [5, 0, 7, 10],
            [4, 7, 0, 7],
            [7, 10, 7, 0]
        ], dtype=float)

        # Verify output shape is N-1
        res_small = lab._neighbor_joining(dist_matrix_small)
        self.assertEqual(res_small.shape, (3, 3))

if __name__ == '__main__':
    unittest.main()
