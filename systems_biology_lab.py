"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
Systems Biology Lab - Multi-scale biological modeling
"""

import numpy as np
from typing import Dict, List

class SystemsBiologyLab:
    """Integrated biological systems modeling"""
    
    def __init__(self):
        self.network_size = 1000
        self.time_points = 100
        
    def simulate_gene_regulatory_network(self, n_genes: int = 50) -> np.ndarray:
        """Simulate gene regulatory network dynamics"""
        # Random interaction matrix
        W = np.random.randn(n_genes, n_genes) * 0.1
        np.fill_diagonal(W, -1)  # self-regulation
        
        # Initial expression
        expression = np.random.rand(n_genes)
        trajectory = []
        
        for _ in range(self.time_points):
            d_expression = W @ expression + np.random.randn(n_genes) * 0.01
            expression = np.maximum(0, expression + d_expression * 0.01)
            trajectory.append(expression.copy())
            
        return np.array(trajectory)
        
    def analyze_network_motifs(self, adjacency_matrix: np.ndarray) -> Dict:
        """
        Identify network motifs
        
        Performance optimization: Replaced O(N^3) nested Python loops with
        vectorized matrix multiplication operations. Computing paths of length
        2 via A @ A enables counting motifs in O(N^w) where w <= 3 (typically
        much faster in NumPy due to BLAS optimization).
        """
        # Ensure integer matrix and remove self-loops to prevent false motifs
        A = adjacency_matrix.astype(int).copy()
        np.fill_diagonal(A, 0)
        
        # A2[i, k] represents the number of paths of length 2 from i to k
        A2 = A @ A

        # Feed-forward loop: path of length 2 (A2) AND direct edge (A)
        # Element-wise multiplication counts overlapping paths
        feed_forward = np.sum(A * A2)

        # Feedback loop (3-cycle): path of length 2 from i to j, and edge from j to i
        # The trace of A^3 (or A2 @ A) counts all 3-cycles, each counted exactly 3 times.
        # However, the original logic iterated over i, j, k such that
        # A[i,j], A[j,k], A[k,i] were all true. Because it iterated all permutations,
        # it inherently counted each 3-cycle 3 times as well. So we match that behavior.
        feedback = np.trace(A2 @ A)
                                
        return {
            'feed_forward_loops': int(feed_forward),
            'feedback_loops': int(feedback),
            'total_edges': int(adjacency_matrix.sum())
        }
        
    def predict_cell_fate(self, initial_state: np.ndarray) -> str:
        """Predict cell differentiation fate"""
        # Simplified Waddington landscape
        attractors = {
            'stem': np.array([1, 1, 1]),
            'neuron': np.array([1, 0, 0]),
            'muscle': np.array([0, 1, 0]),
            'epithelial': np.array([0, 0, 1])
        }
        
        min_dist = float('inf')
        fate = 'unknown'
        
        for cell_type, attractor in attractors.items():
            dist = np.linalg.norm(initial_state[:3] - attractor)
            if dist < min_dist:
                min_dist = dist
                fate = cell_type
                
        return fate