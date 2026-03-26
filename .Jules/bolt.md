## 2025-01-28 - [Vectorized Neighbor-Joining Q-matrix]
**Learning:** In `bioinformatics_lab.py`, calculating the neighbor-joining Q-matrix manually using nested loops over the distance matrix (`O(N^3)` operations due to `row_sum` and `col_sum` inside the inner loop) creates a severe performance bottleneck for large phylogenies.
**Action:** Replace manual row/col summing and matrix construction with O(N^2) vectorized NumPy broadcasting (`row_sums[:, np.newaxis] - row_sums[np.newaxis, :]`).
