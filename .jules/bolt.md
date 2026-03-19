## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-03-05 - [Vectorizing Q-Matrix Calculation in Neighbor-Joining]
**Learning:** O(N^2) Python loops containing O(N) NumPy operations like `np.sum` can cause massive performance bottlenecks (O(N^3) overall). The `_neighbor_joining` algorithm in `bioinformatics_lab.py` suffered from this when calculating the Q-matrix and when reconstructing the distance matrix.
**Action:** Replace nested Python loops with fully vectorized NumPy operations (e.g., `row_sums[:, np.newaxis] - row_sums[np.newaxis, :]` and boolean masking `keep_mask[[i, j]] = False`) to drastically reduce execution time (from ~0.1s to ~0.003s for N=100) while maintaining identical mathematical output.
