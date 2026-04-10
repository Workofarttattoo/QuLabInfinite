## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-22 - Floyd-Warshall Vectorization Optimization
**Learning:** The floyd_warshall method in algorithm_design_lab.py is bottlenecked by O(V^3) nested Python loops. This can be optimized drastically by replacing the inner two loops with a vectorized NumPy broadcasting approach (np.minimum(dist, dist[:, k:k+1] + dist[k:k+1, :])), reducing execution time for graph path calculations by orders of magnitude.
**Action:** When implementing algorithms over large adjacency matrices, always replace explicit nested loops with vectorized operations where possible, as pure Python loops impose severe overhead compared to NumPy's compiled backend.
