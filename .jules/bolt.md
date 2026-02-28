## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2026-02-28 - Multiple Sequence Alignment $O(N^3)$ Clustering Bottleneck
**Learning:** During UPGMA progressive alignment clustering, using list comprehensions like `np.mean([distances[a, b] for a in clusters[i] for b in clusters[j]])` inside a nested loop causes a massive $O(N^3)$ slowdown in Python due to overhead.
**Action:** Replace nested loops performing matrix reductions with NumPy's `np.ix_` indexing and vectorized operations (e.g., `np.mean(distances[np.ix_(cluster_i, cluster_j)])`) to push loops to C level and vastly speed up array-based graph algorithms.
