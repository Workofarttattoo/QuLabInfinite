## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-22 - Dynamic Programming Loop Optimization
**Learning:** Dense matrix dynamic programming algorithms (like Floyd-Warshall) heavily bottleneck on O(V³) nested pure-Python loops. However, they can be natively mapped to NumPy's C backend without loss of logic using intelligent array slice broadcasting (`dist = np.minimum(dist, dist[:, k:k+1] + dist[k:k+1, :])`).
**Action:** When examining 3-dimensional or highly nested loop constructs acting on regular grids/matrices, consider decomposing the inner N-1 loops into vectorized NumPy broadcasting updates.
