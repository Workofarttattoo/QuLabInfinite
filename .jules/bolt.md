## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-22 - Floyd-Warshall Graph Path Vectorization
**Learning:** O(V³) algorithms like Floyd-Warshall can be incredibly slow due to Python's loop overhead in nested configurations. The inner two loops calculate distances between pairs (i, j) based on an intermediate node `k`. This pattern is perfectly suited for NumPy broadcasting.
**Action:** Always refactor O(N³) Python loops that perform identical independent distance calculations on elements into O(1) vectorized array operations like `np.minimum(dist, dist[:, k:k+1] + dist[k:k+1, :])` to massively reduce execution overhead.
