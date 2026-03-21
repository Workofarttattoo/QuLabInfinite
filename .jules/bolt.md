## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-23 - Jet Clustering Optimization
**Learning:** O(N^3) nested Python loops calculating pairwise physical distances (pseudorapidity, azimuth) and cross-sections per particle interaction are extremely slow due to repeated object method dispatch and scalar math.
**Action:** Hoist scalar object fields (e.g., FourVector px, py, pz) into NumPy arrays once per step, and calculate distances simultaneously using broadcasting (`x[:, np.newaxis] - x[np.newaxis, :]`), reducing overhead massively.
