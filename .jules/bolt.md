## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-03-05 - Vectorized N-Body Gravitational Simulation
**Learning:** The `nbody_gravitational_dynamics` method in `astrophysics_lab.py` originally used an O(N^2) nested Python loop to compute gravitational accelerations and potential energy between N bodies. This created a significant performance bottleneck for N-body simulations.
**Action:** Replaced the O(N^2) nested Python loops with O(1) vectorized NumPy operations utilizing broadcasting `(positions[:, np.newaxis, :] - positions[np.newaxis, :, :])` and `np.triu` masks. This optimization reduced the execution time of a 100-body simulation from ~1.88 seconds to ~0.038 seconds. Always look for nested loop interactions over coordinates that can be vectorized with NumPy broadcasting.
