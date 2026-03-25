## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.
## 2024-05-24 - Particle Jet Clustering Vectorization
**Learning:** In physics simulations (like jet clustering), O(N^3) explicit nested loops are a massive performance bottleneck. Vectorized NumPy matrix sweeps calculating all pairwise distances simultaneously reduces complexity to essentially O(N^2) at C-speed, offering massive speedups (~150x).
**Action:** Always look for O(N^3) explicit Python loop bottlenecks in physical simulations involving N-body pairwise checks, and replace them with numpy vectorized matrix calculations.
