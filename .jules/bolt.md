## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-22 - N-Body Gravity Vectorization and Fallbacks
**Learning:** Pure Python nested loops `O(N^2)` for N-body gravity in `physics_engine/mechanics.py` severely impact simulation performance. NumPy broadcasting and vectorization provide massive speedups for large N (e.g., ~19s to ~0.3s for 500 particles over 10 steps). However, creating new NumPy arrays in hot loops introduces significant overhead for small N, causing performance regressions.
**Action:** When vectorizing `O(N^2)` operations over objects, provide a fallback to pure Python loops for small N (e.g., `N < 50`) where the cost of NumPy array allocation outweighs the benefits of vectorization. When vectorizing, use broadcasting and safe math (e.g. `np.fill_diagonal(dist_sq, np.inf)`) to handle self-interaction correctly.
