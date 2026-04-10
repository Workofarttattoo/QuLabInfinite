## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2026-03-22 - Replacing O(N) array search with O(log N) `np.searchsorted`
**Learning:** In calculations iterating over uniformly spaced arrays (like frequencies returned by `rfftfreq`), using `np.argmin(np.abs(array - val))` creates an O(N) bottleneck, especially when computed inside a loop.
**Action:** Replace `np.argmin` search with O(log N) `np.searchsorted` for any array guaranteed to be sorted, dropping time complexity dramatically (7x speedup locally).
