## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2023-10-27 - Vectorized Sequence Mutation Operations
**Learning:** Simulating DNA sequencing reads with per-base random error insertion using `np.random.random()` inside a nested Python loop is severely slow.
**Action:** Always pre-calculate starting points in bulk with `np.random.randint(..., size=N)` and use boolean mask vectorization (`error_mask = np.random.random(read_len) < error_rate`) with `np.where` for per-character mutation steps.
