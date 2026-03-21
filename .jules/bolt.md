## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-22 - Nested loop optimization with NumPy Broadcasting
**Learning:** In bioinformatics tree generation and other graph-related algorithms, nested loops calculating neighbor-joining Q matrices are often O(N^3). Replacing this with O(N^2) NumPy broadcasting `row_sums[:, np.newaxis] - col_sums[np.newaxis, :]` yields drastic speedups (e.g., ~750x faster on a 500x500 matrix).
**Action:** Always refactor explicit nested loops into vectorized broadcasting operations when calculating values dependent on rows and columns in large numerical matrices.
