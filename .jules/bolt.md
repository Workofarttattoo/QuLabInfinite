## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-03-05 - [Optimize edit distance string assignment]
**Learning:** Initializing 2D NumPy arrays using `np.zeros` for string distance comparisons incurs a heavy performance penalty in standard Python loops due to extensive element-by-element scalar assignment overhead. Since `edit_distance` returns just an integer and operations only ever rely on row values of `prev_row` and `curr_row`, the O(M*N) memory layout can be completely bypassed for performance.
**Action:** When performing dynamic programming that does not require saving every state vector or array across iterations, downgrade 2D structures to 1D when the result relies only on immediate adjacent bounds. In pure python `for` loops with scalar updates, consider native Python list elements over NumPy matrices.
