## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-23 - NumPy Scalar Assignment Overhead
**Learning:** Initializing an O(M*N) 2D NumPy array with `np.zeros()` and then performing individual scalar assignments in a nested double `for` loop (as seen in `edit_distance`) carries immense overhead compared to native Python operations, completely negating any benefits of using NumPy.
**Action:** When an algorithm strictly requires iterative element-by-element assignment where vectorization is impossible (e.g., dynamic programming like Levenshtein distance), use 1D pure Python lists and update them iteratively over memory-heavy and slow NumPy objects.
