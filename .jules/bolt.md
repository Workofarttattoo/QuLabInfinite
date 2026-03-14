## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-03-05 - [Optimize Levenshtein Distance DP]
**Learning:** Using NumPy arrays with nested Python loops for dynamic programming can be significantly slower than using pure Python lists, especially when memory overhead can be optimized from O(N*M) to O(N). The hidden constant factor in NumPy scalar assignments from within Python loops causes severe slowdowns.

**Action:** Whenever implementing nested-loop DP algorithms in Python, default to 1D Python list arrays over 2D NumPy matrices unless the operations can be fully vectorized. Look for O(N*M) space algorithms that only depend on the previous row and reduce them to O(N) space to reduce array allocation and garbage collection overhead.
