## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-03-02 - SciPy ANOVA Overhead in Tight Loops
**Learning:** `scipy.stats.f_oneway` introduces significant validation and function-call overhead when executed inside tight iterative loops (like QTL mapping or permutation testing).
**Action:** Replace `f_oneway` with direct, vectorized NumPy calculations for Sum of Squares (Between and Within) and use `scipy.stats.f.sf` to calculate the p-value. This can reduce runtime by roughly ~75% while maintaining mathematical correctness.
