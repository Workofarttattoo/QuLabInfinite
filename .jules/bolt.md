## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-03-05 - [bioinformatics_lab.py: String Slicing Bottleneck]
**Learning:** String slicing and appending inside tight nested loops in Python space carries a huge bytecode overhead, making O(N) operations behave effectively like O(N^2) in execution time compared to C-native equivalents.
**Action:** Replace manual character-by-character loops with Python's native `str.find` or regex patterns which are implemented in highly optimized C loops underneath.
