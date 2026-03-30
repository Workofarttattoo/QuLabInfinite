## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.
## 2025-03-05 - Avoid Full Array Copies in Iterative Finite Differences
**Learning:** In nested loops computing numerical derivatives (like gradients or Hessians), full array copies (`x.copy()`) inside the inner loops scale horribly (e.g. O(N^3) memory allocation overhead for Hessians). Even if true NumPy vectorization is impossible due to the black-box scalar nature of the function `f(x)`, massive performance gains can be achieved through simple in-place scalar modification of the array.
**Action:** When computing multi-variable finite differences element-by-element, modify the specific coordinate `x[i] += epsilon`, call `f(x)`, and immediately restore `x[i] -= epsilon` instead of creating `N` copies of the array.

## 2025-03-05 - Vectorizing Image Operations with stride_tricks
**Learning:** Pure Python nested loops over NumPy arrays (e.g., sliding window operations in 2D convolutions or pooling) create hidden O(N^3) memory/execution overhead and are prohibitively slow.
**Action:** Always replace nested Python sliding-window image loops with `np.lib.stride_tricks.as_strided` to create a virtual window array, then use vectorized operations like `np.tensordot`, `.max()`, or `.mean()` on the strided view. This keeps computations entirely in C and can yield ~100x speedups.
