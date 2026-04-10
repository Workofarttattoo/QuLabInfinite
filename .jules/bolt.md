## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.
## 2025-03-05 - Avoid Full Array Copies in Iterative Finite Differences
**Learning:** In nested loops computing numerical derivatives (like gradients or Hessians), full array copies (`x.copy()`) inside the inner loops scale horribly (e.g. O(N^3) memory allocation overhead for Hessians). Even if true NumPy vectorization is impossible due to the black-box scalar nature of the function `f(x)`, massive performance gains can be achieved through simple in-place scalar modification of the array.
**Action:** When computing multi-variable finite differences element-by-element, modify the specific coordinate `x[i] += epsilon`, call `f(x)`, and immediately restore `x[i] -= epsilon` instead of creating `N` copies of the array.
## 2025-03-05 - Vectorize 2D Convolution
**Learning:** Explicit Python nested loops over NumPy arrays for operations like 2D convolution introduce massive overhead, especially on larger images.
**Action:** Always prefer vectorized operations using `np.lib.stride_tricks.sliding_window_view` paired with `np.tensordot` (using exact axes mapping like `axes=((2, 3), (0, 1))`) for sliding-window operations over images to keep execution entirely in C, avoiding Python's iteration overhead.
