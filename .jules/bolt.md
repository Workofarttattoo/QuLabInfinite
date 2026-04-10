## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-22 - Random Forest Split Optimization
**Learning:** Finding the best split threshold for random forest features using boolean masking on arrays runs in O(N^2) time, which causes significant performance bottlenecks for larger samples. By pre-sorting feature values and utilizing cumulative sums (`np.cumsum`), variance calculations across all possible splits can be computed mathematically in O(N) time (giving a total complexity of O(N log N) for the sorting), resulting in orders of magnitude speedups (~60x).
**Action:** When calculating aggregations over sequential partitions (like variance for decision tree splits), avoid using boolean masks in a loop and instead use sorted arrays and prefix sum mathematical optimizations.
