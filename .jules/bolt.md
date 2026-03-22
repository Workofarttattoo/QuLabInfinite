## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-22 - Random Forest Decision Tree Splitting Optimization
**Learning:** Finding the optimal split threshold by re-evaluating the variance of partitioned arrays via boolean masking (`np.var(y[mask])`) scales at O(N^2) because each threshold checks require a full O(N) scan. This causes decision trees to become extremely slow for larger datasets.
**Action:** When evaluating sequential splits on continuous features, sort the data first (O(N log N)) and use prefix sums (`np.cumsum`) to sequentially compute partitioned sums and sums-of-squares in O(N). This reduces the split evaluation complexity from O(N^2) to O(N log N) per feature.
