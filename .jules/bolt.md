## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.
## 2025-03-05 - Vectorize bioinformatics neighbor joining Q-matrix calculation
**Learning:** Explicit O(N^3) nested loops containing `np.sum` array aggregations can become an extreme bottleneck, specifically within the Neighbor-Joining (`_neighbor_joining`) tree algorithm. NumPy array broadcasting operations (e.g., `row_sums[:, np.newaxis] - row_sums[np.newaxis, :]`) significantly improve iteration speed by collapsing Python loops into vectorized C operations.
**Action:** Always favor calculating properties via dimension broadcasting operations natively in NumPy when manipulating 2D arrays (like Q-matrices in Bioinformatics algorithms) instead of scalar modifications in loop nests.
