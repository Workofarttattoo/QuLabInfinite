## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-23 - Dictionary Creation Overhead in Inner Loops
**Learning:** Creating a dictionary (e.g., `field_map`) inside a function called repeatedly in a tight loop (20,000+ times) can dominate execution time, even more than complex math like `np.linalg.norm`.
**Action:** Always verify if constant mappings are being reconstructed inside loops. Move them to class attributes or constants.

## 2025-05-23 - Vectorized Ecology Spatial Analysis
**Learning:** O(N^2) spatial distance checks like `spatial.distance_matrix(centroids, centroids) < threshold` bottleneck severely in ecology labs for a large number of components. `scipy.spatial.cKDTree(centroids).query_pairs(threshold)` performs the exact equivalent at a fraction of the time, leading to ~1000x speedup combined with global vectorized `ndimage.center_of_mass`, `ndimage.binary_erosion` and `np.bincount` replacing loops over individual labels.
**Action:** Identify multi-entity loops that rely on spatial queries and replace distance matrices with KD Trees, avoiding element-wise checks entirely. Use `bincount` to vectorize per-patch logic mapped to array inputs over the whole grid.
