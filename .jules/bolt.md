## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-23 - Dictionary Creation Overhead in Inner Loops
**Learning:** Creating a dictionary (e.g., `field_map`) inside a function called repeatedly in a tight loop (20,000+ times) can dominate execution time, even more than complex math like `np.linalg.norm`.
**Action:** Always verify if constant mappings are being reconstructed inside loops. Move them to class attributes or constants.

## 2024-05-18 - Replacing loops and distance matrices for fast spatial analysis
**Learning:** O(N^2) algorithms like `spatial.distance_matrix` over connected patch centroids (via `ndimage.label`) scale horribly and cause bottlenecks. Additionally, sequential loops per patch iterating with boolean masking (`== patch_id`) is slow.
**Action:** Replace `distance_matrix` with `spatial.cKDTree` (`tree.query_pairs(threshold)`) which is extremely fast. Replace patch iterative loops over masks with global vectorization using `np.bincount` to globally summarize labels (e.g. `np.bincount(labeled.ravel())`) and compute edge metrics by applying `ndimage.binary_erosion` to the whole grid at once.
