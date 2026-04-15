## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-23 - Dictionary Creation Overhead in Inner Loops
**Learning:** Creating a dictionary (e.g., `field_map`) inside a function called repeatedly in a tight loop (20,000+ times) can dominate execution time, even more than complex math like `np.linalg.norm`.
**Action:** Always verify if constant mappings are being reconstructed inside loops. Move them to class attributes or constants.

## 2025-05-23 - Vectorizing Graph and Edge Calculations in Ecology Lab
**Learning:** Finding connected patches in a large `landscape_matrix` and analyzing their centers of mass and edges was incredibly slow due to iterating through `range(n_patches)` and using O(N^2) algorithms like `spatial.distance_matrix(centroids, centroids)`. A 1000x1000 landscape with ~100k patches previously consumed 83 GB RAM and failed. Using vectorized metrics with `scipy.ndimage.center_of_mass` and replacing `distance_matrix` with `scipy.spatial.cKDTree(centroids).query_pairs(distance_threshold)` reduces memory to O(N) and time to less than a second. Similarly, `np.bincount` calculates patch sizes globally instead of using boolean indexing per patch.
**Action:** When working with connected component properties (like patch sizes, perimeters, or centers of mass) in `scipy.ndimage`, avoid `for patch_id in range(1, n_patches + 1):`. Instead, use fully vectorized alternatives like `np.bincount` and global mask arrays, and prefer `cKDTree` for distance proximity tasks over `distance_matrix`.
