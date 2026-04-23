## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-23 - Dictionary Creation Overhead in Inner Loops
**Learning:** Creating a dictionary (e.g., `field_map`) inside a function called repeatedly in a tight loop (20,000+ times) can dominate execution time, even more than complex math like `np.linalg.norm`.
**Action:** Always verify if constant mappings are being reconstructed inside loops. Move them to class attributes or constants.

## 2025-05-23 - Distance Matrix Overhead in Spatial Queries
**Learning:** Using `scipy.spatial.distance_matrix` to compute distances between thousands of centroids creates an O(N^2) memory and performance bottleneck (e.g., in ecology connectivity analysis). Extracting centroids with a loop over `np.where` for each patch is also very slow.
**Action:** Replace `distance_matrix(x, x) < threshold` with `scipy.spatial.cKDTree(x).query_pairs(threshold)`. Use `scipy.ndimage.center_of_mass(matrix, labels, index=np.arange(1, n_patches+1))` instead of iterating with `np.where` to find centroids.
