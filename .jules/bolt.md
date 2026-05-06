## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-23 - Dictionary Creation Overhead in Inner Loops
**Learning:** Creating a dictionary (e.g., `field_map`) inside a function called repeatedly in a tight loop (20,000+ times) can dominate execution time, even more than complex math like `np.linalg.norm`.
**Action:** Always verify if constant mappings are being reconstructed inside loops. Move them to class attributes or constants.

## 2024-05-18 - Avoid distance_matrix OOM errors with cKDTree
**Learning:** Using `scipy.spatial.distance_matrix(x, x)` to evaluate pairwise connectivity inside Python hot loops introduces O(N^2) memory and time overhead, often leading to OOM (e.g., ArrayMemoryError) or blocking on large inputs when N hits tens of thousands of patches (like `ecology_lab.py`). Additionally, nested `np.where` loops for finding per-patch properties in `ndimage.label` matrices is excessively slow.
**Action:** Replaced O(N^2) distance calculations by instantiating `scipy.spatial.cKDTree(centroids)` and calling `.query_pairs(threshold)`. Also replaced per-patch spatial iteration by using vectorized tools like `np.bincount`, global boolean masking (`ndimage.binary_erosion`), and `ndimage.center_of_mass(matrix, labels, index)`. When migrating `query_pairs` from a strict less-than `distances < threshold` calculation, a small offset (e.g., `threshold - 1e-9`) was included to preserve identical logic.
