## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-23 - Dictionary Creation Overhead in Inner Loops
**Learning:** Creating a dictionary (e.g., `field_map`) inside a function called repeatedly in a tight loop (20,000+ times) can dominate execution time, even more than complex math like `np.linalg.norm`.
**Action:** Always verify if constant mappings are being reconstructed inside loops. Move them to class attributes or constants.
## 2024-04-20 - Global Vectorization for Connected Component Spatial Metrics
**Learning:** Sequential per-patch property calculations (such as patch area, perimeter, and centroids via `np.where`) and $O(N^2)$ cross-patch distance calculations (`scipy.spatial.distance_matrix`) severely bottleneck processing of spatial grid maps (like habitat layouts) when the number of connected components is high. Iterating through patches explicitly scales terribly.
**Action:** Replace $O(N)$ patch-iterative loops with $O(1)$ global map operations. Use `np.bincount` to instantly find all patch sizes, apply global morphological operators like `ndimage.binary_erosion(mask)` combined with `bincount` on edge pixels for perimeters, use `ndimage.center_of_mass(mask, labels, index)` for global centroids, and deploy `scipy.spatial.cKDTree` for thresholded spatial distance queries instead of dense distance matrices.
