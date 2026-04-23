## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-23 - Dictionary Creation Overhead in Inner Loops
**Learning:** Creating a dictionary (e.g., `field_map`) inside a function called repeatedly in a tight loop (20,000+ times) can dominate execution time, even more than complex math like `np.linalg.norm`.
**Action:** Always verify if constant mappings are being reconstructed inside loops. Move them to class attributes or constants.

## $(date +%Y-%m-%d) - [O(n^2) distance matrix bottleneck resolved in spatial analysis]
**Learning:** Calculating metrics over thousands of spatial patches iteratively using O(n^2) operations like `spatial.distance_matrix` causes immense performance bottlenecks in Python. Using `ndimage.center_of_mass` for vectorized centroid generation across all labels alongside `spatial.cKDTree` for thresholded pair queries transforms heavy computational loops into fast queries. Additionally, extracting metrics like patch size and perimeters from label matrices can be done globally using `np.bincount` instead of boolean masking in a loop.
**Action:** When calculating metrics for connected components or distances across many patches, always replace Python loops (`for patch_id...`) with global boolean array masks and fast matrix algorithms like `np.bincount`, `ndimage.binary_erosion`, `ndimage.center_of_mass`, and `spatial.cKDTree`. Ensure native Python type casting for numerical values computed this way if they are to be JSON-serialized.
