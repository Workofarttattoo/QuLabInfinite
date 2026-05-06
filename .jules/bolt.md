## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-23 - Dictionary Creation Overhead in Inner Loops
**Learning:** Creating a dictionary (e.g., `field_map`) inside a function called repeatedly in a tight loop (20,000+ times) can dominate execution time, even more than complex math like `np.linalg.norm`.
**Action:** Always verify if constant mappings are being reconstructed inside loops. Move them to class attributes or constants.

## 2024-05-18 - Ecology Lab Optimization
**Learning:** O(N^2) behavior in multi-entity distance evaluations using `scipy.spatial.distance_matrix` severely degrades spatial metric computations on large grid landscapes. When counting metrics on `labeled_patches`, individual iterations can be replaced by `np.bincount`, global structural modifications like `scipy.ndimage.binary_erosion`, and efficient nearest-neighbor searches via `scipy.spatial.cKDTree`, resulting in an immediate massive speedup (e.g. ~240x faster) without sacrificing accuracy.
**Action:** Always prefer `cKDTree.query_pairs` over full NxN distance matrices, and replace sequential patch iterations with whole-matrix masking and `np.bincount` aggregation for grid metric evaluations.
