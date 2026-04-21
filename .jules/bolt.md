## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-23 - Dictionary Creation Overhead in Inner Loops
**Learning:** Creating a dictionary (e.g., `field_map`) inside a function called repeatedly in a tight loop (20,000+ times) can dominate execution time, even more than complex math like `np.linalg.norm`.
**Action:** Always verify if constant mappings are being reconstructed inside loops. Move them to class attributes or constants.
## 2024-03-24 - Ecology Lab Vectorization Bottleneck
**Learning:** Found an O(N^2) bottleneck when checking habitat connectivity between isolated ecosystem patches (`distance_matrix(centroids, centroids)`). As N (number of patches) scales to thousands in large models, this grid-locks execution taking several seconds in Python loops.
**Action:** Always replace per-patch looping with global `np.bincount` weighting masks for size/perimeter metrics, and swap O(N^2) naive coordinate matrices with `spatial.cKDTree` for distance proximity checks `tree.query_pairs(threshold)`. This reduced calculation time from ~3.3s to ~0.013s (>200x speedup). Make sure to cast back to Python native floats before JSON returning to prevent pipeline data-type errors.
