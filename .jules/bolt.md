## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-23 - Dictionary Creation Overhead in Inner Loops
**Learning:** Creating a dictionary (e.g., `field_map`) inside a function called repeatedly in a tight loop (20,000+ times) can dominate execution time, even more than complex math like `np.linalg.norm`.
**Action:** Always verify if constant mappings are being reconstructed inside loops. Move them to class attributes or constants.

## 2023-10-30 - [Optimize Habitat Fragmentation Analysis in Ecology Lab]
**Learning:** For extensive N-body-like distance calculations in spatial modeling, $O(N^2)$ computations with `scipy.spatial.distance_matrix` become a severe bottleneck (e.g., taking over 218s for 26,000 components). Replacing it with `scipy.spatial.cKDTree` significantly reduces query times to a fraction of a second. Additionally, instead of iterating over individual labeled patches using python loops and calculating properties like size and edge manually, global vectorization approaches such as `np.bincount` combined with morphological operations (`ndimage.binary_erosion`) can yield an overall >1000x speedup.
**Action:** When working on grid or spatial operations involving distance evaluations over thousands of entities, always opt for cKDTree instead of computing explicit full distance matrices, and default to global vectorization `np.bincount` instead of localized iteration.
