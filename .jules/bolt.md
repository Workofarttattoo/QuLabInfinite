## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-23 - Dictionary Creation Overhead in Inner Loops
**Learning:** Creating a dictionary (e.g., `field_map`) inside a function called repeatedly in a tight loop (20,000+ times) can dominate execution time, even more than complex math like `np.linalg.norm`.
**Action:** Always verify if constant mappings are being reconstructed inside loops. Move them to class attributes or constants.
## 2026-04-22 - Optimize ecology_lab habitat fragmentation spatial connectivity calculation
**Learning:** When analyzing distances between thousands of spatial items in Python (like patches in `ecology_lab.py`), computing the entire distance matrix using `scipy.spatial.distance_matrix(centroids, centroids)` results in a huge O(N^2) bottleneck. Additionally, calculating patch centroids individually by looping through labeled sections using `np.where` takes significant time. Replacing these with `scipy.ndimage.center_of_mass` globally and `scipy.spatial.cKDTree` for proximity queries provides over 200x performance increase.
**Action:** Always use vectorized `ndimage.center_of_mass` to compute centroids for labeled spatial regions. For threshold-based pair matching, switch from `distance_matrix` (O(N^2)) to `cKDTree.query_pairs` (O(N log N)) when large arrays are involved.
