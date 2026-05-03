## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-23 - Dictionary Creation Overhead in Inner Loops
**Learning:** Creating a dictionary (e.g., `field_map`) inside a function called repeatedly in a tight loop (20,000+ times) can dominate execution time, even more than complex math like `np.linalg.norm`.
**Action:** Always verify if constant mappings are being reconstructed inside loops. Move them to class attributes or constants.

## 2025-05-23 - Global Vectorization of Spatial Metrics
**Learning:** In spatial ecology or image analysis workflows, iterating over labeled connected components (e.g., using a `for` loop to compute metrics for thousands of small patches) is a severe bottleneck due to repeated explicit NumPy array mask creations and `ndimage` filter calls (like `binary_erosion`).
**Action:** Replace iterative boolean mask evaluations with global O(1) mathematical ops whenever possible. Specifically, use `np.bincount` to compute sums by label in a single C-level pass, and apply spatial filters (like `binary_erosion`) once globally over the entire matrix rather than per-patch. This turned an 81-second loop into a 0.02-second operation.
