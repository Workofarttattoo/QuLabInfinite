## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-23 - Dictionary Creation Overhead in Inner Loops
**Learning:** Creating a dictionary (e.g., `field_map`) inside a function called repeatedly in a tight loop (20,000+ times) can dominate execution time, even more than complex math like `np.linalg.norm`.
**Action:** Always verify if constant mappings are being reconstructed inside loops. Move them to class attributes or constants.
## 2025-05-23 - Python Loop Overhead in Distance Calculation
**Learning:** In tumor simulations with thousands of cells and vessels, a Python loop using `np.linalg.norm` for each cell creates massive overhead. By extracting cell positions and vessel locations into NumPy arrays and calculating distances using broadcasting (`np.sqrt(np.sum((positions[:, np.newaxis, :] - vessels[np.newaxis, :, :])**2, axis=2))`), simulation time drops drastically (e.g., 50x speedup for 5000 cells).
**Action:** Always vectorize agent-to-environment distance calculations across all agents simultaneously instead of looping over agents in Python.
