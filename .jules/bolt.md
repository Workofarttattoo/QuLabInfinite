## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-23 - Dictionary Creation Overhead in Inner Loops
**Learning:** Creating a dictionary (e.g., `field_map`) inside a function called repeatedly in a tight loop (20,000+ times) can dominate execution time, even more than complex math like `np.linalg.norm`.
**Action:** Always verify if constant mappings are being reconstructed inside loops. Move them to class attributes or constants.

## 2026-05-20 - Vectorized Distances and Static Maps
**Learning:** In heavily iterated simulation loops across the qulab/labs/ codebase computing distances (e.g., cell to vessel distances in agent-based simulations), using Python list comprehensions and sequential np.linalg.norm operations incurs significant Python iteration overhead. Similarly, dynamic dictionary creation inside per-cell loop functions (like _get_field_value) compounds execution time.
**Action:** Pre-convert reference lists into a NumPy array outside the loop and use np.linalg.norm(array - target, axis=1) to vectorize the calculation. Define dictionaries as class-level constant attributes (e.g., _FIELD_MAP) for O(1) lookups.
