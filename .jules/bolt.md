## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-23 - Dictionary Creation Overhead in Inner Loops
**Learning:** Creating a dictionary (e.g., `field_map`) inside a function called repeatedly in a tight loop (20,000+ times) can dominate execution time, even more than complex math like `np.linalg.norm`.
**Action:** Always verify if constant mappings are being reconstructed inside loops. Move them to class attributes or constants.
\n## 2026-04-24 - Metapopulation Spatial Vectorization\n**Learning:** Simulating spatial metapopulation dynamics (colonization and survival events) across multiple patches using sequential iteration over time and patches in Python loops can be a severe bottleneck.\n**Action:** Replace nested loops tracking entity states with vectorized matrix multiplication (e.g., `connectivity_matrix @ prev_occ`) and bulk random number generation (`np.random.random(size)`) to evaluate conditions using bitwise logic, resulting in massive speedups (~28x for 500 patches).
