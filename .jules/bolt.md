## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-23 - Dictionary Creation Overhead in Inner Loops
**Learning:** Creating a dictionary (e.g., `field_map`) inside a function called repeatedly in a tight loop (20,000+ times) can dominate execution time, even more than complex math like `np.linalg.norm`.
**Action:** Always verify if constant mappings are being reconstructed inside loops. Move them to class attributes or constants.

## 2024-04-11 - [Vectorize Metapopulation Stochastic Simulation]
**Learning:** In stochastic simulations like `ecology_lab.py`, calculating probabilities and state updates across multiple entities (e.g., habitat patches) using pure Python loops can be extremely slow due to interpreter overhead and per-iteration random number generation.
**Action:** Replace `for` loops that iterate over entities and update states conditionally with vectorized Numpy operations. Use boolean arrays (masks) to select subsets (e.g., `empty_mask = (prev == 0)`), generate arrays of random numbers at once (`np.random.random(patches)`), and apply bitwise logic (`new_colonized = empty_mask & (rand_col < prob_col)`). This leverages fast C-level execution for ~20x performance improvements.
