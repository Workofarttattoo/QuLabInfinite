## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-23 - Dictionary Creation Overhead in Inner Loops
**Learning:** Creating a dictionary (e.g., `field_map`) inside a function called repeatedly in a tight loop (20,000+ times) can dominate execution time, even more than complex math like `np.linalg.norm`.
**Action:** Always verify if constant mappings are being reconstructed inside loops. Move them to class attributes or constants.
## 2026-05-28 - Optimize NumPy simulation loops
**Learning:** The codebase contains multiple identical implementations of mathematical simulations (like `sequence_dna`) and calling `np.random.randint` within a Python loop creates massive overhead, especially when duplicate loops iterate over the same sets. Replacing per-element rolls with vectorized NumPy calls on flattened arrays and combining duplicate iteration loops drastically improves simulation performance.
**Action:** Always vectorize the generation of random indices using NumPy arrays (e.g., `np.random.randint(0, bound, size=num_reads)`) outside loops, eliminate duplicate iterations over identically-sized collections, and ensure optimizations are replicated across all identical implementations in the codebase.
