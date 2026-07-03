## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-23 - Dictionary Creation Overhead in Inner Loops
**Learning:** Creating a dictionary (e.g., `field_map`) inside a function called repeatedly in a tight loop (20,000+ times) can dominate execution time, even more than complex math like `np.linalg.norm`.
**Action:** Always verify if constant mappings are being reconstructed inside loops. Move them to class attributes or constants.
## 2025-05-23 - Massive Dynamic Object Allocation Overhead in NLP Layer
**Learning:** In the NLP `EmbeddingLayer`, dynamically generating large uniform arrays (e.g. `np.random.uniform(low=-1.0, high=1.0, size=(10000, 50))`) repeatedly on every `forward` pass creates a monumental memory and execution overhead, especially inside high-iteration loops like text embeddings. Generating these constants iteratively takes >95% of processing time.
**Action:** Always verify that constant matrices (like random embeddings or pre-computed lookup tables) are constructed exactly once in the class `__init__` rather than dynamically during `forward` or `update` passes.

## 2024-05-18 - Vectorize distances in nested simulation loops
**Learning:** Sequential `np.linalg.norm` and list comprehensions in Python loops are extremely slow for computing distance fields inside highly recurrent simulation update loops (like step functions computing cell-to-vessel distance). Converting static sets to numpy arrays outside the main loop and manually calculating squared distance components `np.sum((cell_pos - vessels_array)**2)` provides huge speedups compared to `np.linalg.norm`. Also, do not cache the array persistently in mutable environments where it might get stale—do it right before the loop within the epoch.
**Action:** When evaluating distance queries iteratively against a static set inside metaheuristics or simulation steps, always hoist array conversion out of the entity-level `for` loop. Inside the loop, perform per-entity vectorized math with sum-of-squares rather than invoking generic linear algebra methods for every item.
