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
## 2025-02-27 - Vectorizing Cell-to-Entity Distance Calculations
**Learning:** In simulation loops processing thousands of entities iteratively (like cells evaluating distance to blood vessels), using list comprehensions and computing `np.linalg.norm()` separately per element incurs significant Python loop and memory allocation overhead.
**Action:** Always pre-convert static coordinate lists into a single NumPy array and evaluate distances using completely vectorized array operations (e.g., `np.sum((target - points_array)**2, axis=1)`) to dramatically reduce computation time per cycle.
## 2025-02-27 - Vectorization Array Caching Mutability
**Learning:** When vectorizing entity relationships with numpy arrays (e.g., cell to vessel distances), caching the array instance variable permanently (e.g., `self._vessels_array = ...`) is an anti-pattern if the microenvironment is mutable. It creates bugs where simulation entities reference stale environmental features.
**Action:** Always place the pre-conversion of static coordinates into a fresh numpy array at the start of the simulation tick (outside the entity loop), rather than persistently on the instance. This guarantees the array reflects the current timestep while avoiding the overhead of re-evaluating it for every single entity.
