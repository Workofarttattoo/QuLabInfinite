## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-23 - Dictionary Creation Overhead in Inner Loops
**Learning:** Creating a dictionary (e.g., `field_map`) inside a function called repeatedly in a tight loop (20,000+ times) can dominate execution time, even more than complex math like `np.linalg.norm`.
**Action:** Always verify if constant mappings are being reconstructed inside loops. Move them to class attributes or constants.

## 2025-05-23 - Python Function Call Overhead in Tight Simulation Loops
**Learning:** In highly iterated loops (like ODE step functions or time series simulations such as those in `neurotransmitter.py`), calling standard library functions like `max(0.0, x)` or `min(1.0, max(0.0, x))` introduces significant Python call stack overhead. Benchmarks show that replacing these with inline ternary operators (`x if x > 0.0 else 0.0`) reduces iteration time by roughly 50-60%.
**Action:** Always replace 2-variable `min` and `max` calls with inline ternary conditional operators within deep simulation loops or numerical array iterators when performance is a bottleneck, but avoid complex nested ternaries (3+ variables) to preserve readability.
