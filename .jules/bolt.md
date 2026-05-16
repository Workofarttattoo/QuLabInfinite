## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-23 - Dictionary Creation Overhead in Inner Loops
**Learning:** Creating a dictionary (e.g., `field_map`) inside a function called repeatedly in a tight loop (20,000+ times) can dominate execution time, even more than complex math like `np.linalg.norm`.
**Action:** Always verify if constant mappings are being reconstructed inside loops. Move them to class attributes or constants.

## 2025-05-09 - Avoid convoluted inline logic

**Learning:** While replacing `min()` with an inline ternary operator `a if a < b and a < c else b if b < c else c` is technically faster by avoiding function call overhead, it becomes too convoluted for three variables and sacrifices code readability, constituting an anti-pattern. However, replacing `max()` for two variables with `a if a > b else b` is a clean, readable optimization that yields ~38% speedup in Python tight loops.

**Action:** Only apply inline ternary optimizations for `min()` or `max()` when dealing with two variables. Avoid using it for three or more variables to maintain codebase readability and respect micro-optimization constraints.
