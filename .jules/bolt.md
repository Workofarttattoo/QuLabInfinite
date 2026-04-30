## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-23 - Dictionary Creation Overhead in Inner Loops
**Learning:** Creating a dictionary (e.g., `field_map`) inside a function called repeatedly in a tight loop (20,000+ times) can dominate execution time, even more than complex math like `np.linalg.norm`.
**Action:** Always verify if constant mappings are being reconstructed inside loops. Move them to class attributes or constants.

## 2025-05-23 - String processing overhead in genomic analysis
**Learning:** For simple counting operations on DNA sequences, generator expressions combined with `.upper()` (e.g. `sum(1 for b in seq.upper() if b in 'GC')`) are significantly slower (~6x) than explicitly summing individual `str.count()` calls for each variant ('G', 'g', 'C', 'c'). The function call overhead and generator iteration dominate execution time compared to C-level string operations.
**Action:** When performing simple motif or character counting in long genomic sequences, prefer direct string methods (`count`, `find`) combined with explicit variant checks over list/generator comprehensions.
