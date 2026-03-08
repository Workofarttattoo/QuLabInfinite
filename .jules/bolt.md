## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-03-05 - Vectorize Neural Population Coding Loops
**Learning:** Python loops over neural populations for tuning curve projections (cosine tuning) and fisher information matrices create massive overhead when scaled (O(N) nested outer products).
**Action:** Always favor native `np.dot` over iterating arrays, and vectorize matrix accumulations via broadcasting (e.g. `derivatives.T @ derivatives` instead of looping `np.outer`) when applying population-wide math in simulation routines. This specific optimization yielded ~130x speedups in the `fisher_information` method.
