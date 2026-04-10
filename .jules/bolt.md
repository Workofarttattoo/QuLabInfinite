## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2026-03-16 - Hoisting loop-invariant array calculations
**Learning:** In iterative PDE solvers like Navier-Stokes projection methods, source terms based on the previous time step's velocities are constant during the internal pressure Poisson iterations. Computing them inside the inner loop performs redundant O(N^2) array operations.
**Action:** Always pre-compute loop-invariant array operations (like the pressure source term) outside of inner convergence loops.
