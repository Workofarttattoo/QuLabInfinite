## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-22 - Analytical vs Numerical Integration
**Learning:** Found a major performance bottleneck where O(N) numerical integration (`scipy.integrate.quad`) was being used for basic polynomial sequences in thermodynamic property loops (enthalpy/entropy). Replacing this with O(1) exact analytical mathematical integration provides immense speedups (~30x faster) without sacrificing accuracy or modifying package dependencies.
**Action:** Always verify if numerical PDE/ODE solvers or quadratures (`quad`, `odeint`) can be replaced by exact analytical equivalents when the underlying mathematical form is constrained (like polynomials or two-compartment PK models).
