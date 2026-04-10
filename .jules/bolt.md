## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-22 - PK Model ODE Solver Optimization
**Learning:** Standard linear compartmental PK models (like the 2-compartment model in `pharmacology_lab.py`) have exact analytical mathematical solutions based on their characteristic equation roots. Using `scipy.integrate.odeint` for these models introduces unnecessary O(N) numerical integration overhead, whereas the analytical solution requires only O(1) vectorized operations (e.g., `np.exp`), offering massive speedups while maintaining absolute precision.
**Action:** Always identify if standard linear differential equations have well-known closed-form exact solutions before relying on generic numerical solvers like `odeint`.
