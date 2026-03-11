## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2024-03-05 - Replacing ODE solvers with analytical solutions for compartmental models
**Learning:** Found that numerical integration (`scipy.integrate.odeint`) for pharmacokinetic (PK) multi-compartment models introduced massive overhead. Standard linear compartmental models (like the 2-compartment IV bolus model) have exact analytical mathematical solutions consisting of a sum of exponential terms.
**Action:** Replaced O(N) numerical ODE integration with O(1) vectorized analytical mathematical solution in `pharmacology_lab.py`. This resulted in a ~15x speedup (from 0.61ms to 0.04ms) while avoiding numerical integration margins of error entirely. This pattern should be applied wherever linear differential equations with known analytical solutions are used instead of running costly numerical solvers.
