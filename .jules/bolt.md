## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-22 - Vectorized Dispersion and Amplification
**Learning:** Found O(N) Python loops over `frequency` and `periods` arrays in `seismology_lab.py` (`site_amplification` and `surface_wave_dispersion`) computing conditional mathematical operations.
**Action:** Replaced element-by-element assignment and loops with O(1) vectorized `numpy` array operations and boolean masking (`np.where` logic via boolean slicing), resulting in ~20x speedup for typical array sizes while retaining exact mathematical equivalence.
