## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-22 - Network Motif Optimization
**Learning:** $O(N^3)$ nested loops used for finding path-based network motifs (like feed-forward and feedback loops) in an adjacency matrix can be perfectly represented by $O(N^\omega)$ matrix multiplication. Specifically, `sum((A @ A) * A)` calculates feed-forward loops, and `trace(A @ A @ A)` calculates feedback loops.
**Action:** When searching for graph motifs or multi-hop paths, always consider replacing nested loops with powers of the adjacency matrix (`A @ A`, `A @ A @ A`).
