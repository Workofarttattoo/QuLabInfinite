## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-22 - Graph Motif Optimization via Adjacency Matrix Multiplication
**Learning:** Computing paths (feed-forward, feedback loops) in large graphs using deeply nested O(N³) Python `for` loops is excessively slow. The adjacency matrix's exponentiation properties (`A @ A`) inherently count paths of length 2 (and 3 with trace(A^3)), pushing the workload to BLAS-optimized C routines underneath NumPy, allowing a drop from ~2.5 seconds to ~0.02 seconds for a 200-node graph.
**Action:** Always map generic path-finding or motif counting algorithms into vectorized adjacency matrix operations rather than Python-level iterations over nodes, leveraging the definition that A^k represents the number of walks of length k between nodes.
