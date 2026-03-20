## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-22 - Quantum Gate Vectorization
**Learning:** Dense matrix allocations (e.g. `np.eye(2**N)`) and string-based index parsing in quantum simulation methods scale terribly in both execution time and memory (O(4^N)). By vectorizing qubit manipulation using NumPy bitwise operations (`(idx >> bit) & 1`) and boolean masking on the O(2^N) state vector directly, execution times for large qubit operations (like QFT) dropped drastically (e.g. ~5s to ~1.1s for 12 qubits) without changing the mathematical outcome.
**Action:** When implementing quantum logic or large discrete state-space simulations, always replace explicit dense matrix construction and iterative string parsing with vectorized bitwise arithmetic and array masking on the state vector itself.
