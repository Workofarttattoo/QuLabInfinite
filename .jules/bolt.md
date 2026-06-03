## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-23 - Dictionary Creation Overhead in Inner Loops
**Learning:** Creating a dictionary (e.g., `field_map`) inside a function called repeatedly in a tight loop (20,000+ times) can dominate execution time, even more than complex math like `np.linalg.norm`.
**Action:** Always verify if constant mappings are being reconstructed inside loops. Move them to class attributes or constants.
## 2025-02-14 - Vectorize Word2Vec Negative Sampling
**Learning:** In machine learning implementations like Word2Vec skip-gram, O(K) Python loops for negative sampling calculations create massive overhead during training epochs. Using pure Python loops over negative samples is an anti-pattern when NumPy's vectorized operations can handle dot products across the entire batch simultaneously.
**Action:** Vectorize the retrieval of negative contexts (e.g., `neg_vecs = context_embeddings[valid_neg_indices]`) and compute scores/gradients in a single step using `np.dot()` or matrix multiplication to eliminate iteration overhead and nearly halve the execution time.
