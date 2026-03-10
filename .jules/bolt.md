## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-03-10 - Vectorize UPGMA distance calculations
**Learning:** During UPGMA clustering for phylogenetic trees or sequence alignments, calculating the average distance between two clusters using nested Python list comprehensions (`[distances[a, b] for a in clusters[i] for b in clusters[j]]`) creates an $O(N^3)$ bottleneck.
**Action:** Use NumPy's advanced indexing with `np.ix_` (`distances[np.ix_(clusters[i], clusters[j])]`) to extract the submatrix and compute the mean directly in C, eliminating slow Python loops.
