## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-22 - Vectorize Harmonic Iteration
**Learning:** Signal processing code computing frequency metrics like Total Harmonic Distortion (THD) often iterates over harmonics to find frequency peaks. Since `scipy.fft.rfftfreq` produces uniformly spaced arrays, using O(1) mathematical index calculation (`freq / df`) is much faster than an O(N) search (`np.argmin(np.abs(freqs - target))`). Combined with numpy vectorization, this replaced an O(N * M) nested loop with an O(1) slice and sum operation.
**Action:** When working with FFT frequencies, compute array indices directly from frequencies `index = round(freq / df)` instead of searching the frequency array.
