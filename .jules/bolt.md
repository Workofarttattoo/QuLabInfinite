## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-05-22 - Vectorize synaptic current calculation
**Learning:** `_compute_synaptic_current` in `neural_networks_lab.py` previously contained a loop over all neurons (e.g. 1000) for NMDA calculation. This was very slow. Vectorizing the calculation by extracting an array of potentials `V = np.array([neuron.V for neuron in self.neurons])` and replacing `self.W[:, i] @ self.synaptic_traces[:, 1]` with `(self.W.T @ self.synaptic_traces[:, 1])` without indexing results in a ~6-10x speedup depending on neuron counts.
**Action:** When calculating synaptic currents or state updates across a large group of neurons, always extract states into a NumPy array and use vectorized operations and array broadcasting instead of iterating explicitly in Python.
