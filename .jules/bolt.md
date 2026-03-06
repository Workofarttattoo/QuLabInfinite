## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.

## 2025-03-05 - Vectorization of Neural Network Simulations
**Learning:** In highly mathematical Python simulations (like Spiking Neural Networks in `neural_networks_lab.py`), the internal loops inside time-step integrators (`simulate`) computing equations per neuron generate massive overhead when the number of neurons is large (1,000-10,000+).
**Action:** Always favor bulk NumPy array operations over iterating `for neuron in self.neurons` within the simulation loop. Vectorizing state variables (e.g., fetching `V`, computing `dV`, handling bounds, tracking spikes) reduces execution times drastically (e.g., from ~1.88s to ~0.60s for 2,000 neurons).
