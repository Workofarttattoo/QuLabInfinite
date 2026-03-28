
## 2025-03-05 - NumPy Broadcasting in Particle Physics Jet Clustering
**Learning:** Explicit nested loops over elements in a list representing a distance matrix computation in particle jet clustering creates a major $O(N^3)$ bottleneck.
**Action:** Use $O(N^2)$ NumPy broadcasting and use `np.tril_indices` to properly mask the lower triangle/diagonal distance matrix without overwriting valid 0.0 values.
