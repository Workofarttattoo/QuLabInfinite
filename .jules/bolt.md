## 2025-05-15 - Vectorizing O(n) Distance Calculations
**Learning:** In highly-iterated agent-based simulations (like the Oncology lab), calculating distance from each agent to a set of points (like blood vessels) using Python list comprehensions and sequential `np.linalg.norm` is a massive bottleneck.
**Action:** Always pre-convert lists of target points into a single NumPy array outside the loop and use vectorized math `np.linalg.norm(array - target, axis=1)` to eliminate Python loop overhead entirely.


## 2025-05-15 - Vectorizing O(n) Distance Calculations
**Learning:** In highly-iterated agent-based simulations (like the Oncology lab), calculating distance from each agent to a set of points (like blood vessels) using Python list comprehensions and sequential `np.linalg.norm` is a massive bottleneck.
**Action:** Always pre-convert lists of target points into a single NumPy array outside the loop and use vectorized math `np.linalg.norm(array - target, axis=1)` to eliminate Python loop overhead entirely.
