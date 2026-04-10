## 2025-05-22 - Finite Difference Optimization
**Learning:** Implicit finite difference solvers often re-solve the same linear system Ax=b every step. Pre-factorizing A using `scipy.sparse.linalg.factorized` when the time step `dt` is constant yields massive speedups (~90x).
**Action:** Always check for repeated linear system solves in physics loops and cache the factorization if the matrix is constant.

## 2025-05-22 - Test Gaps and Bugs
**Learning:** Found a critical bug (immutable sparse matrix assignment) in `thermodynamics_grid.py` only because I wrote a benchmark script. The existing test suite did not cover this module.
**Action:** When optimizing, if no specific test exists for the target module, write a reproduction/benchmark script first to verify it works at all.
## 2025-03-05 - Avoid Full Array Copies in Iterative Finite Differences
**Learning:** In nested loops computing numerical derivatives (like gradients or Hessians), full array copies (`x.copy()`) inside the inner loops scale horribly (e.g. O(N^3) memory allocation overhead for Hessians). Even if true NumPy vectorization is impossible due to the black-box scalar nature of the function `f(x)`, massive performance gains can be achieved through simple in-place scalar modification of the array.
**Action:** When computing multi-variable finite differences element-by-element, modify the specific coordinate `x[i] += epsilon`, call `f(x)`, and immediately restore `x[i] -= epsilon` instead of creating `N` copies of the array.

## 2024-04-06 - Optimize materials API endpoints by offloading synchronous SQLite queries to threadpool
**Learning:** Returning large dictionaries from FastAPI endpoints defined with `def` causes `jsonable_encoder` to block the main event loop thread during serialization. Using `JSONResponse` directly avoids this overhead and prevents the event loop from stalling. Additionally, synchronous I/O such as standard SQLite queries should be kept out of `async def` endpoints, as they will block the entire server; instead, they should be defined as `def` so FastAPI automatically runs them in a separate thread pool.
**Action:** Changed synchronous SQLite database endpoints in `materials_api.py` (`get_stats`, `search`, `get_material`, `get_categories`, `recommend`) from `async def` to `def`, and explicitly returned `JSONResponse` for large JSON payloads, allowing background health checks and parallel asynchronous tasks to execute smoothly.
