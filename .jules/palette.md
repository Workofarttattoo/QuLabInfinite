
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.

## 2025-03-05 - Empty State for Filtering and Composite Input Focus
**Learning:** When building composite input components (like `.prompt-bar` containing `.prompt-input`), using `:focus-within` on the container allows cohesive focus styling without double-rings. Additionally, providing explicit empty states (e.g., "No labs found") for filtered lists significantly improves UX by confirming the system's status when results are zero.
**Action:** Use `:focus-within` on input wrappers combined with `outline: none` on the inputs themselves for unified focus states. Always ensure dynamic lists have an empty state fallback when filtering mechanisms return zero results.
