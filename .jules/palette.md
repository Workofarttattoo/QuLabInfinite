
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.

## 2025-03-06 - Interactive Focus Styles for Composite Inputs
**Learning:** Composite input components (like a container with an input and button) can look broken if only the inner input receives focus styles.
**Action:** Apply interactive focus styles using the `:focus-within` pseudo-class on the wrapper container and explicitly set `outline: none` on the inner input element to ensure a cohesive, accessible focus state without double rings.
