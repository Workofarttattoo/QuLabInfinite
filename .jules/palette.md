
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.
## 2024-07-07 - Add Focus Visible State to Composite Input Component
**Learning:** When styling composite input components, like the `.prompt-bar`, the `:focus-within` pseudo-class allows applying focus styles on the wrapper element containing an interactive input. This creates a cohesive, single focus outline. However, to prevent double rings from the inner element's native focus, `outline: none` must be applied explicitly to the `.prompt-input`.
**Action:** Apply `:focus-within` to wrappers for cohesive focus states, and use `outline: none` on inner interactables to avoid duplicate focus indicators.
