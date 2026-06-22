
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.

## 2025-05-24 - Composite Input Focus Styling
**Learning:** When styling composite input components in the frontend UI (e.g., `.prompt-bar`), interactive focus styles should be applied using the `:focus-within` pseudo-class on the wrapper container. The inner input element itself should explicitly set `outline: none` to ensure a cohesive, accessible focus state without double rings.
**Action:** Use `:focus-within` on the container for complex inputs, rather than relying on individual inner elements to handle the visible focus.
