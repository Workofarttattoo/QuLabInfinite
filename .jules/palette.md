
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.

## 2025-03-05 - Cohesive Focus States for Composite Inputs
**Learning:** When styling composite input components (like a container with an inner `contentEditable` element), native focus rings on the inner element can look disconnected and unpolished. Applying `:focus-within` to the wrapper container and explicitly removing the `outline` on the inner element creates a single, cohesive, accessible focus state.
**Action:** When building composite inputs, apply interactive focus styles (using the brand's accent colors) using the `:focus-within` pseudo-class on the wrapper, and explicitly set `outline: none` on the inner focusable element.
