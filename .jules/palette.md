
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.

## 2025-03-05 - Focus styles for composite inputs
**Learning:** When creating composite input components (like a div containing a contenteditable span and a button), the standard `:focus` style on the inner element can look disconnected or cause double-rings. By applying `:focus-within` to the wrapper container and `outline: none` to the inner input, we create a cohesive, accessible focus state for the entire interactive group.
**Action:** When styling complex input components, apply interactive focus styling to the wrapper container using `:focus-within` and suppress inner element outlines to provide a unified visual focus state.
