
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.

## 2024-05-30 - Accessible Focus States for Composite Inputs
**Learning:** When creating composite input components (like a div containing a contenteditable span and a button), the default focus outline on the inner editable element can look disjointed or incomplete. Using `:focus-within` on the parent container and removing the outline on the inner child provides a much more cohesive and accessible focus ring.
**Action:** Apply `:focus-within` on the wrapper and `outline: none` on the inner interactive elements when building custom composite text inputs to ensure a unified focus state.
