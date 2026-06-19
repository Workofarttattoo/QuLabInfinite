
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.

## 2025-03-05 - Styling Focus States for Composite Inputs
**Learning:** When styling a composite input component (like a `.prompt-bar` container wrapping a `.prompt-input` contenteditable element), users expect the entire container to visually indicate focus. Relying solely on the inner element's outline creates disconnected focus rings.
**Action:** Use the CSS `:focus-within` pseudo-class on the parent wrapper container to trigger focus styles, and explicitly set `outline: none` on the child input element to ensure a cohesive, accessible focus state without double rings.
