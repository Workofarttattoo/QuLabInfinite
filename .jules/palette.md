
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.

## 2024-09-09 - Focus-within on composite UI elements
**Learning:** When using `contenteditable` span inside a complex wrapper (`.prompt-bar`), moving the focus visual styling from the inner input to the wrapper using `:focus-within` significantly improves UI coherence and makes the whole component act as the interactive area.
**Action:** Always apply focus visual states to the parent container (`:focus-within`) for composite input components instead of just the inner input element.
