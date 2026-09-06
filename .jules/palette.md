
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.

## 2024-05-19 - Composite UI Input Focus States
**Learning:** When adding focus states to composite UI input elements (like `.prompt-bar` containing a `contenteditable` span or input) in a vanilla JS frontend, styling the inner element directly can break the visual container boundaries.
**Action:** Use the `:focus-within` pseudo-class on the wrapper element to provide a unified accessibility focus ring, and ensure `outline: none` is set on the inner interactive element to prevent double focus rings.
