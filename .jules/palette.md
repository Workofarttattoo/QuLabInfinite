
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.
## 2024-06-17 - Accessible Focus States on Composite Input Components
**Learning:** When dealing with composite inputs like a contentEditable element inside a stylized wrapper `.prompt-bar`, native focus states on the contentEditable inner element often appear disconnected from the wrapper. Keyboard users need clear visual feedback.
**Action:** Use `:focus-within` on the parent wrapper container to display an accessible, cohesive focus ring, and explicitly apply `outline: none` to the inner input element to prevent confusing double focus rings. Always ensure input fields like `.search-input` explicitly define `:focus-visible` to support keyboard navigation.
