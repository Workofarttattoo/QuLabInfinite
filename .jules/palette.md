
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.

## 2025-04-13 - Aria-Current for Dynamic Active States
**Learning:** When managing list items or buttons with dynamic `.active` classes, purely visual indicators are insufficient for screen readers. The `aria-current="true"` attribute must be synchronized programmatically along with the CSS class to ensure accessibility context is communicated properly, especially in dynamically generated lists.
**Action:** Always ensure that JS toggling logic for visual 'active' states also explicitly updates `aria-current="true"` and removes it from inactive sibling elements.
