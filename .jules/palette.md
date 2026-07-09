
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.

## 2025-04-16 - Synchronizing ARIA States with Visual Classes
**Learning:** When using custom visual classes like `.active` to indicate selection states on interactive elements (e.g. custom button lists), screen readers are unaware of the selection unless explicitly told. `aria-current="true"` is essential for communicating the currently active item within a set.
**Action:** When programmatically toggling active visual classes using JavaScript, always ensure `aria-current="true"` is synchronized with the active item and removed from inactive items.
