
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.

## 2025-05-05 - Synchronize aria-current with active states
**Learning:** When dynamically toggling visual active states (e.g., using a `.active` CSS class) on interactive elements like tab buttons or selection lists, screen readers are often unaware of the change.
**Action:** Programmatically synchronize the `aria-current="true"` attribute alongside visual state changes (like adding/removing an `.active` class) to ensure screen reader accessibility.
