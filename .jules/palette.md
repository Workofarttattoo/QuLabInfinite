
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.

## 2025-05-04 - Synchronizing Active Classes with ARIA Attributes
**Learning:** When dynamically toggling custom `.active` classes on interactive elements like side navigation buttons to indicate selection, screen readers do not automatically announce this state change.
**Action:** Always programmatically synchronize the `aria-current` attribute (e.g., `aria-current="true"`) alongside visual `.active` class toggles to ensure accessible state communication for assistive technologies.
