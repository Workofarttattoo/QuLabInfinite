
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.

## 2025-05-09 - Syncing ARIA with Dynamic Active States
**Learning:** For lists of interactive elements acting like tabs or active selections (e.g., lab selection buttons), dynamically toggling visual `.active` classes is not enough for screen readers. Screen reader users rely on attributes like `aria-current="true"` to understand which item is currently selected out of the list.
**Action:** When programmatically toggling an active CSS class on interactive elements via JavaScript, always synchronize it with the corresponding `aria-current` attribute (or `aria-selected` for ARIA tablists) so that accessibility state exactly matches the visual state.
