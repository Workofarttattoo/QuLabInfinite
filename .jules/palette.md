
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.

## 2025-05-07 - Screen Reader Accessibility for Dynamic Active States
**Learning:** When using JavaScript to toggle .active CSS classes on interactive selection elements, the active state is invisible to screen readers unless the aria-current attribute is explicitly updated.
**Action:** Always programmatically synchronize aria-current="true" (and remove it when inactive) on elements where .active classes are toggled dynamically.
