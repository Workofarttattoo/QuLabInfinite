
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.

## 2024-05-14 - Communicating Active States and Form Accessibility
**Learning:** Adding an `active` CSS class to buttons isn't enough for screen readers to understand the state. Dynamically generated input fields also often lack explicit labels.
**Action:** When dynamically adding an `active` class, always synchronize it with the `aria-current="true"` attribute (and remove the attribute when inactive). Furthermore, ensure all `input` elements have an explicit `aria-label` or linked `label`, especially when they only rely on placeholder text visually.
