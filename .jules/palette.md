
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.
## 2024-05-24 - Composite Input Focus States
**Learning:** When styling composite input components in the frontend UI (e.g., '.prompt-bar'), focusing the child input does not naturally highlight the parent container, which can lead to a disconnected or incomplete focus indicator for keyboard navigation.
**Action:** Apply interactive focus styles using the ':focus-within' pseudo-class on the wrapper container and explicitly set 'outline: none' on the inner input element to ensure a cohesive, accessible focus state without double rings.
