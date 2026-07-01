
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.
## 2024-07-01 - [Composite Input Focus State]
**Learning:** When creating composite UI components (like a prompt bar with an input and button), applying focus styles to the inner `input` element often looks disjointed.
**Action:** Use the `:focus-within` pseudo-class on the outer container to apply a unified focus ring to the entire component, and remove the `outline` on the inner input element to avoid double rings.
