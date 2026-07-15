
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.
## 2024-05-18 - Prompt Bar Focus Within
**Learning:** When dealing with compound input components (like a container with an input field and a submit button), applying focus styles directly to the internal input element can break the visual cohesion of the component.
**Action:** Use the `:focus-within` pseudo-class on the outer container to apply a unified focus ring, and ensure `outline: none` is applied to the internal input to prevent double focus rings, creating a much cleaner accessible experience.
