
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.
## 2024-10-25 - Composite Input Focus States
**Learning:** When a custom input component consists of a wrapper styling a bare, `contentEditable` or nested input element (like `.prompt-bar` and `.prompt-input`), native browser focus fails to intuitively highlight the component's interactive boundaries.
**Action:** Apply the `:focus-within` pseudo-class to the wrapper container for a cohesive focus ring and explicitly set `outline: none` on the inner interactive element to prevent awkward double rings.
