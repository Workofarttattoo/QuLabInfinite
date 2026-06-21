
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.
## 2025-03-05 - Interactive Focus Styles on Composite Inputs
**Learning:** When styling composite input components in the frontend UI (like a `.prompt-bar` container with an inner `contentEditable` input), placing focus states on the wrapper instead of the inner element makes for a more cohesive UI. However, this causes double focus rings if the inner element still retains default outline behavior.
**Action:** Always apply the `:focus-within` pseudo-class on the outer wrapper for visual cohesion, and explicitly set `outline: none` on the inner interactive element to suppress browser defaults.
