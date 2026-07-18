
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.

## 2024-05-30 - Added focus state to contenteditable inputs
**Learning:** Found that `.prompt-input` which uses a `contenteditable` span lacked explicit focus styles in the CSS compared to button and input elements, resulting in missing or jarring default browser outlines.
**Action:** Consistently ensure that `contenteditable` regions use `outline: none` and apply a `:focus-within` structural enhancement to their parent container for a cohesive focus ring.
