
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.

## 2025-03-05 - Focus Styles on Composite Input Containers
**Learning:** When creating composite input components in the frontend UI (like `.prompt-bar`), applying interactive focus styles directly to the input element may look disjointed or cause double outlines. Using the `:focus-within` pseudo-class on the wrapper container and explicitly setting `outline: none` on the inner input element ensures a cohesive, accessible focus state.
**Action:** Always verify focus rings on composite components and apply `:focus-within` to the container while suppressing inner outlines to maintain a polished interactive state.
