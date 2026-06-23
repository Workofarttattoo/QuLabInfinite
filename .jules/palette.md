
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.
## 2024-05-24 - Focus states on composite input containers
**Learning:** When using custom composite inputs (like a `contenteditable` span inside a styled container), the native outline can look awkward or not appear at all. Adding `:focus-within` to the parent container and removing the inner focus outline provides a much cleaner, more cohesive visual feedback loop.
**Action:** Always check interactive composite containers for proper focus rings and use `:focus-within` combined with `outline: none` on the inner element to ensure accessible and consistent styling without double rings.
