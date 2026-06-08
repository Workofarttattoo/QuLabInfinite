
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.
## 2024-05-18 - Improved Chat Input Focus Area
**Learning:** Users naturally click anywhere in a surrounding input container (like a search bar or chat box) expecting it to focus the input field, but by default it only focuses if they click the text directly.
**Action:** Always wrap text inputs in a container with a click listener that focuses the input, and use `:focus-within` on the container to display the focus state, while removing the default `outline` on the input itself.
