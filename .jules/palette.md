
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.

## 2025-03-08 - Added Focus Outlines to Composite Input Components
**Learning:** For accessibility on custom UI elements that lack standard form elements (such as `.prompt-bar` serving as an input container holding a `contenteditable` span), the native focus styles are missing or misaligned. Applying `:focus-within` on the container element provides a unified, clearly visible focus ring when users focus the child element via keyboard or click, improving UX.
**Action:** When implementing custom composite inputs, suppress internal outlines (`outline: none;` on inner `contenteditable` or un-styled `<input>`) and style `:focus-within` on the parent container (e.g., using `--accent-soft` glow) to maintain a cohesive semantic focus indicator.
