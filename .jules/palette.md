
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.

## 2025-03-05 - Focus Styles for Composite Inputs
**Learning:** When building composite input fields containing a styled container with an internal content-editable or input element, users easily lose track of keyboard focus because the child element captures focus but the container doesn't reflect it visually.
**Action:** Use `:focus-within` on the `.prompt-bar` parent container to apply a cohesive visual focus ring (e.g., matching theme accents), and enforce `outline: none` on the inner input to avoid double rings, ensuring a unified and accessible focus state.
