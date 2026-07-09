
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.
## 2024-06-25 - Custom contenteditable span focus state
**Learning:** When using `contenteditable="true"` on non-input elements (like spans or divs) for custom text inputs, they require `tabindex="0"` to be keyboard focusable in the document tab order before JavaScript initialization, as well as an explicit `:focus-visible` outline for an accessible focus state for keyboard users.
**Action:** Always include `tabindex="0"` and `:focus-visible` styles when implementing custom inputs via `contenteditable` on generic HTML elements.
