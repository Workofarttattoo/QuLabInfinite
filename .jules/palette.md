
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.

## 2023-10-25 - Unified Focus Rings on Composite Inputs
**Learning:** For composite input containers (like a `.prompt-bar` holding a contenteditable `span`), default focus outlines on the inner element can look disjointed or be invisible. Using the CSS `:focus-within` pseudo-class on the container allows a unified, branded focus ring (e.g., using `--accent`) while suppressing the default outline on the inner element.
**Action:** When designing custom input components with wrapper divs, apply `:focus-within` to the wrapper to handle focus styling and set `outline: none;` on the inner interactive element.
