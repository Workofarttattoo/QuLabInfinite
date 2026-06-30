
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.

## 2024-05-30 - Focus Within for Composite Inputs
**Learning:** Applying `:focus-within` to a wrapper component (like `.prompt-bar`) instead of styling the inner editable element directly provides a much more cohesive, aesthetically pleasing focus state.
**Action:** When creating composite input UI components, explicitly remove the outline from the input element itself (`outline: none;`) and rely on the `:focus-within` pseudo-class on the container to signal keyboard focus.
