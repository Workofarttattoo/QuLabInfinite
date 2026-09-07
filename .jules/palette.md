
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.
## 2023-10-24 - Missing Focus Outlines on Composite UI Input Elements
**Learning:** When dealing with composite UI input elements (like a `.prompt-bar` containing a `contenteditable` span) in vanilla JS frontends, relying on default focus rings on the inner element can look disjointed or be insufficient.
**Action:** Use the `:focus-within` pseudo-class on the wrapper element to provide a unified accessibility focus ring, and apply `outline: none` to the inner element. This ensures the entire composite element appears focused when the user interacts with the input, improving keyboard accessibility and visual coherence.
