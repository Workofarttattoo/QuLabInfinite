
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.
## 2024-05-18 - Missing focus states on custom contentEditable inputs
**Learning:** When building custom inputs (like using `contentEditable` inside a stylized wrapper like `.prompt-bar`), the default browser focus ring might not style the entire input area appropriately or might be manually hidden. Without it, keyboard users and screen readers lose context of where focus lies.
**Action:** Always explicitly define `:focus-within` on the parent wrapper container to provide a clear, accessible focus ring that encompasses the entire interactive area, and suppress the default outline on the child element.
