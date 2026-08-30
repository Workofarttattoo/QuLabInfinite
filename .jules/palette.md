
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.
## 2024-05-24 - Keyboard Shortcuts & Dynamic Empty States
**Learning:** When adding keyboard shortcuts like '/', explicitly visualizing them with <kbd> elements dramatically improves discoverability for screen reader and keyboard power users. Furthermore, dynamic lists filtered by search require explicit, constructive empty states to provide clear feedback when no results match, preventing user confusion.
**Action:** Always pair global keyboard event listeners with a visible, stylized UI hint, and enforce empty states on client-side filtered lists.
