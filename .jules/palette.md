
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.
## 2026-04-30 - Focus and ARIA for Custom Interactive Elements
**Learning:** When dynamically toggling active states on buttons (like lab selections) or using custom `contenteditable` spans for text inputs, screen readers require explicit `aria-current` updates and `tabindex="0"` to maintain focusability and state awareness.
**Action:** Always programmatically sync `aria-current` with `.active` classes and assign `tabindex="0"` to non-standard input elements.
