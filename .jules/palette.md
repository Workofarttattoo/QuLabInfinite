
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.

## 2025-05-13 - Add Skip-to-Content Links
**Learning:** Implementing skip-to-content links helps screen reader and keyboard users navigate past recurring header elements efficiently. Adding aria-labels to text inputs improves context for screen readers.
**Action:** Add `.skip-link` pointing to `#main-content` and provide `aria-label` attributes to text inputs in form bars.
