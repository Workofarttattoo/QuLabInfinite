
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.
## 2024-06-13 - [Make dynamic chat updates accessible]
**Learning:** Adding dynamic chat messages to a DOM element via JavaScript without `aria-live` means screen readers won't announce them. This is a crucial pattern for any AI chat interface.
**Action:** Always ensure chat containers use `aria-live="polite"` and `aria-relevant="additions"` so users are notified of new assistant responses automatically without losing context.
