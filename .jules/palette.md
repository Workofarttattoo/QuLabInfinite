
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.

## 2025-05-15 - Glassmorphism Dashboard Integration
**Learning:** Glassmorphism (backdrop-filter) paired with Tailwind's `group-hover` and `focus-visible` creates a high-fidelity 'tactical' feel while maintaining accessibility. Ensuring `tabindex="0"` on custom dashboard cards is critical for keyboard navigation in complex grid layouts.
**Action:** Applied glass-panel styling and standardized focus-visible rings across all 7 dashboard modules in `website/qulab.aios.is/index.html`.
