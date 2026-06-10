
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.
## 2024-05-24 - Dynamic Chat Content Requires ARIA Live Regions
**Learning:** The dynamic chat interface injects messages into the DOM without notifying screen readers, making new content invisible to assistive technologies.
**Action:** Always add `aria-live="polite"` and `aria-relevant="additions"` to the parent container of chat message lists to ensure newly appended elements are announced automatically without stealing focus.
