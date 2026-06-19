
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.

## 2025-03-05 - Missing ARIA Live Region for Dynamic Chat
**Learning:** For dynamic regions like a chat window (`<div id="chatWindow">`), new messages injected via JavaScript aren't automatically announced by screen readers unless specifically configured. Using `aria-live="polite"` coupled with `aria-relevant="additions"` guarantees that when new DOM nodes (messages) are appended, screen readers will queue and read them out to users without interrupting their current task.
**Action:** Whenever implementing a dynamic log, feed, or chat component, explicitly include `aria-live` (usually "polite") and `aria-relevant="additions"` on the container element.
