
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.

## 2025-03-05 - Skip-to-Content Links
**Learning:** For users relying on keyboard navigation or screen readers, bypassing repeated navigation links to reach the main content is a critical accessibility requirement. Adding a visually hidden "skip to main content" link that becomes visible on `:focus` provides this ability while maintaining visual design. Ensure the main content is wrapped in a `<main>` tag with an `id` that matches the link's `href` (e.g., `<main id="main-content">`), and use smooth scrolling (`html { scroll-behavior: smooth; }`) for a better user experience when the jump occurs.
**Action:** When working on application entry points or main layouts, always verify the presence of a "Skip to main content" link and the corresponding semantic `<main>` tag.
