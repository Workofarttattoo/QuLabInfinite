## 2026-02-23 - Missing Focus States in Custom Navigation
**Learning:** The custom button implementation for lab items completely lacks keyboard focus indicators, making navigation impossible for keyboard users. This is a critical pattern failure in this design system.
**Action:** Always verify custom interactive elements like `.lab-item` have explicit `:focus-visible` styles matching their hover states.

## 2026-03-16 - Custom Input Wrappers Stripping Focus and Dynamic Containers Missing ARIA Live
**Learning:** In the frontend UI (e.g., `website/qulab.aios.is/index.html`), dynamic output containers like `#labChat` lack `aria-live` preventing screen reader announcements. Additionally, custom-styled input wrappers consistently strip keyboard focus rings due to missing `:focus-within` styling when child inputs use `outline: none`.
**Action:** Always ensure dynamic output containers use `aria-live="polite"` for screen reader support, and explicitly implement `:focus-within` and `:focus-visible` styles for any custom input wrappers to preserve keyboard focus rings.
