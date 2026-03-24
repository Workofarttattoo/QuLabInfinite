## 2026-02-23 - Missing Focus States in Custom Navigation
**Learning:** The custom button implementation for lab items completely lacks keyboard focus indicators, making navigation impossible for keyboard users. This is a critical pattern failure in this design system.
**Action:** Always verify custom interactive elements like `.lab-item` have explicit `:focus-visible` styles matching their hover states.
## 2025-10-30 - [Keyboard Accessibility Pattern]
**Learning:** The frontend component styles lacked `:focus-visible` globally. For mouse users this is fine, but for keyboard users this is inaccessible. Using `:focus-visible` while suppressing `:focus:not(:focus-visible)` solves this for this specific design system across global CSS and inline styles.
**Action:** Always apply `:focus-visible` styles to ensure keyboard navigation accessibility, and use `:focus:not(:focus-visible)` to suppress redundant browser outlines. Note that some component styles are defined in inline `<style>` blocks rather than globally.
