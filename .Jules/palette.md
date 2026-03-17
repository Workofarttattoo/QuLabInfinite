## 2026-02-23 - Missing Focus States in Custom Navigation
**Learning:** The custom button implementation for lab items completely lacks keyboard focus indicators, making navigation impossible for keyboard users. This is a critical pattern failure in this design system.
**Action:** Always verify custom interactive elements like `.lab-item` have explicit `:focus-visible` styles matching their hover states.

## 2026-03-17 - Missing Focus States in Buttons
**Learning:** The button elements (`.btn-primary` and `.btn-secondary`) in `website/styles.css` lack keyboard focus indicators, making navigation difficult for keyboard users.
**Action:** Ensure all interactive elements, especially primary and secondary buttons, have explicit `:focus-visible` styles to maintain keyboard accessibility, and suppress redundant mouse focus outlines with `:focus:not(:focus-visible)`.
