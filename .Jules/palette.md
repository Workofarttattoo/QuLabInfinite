## 2026-02-23 - Missing Focus States in Custom Navigation
**Learning:** The custom button implementation for lab items completely lacks keyboard focus indicators, making navigation impossible for keyboard users. This is a critical pattern failure in this design system.
**Action:** Always verify custom interactive elements like `.lab-item` have explicit `:focus-visible` styles matching their hover states.

## 2024-03-25 - Focus-visible for accessibility in application
**Learning:** Found a systemic lack of focus indicators for all standard interactive elements like buttons and links across the marketing and platform sites (`index.html` and `qulab.aios.is/index.html`). This is a critical pattern failure impacting all keyboard users in the main app flows.
**Action:** Establish a universal `:focus-visible` rule consistently across all global stylesheets (`styles.css`) and inline HTML style blocks to ensure all interactive elements receive clear focus rings.
