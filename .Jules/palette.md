## 2026-02-23 - Missing Focus States in Custom Navigation
**Learning:** The custom button implementation for lab items completely lacks keyboard focus indicators, making navigation impossible for keyboard users. This is a critical pattern failure in this design system.
**Action:** Always verify custom interactive elements like `.lab-item` have explicit `:focus-visible` styles matching their hover states.

## 2025-05-14 - Global Missing Focus States
**Learning:** The entire website lacks visible focus states (`:focus-visible`), which makes keyboard navigation impossible. This is a common pattern in this repository where visual polish was prioritized over accessibility.
**Action:** Always add explicit `:focus-visible` styling using design tokens (like `var(--primary)`) to all interactive elements globally (buttons, links, cards).
