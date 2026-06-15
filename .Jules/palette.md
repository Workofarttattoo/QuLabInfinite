## 2026-02-23 - Missing Focus States in Custom Navigation
**Learning:** The custom button implementation for lab items completely lacks keyboard focus indicators, making navigation impossible for keyboard users. This is a critical pattern failure in this design system.
**Action:** Always verify custom interactive elements like `.lab-item` have explicit `:focus-visible` styles matching their hover states.

## 2026-02-23 - Duplicate ID issue
**Learning:** Having duplicate element IDs (`labSearch` was used twice in `index.html`) can cause the `getElementById` API to retrieve only the first match, resulting in bugs such as a secondary search bar not functioning properly. This violates basic HTML and accessibility rules regarding duplicate IDs.
**Action:** Always verify that input elements have unique IDs and ensure that event listeners are correctly assigned to each instance.

## 2026-04-14 - Missing aria-current on active lab buttons
**Learning:** The custom `.lab-button` implementation in the vanilla HTML/JS frontend toggles the `.active` class for visual indication but fails to programmatically synchronize `aria-current="true"` for screen reader accessibility. This makes it impossible for screen reader users to identify which lab is currently selected.
**Action:** Always ensure that when visually toggling an active state on interactive elements (e.g., via a `.active` class), the corresponding ARIA attribute (like `aria-current="true"`) is also updated simultaneously.
