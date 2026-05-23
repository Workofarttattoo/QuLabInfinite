## 2026-02-23 - Missing Focus States in Custom Navigation
**Learning:** The custom button implementation for lab items completely lacks keyboard focus indicators, making navigation impossible for keyboard users. This is a critical pattern failure in this design system.
**Action:** Always verify custom interactive elements like `.lab-item` have explicit `:focus-visible` styles matching their hover states.

## 2026-02-23 - Duplicate ID issue
**Learning:** Having duplicate element IDs (`labSearch` was used twice in `index.html`) can cause the `getElementById` API to retrieve only the first match, resulting in bugs such as a secondary search bar not functioning properly. This violates basic HTML and accessibility rules regarding duplicate IDs.
**Action:** Always verify that input elements have unique IDs and ensure that event listeners are correctly assigned to each instance.

## 2026-05-23 - Accessible Active States and Input Labels
**Learning:** Relying purely on visual `.active` classes and placeholder text creates a significant barrier for screen reader users, who miss the context of which tab is active and what an input is for without explicit ARIA attributes.
**Action:** Always synchronize visual active states with `aria-current="true"` and ensure placeholder-only inputs have an explicit `aria-label`.
