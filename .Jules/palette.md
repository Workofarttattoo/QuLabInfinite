## 2026-02-23 - Missing Focus States in Custom Navigation
**Learning:** The custom button implementation for lab items completely lacks keyboard focus indicators, making navigation impossible for keyboard users. This is a critical pattern failure in this design system.
**Action:** Always verify custom interactive elements like `.lab-item` have explicit `:focus-visible` styles matching their hover states.

## 2026-02-23 - Duplicate ID issue
**Learning:** Having duplicate element IDs (`labSearch` was used twice in `index.html`) can cause the `getElementById` API to retrieve only the first match, resulting in bugs such as a secondary search bar not functioning properly. This violates basic HTML and accessibility rules regarding duplicate IDs.
**Action:** Always verify that input elements have unique IDs and ensure that event listeners are correctly assigned to each instance.

## 2026-05-21 - Screen Reader Support for Active Navigation & Placeholder Inputs
**Learning:** Relying solely on visual CSS classes like `.active` and placeholder text leaves screen reader users without context about their current navigation state or the purpose of input fields.
**Action:** Always synchronize visual active states with `aria-current="true"` and ensure inputs visually relying on placeholders have an explicit `aria-label`.
