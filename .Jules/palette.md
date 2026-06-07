## 2026-02-23 - Missing Focus States in Custom Navigation
**Learning:** The custom button implementation for lab items completely lacks keyboard focus indicators, making navigation impossible for keyboard users. This is a critical pattern failure in this design system.
**Action:** Always verify custom interactive elements like `.lab-item` have explicit `:focus-visible` styles matching their hover states.

## 2026-02-23 - Duplicate ID issue
**Learning:** Having duplicate element IDs (`labSearch` was used twice in `index.html`) can cause the `getElementById` API to retrieve only the first match, resulting in bugs such as a secondary search bar not functioning properly. This violates basic HTML and accessibility rules regarding duplicate IDs.
**Action:** Always verify that input elements have unique IDs and ensure that event listeners are correctly assigned to each instance.

## 2024-06-07 - Custom Textbox Focus Indicator
**Learning:** When using a non-standard input element like a `span` with `contentEditable="true"` and `role="textbox"` for complex UI interactions, browsers often do not provide a default focus outline or styling. This leads to a severe accessibility gap where keyboard users are unable to perceive when the input has focus, significantly impairing navigation.
**Action:** Always verify that custom text inputs have explicit focus styling (e.g., via `:focus` or `:focus-within` on the parent container) to ensure the focus state is clearly communicated to users relying on keyboard navigation.
