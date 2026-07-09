## 2026-02-23 - Missing Focus States in Custom Navigation
**Learning:** The custom button implementation for lab items completely lacks keyboard focus indicators, making navigation impossible for keyboard users. This is a critical pattern failure in this design system.
**Action:** Always verify custom interactive elements like `.lab-item` have explicit `:focus-visible` styles matching their hover states.

## 2026-02-23 - Duplicate ID issue
**Learning:** Having duplicate element IDs (`labSearch` was used twice in `index.html`) can cause the `getElementById` API to retrieve only the first match, resulting in bugs such as a secondary search bar not functioning properly. This violates basic HTML and accessibility rules regarding duplicate IDs.
**Action:** Always verify that input elements have unique IDs and ensure that event listeners are correctly assigned to each instance.
## 2024-04-11 - Prompt Input Keyboard Accessibility
**Learning:** The main natural language input fields (`.prompt-input` and `.prompt-bar input`) lacked visible focus indicators, making keyboard navigation difficult for screen reader and keyboard-only users. Contenteditable spans often miss default browser focus outlines compared to standard `<input>` fields.
**Action:** Always ensure that custom interactive elements (like contenteditable spans) and custom styled inputs explicitly define a `:focus-visible` state using a high-contrast outline to maintain keyboard accessibility without annoying mouse users.
