## 2026-02-23 - Missing Focus States in Custom Navigation
**Learning:** The custom button implementation for lab items completely lacks keyboard focus indicators, making navigation impossible for keyboard users. This is a critical pattern failure in this design system.
**Action:** Always verify custom interactive elements like `.lab-item` have explicit `:focus-visible` styles matching their hover states.

## 2026-02-23 - Duplicate ID issue
**Learning:** Having duplicate element IDs (`labSearch` was used twice in `index.html`) can cause the `getElementById` API to retrieve only the first match, resulting in bugs such as a secondary search bar not functioning properly. This violates basic HTML and accessibility rules regarding duplicate IDs.
**Action:** Always verify that input elements have unique IDs and ensure that event listeners are correctly assigned to each instance.
## 2026-02-23 - Synchronize aria-current for active elements
**Learning:** Toggling an `.active` CSS class on custom UI elements does not automatically communicate the selected state to assistive technologies. When managing selection states manually, it's critical to synchronize ARIA attributes (like `aria-current="true"`) programmatically with the visual active class.
**Action:** Whenever implementing a custom list of toggleable interactive elements (like custom tabs or buttons), explicitly add `aria-current="true"` to the active element and remove it from inactive ones.
