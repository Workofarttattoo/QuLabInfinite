## 2026-02-23 - Missing Focus States in Custom Navigation
**Learning:** The custom button implementation for lab items completely lacks keyboard focus indicators, making navigation impossible for keyboard users. This is a critical pattern failure in this design system.
**Action:** Always verify custom interactive elements like `.lab-item` have explicit `:focus-visible` styles matching their hover states.

## 2026-02-23 - Duplicate ID issue
**Learning:** Having duplicate element IDs (`labSearch` was used twice in `index.html`) can cause the `getElementById` API to retrieve only the first match, resulting in bugs such as a secondary search bar not functioning properly. This violates basic HTML and accessibility rules regarding duplicate IDs.
**Action:** Always verify that input elements have unique IDs and ensure that event listeners are correctly assigned to each instance.

## 2026-06-04 - DOM element querying for state attributes
**Learning:** When dynamically generating DOM elements that require state attributes like `aria-current` via a central function like `setActiveLab`, querying the DOM using `document.querySelectorAll()` inside the state function will fail to locate the new elements unless they are appended to the document's parent node *before* the function call.
**Action:** Ensure dynamic elements are fully appended to the DOM before invoking functions that query the DOM to set their initial states.
