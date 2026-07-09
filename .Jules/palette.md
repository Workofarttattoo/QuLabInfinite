## 2026-02-23 - Missing Focus States in Custom Navigation
**Learning:** The custom button implementation for lab items completely lacks keyboard focus indicators, making navigation impossible for keyboard users. This is a critical pattern failure in this design system.
**Action:** Always verify custom interactive elements like `.lab-item` have explicit `:focus-visible` styles matching their hover states.

## 2026-02-23 - Duplicate ID issue
**Learning:** Having duplicate element IDs (`labSearch` was used twice in `index.html`) can cause the `getElementById` API to retrieve only the first match, resulting in bugs such as a secondary search bar not functioning properly. This violates basic HTML and accessibility rules regarding duplicate IDs.
**Action:** Always verify that input elements have unique IDs and ensure that event listeners are correctly assigned to each instance.

## 2026-06-01 - DOM Initialisation vs State Management
**Learning:** When dynamically generating and initializing DOM elements in index.html's renderLabs, if state-management functions like setActiveLab rely on document.querySelectorAll() to apply attributes (like aria-current), the new elements must be appended to the DOM *before* invoking the function to prevent missing initial states.
**Action:** Ensure DOM nodes are attached to the document tree before querying them for state updates or attribute modifications.
