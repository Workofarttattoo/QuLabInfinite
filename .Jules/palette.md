## 2026-02-23 - Missing Focus States in Custom Navigation
**Learning:** The custom button implementation for lab items completely lacks keyboard focus indicators, making navigation impossible for keyboard users. This is a critical pattern failure in this design system.
**Action:** Always verify custom interactive elements like `.lab-item` have explicit `:focus-visible` styles matching their hover states.

## 2026-02-23 - Duplicate ID issue
**Learning:** Having duplicate element IDs (`labSearch` was used twice in `index.html`) can cause the `getElementById` API to retrieve only the first match, resulting in bugs such as a secondary search bar not functioning properly. This violates basic HTML and accessibility rules regarding duplicate IDs.
**Action:** Always verify that input elements have unique IDs and ensure that event listeners are correctly assigned to each instance.
## 2026-05-28 - DOM Injection Sequence and ARIA Sync
**Learning:** When dynamically generating UI elements, functions that depend on DOM querying (like `document.querySelectorAll`) to apply state attributes (like `aria-current`) will fail if the new elements haven't been appended to the DOM yet. Furthermore, visual active classes must always be paired with `aria-current="true"` for screen reader accessibility.
**Action:** Always append newly created elements to the DOM before invoking state-management functions that query them, and strictly synchronize visual active classes with semantic ARIA attributes.
