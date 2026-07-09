## 2026-02-23 - Missing Focus States in Custom Navigation
**Learning:** The custom button implementation for lab items completely lacks keyboard focus indicators, making navigation impossible for keyboard users. This is a critical pattern failure in this design system.
**Action:** Always verify custom interactive elements like `.lab-item` have explicit `:focus-visible` styles matching their hover states.

## 2026-02-23 - Duplicate ID issue
**Learning:** Having duplicate element IDs (`labSearch` was used twice in `index.html`) can cause the `getElementById` API to retrieve only the first match, resulting in bugs such as a secondary search bar not functioning properly. This violates basic HTML and accessibility rules regarding duplicate IDs.
**Action:** Always verify that input elements have unique IDs and ensure that event listeners are correctly assigned to each instance.

## 2026-02-23 - Accessible Empty States for Dynamic Search
**Learning:** The lab search component completely lacked a render function, resulting in an empty list. When implementing dynamic search filtering, it's critical to provide an accessible empty state (e.g., `<div role="status" aria-live="polite">`) to notify screen reader users when a search yields no results.
**Action:** Always ensure dynamic lists include a fallback UI for zero results with `aria-live="polite"` so the user understands the outcome of their search immediately.
