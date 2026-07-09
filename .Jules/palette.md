## 2026-02-23 - Missing Focus States in Custom Navigation
**Learning:** The custom button implementation for lab items completely lacks keyboard focus indicators, making navigation impossible for keyboard users. This is a critical pattern failure in this design system.
**Action:** Always verify custom interactive elements like `.lab-item` have explicit `:focus-visible` styles matching their hover states.

## 2026-02-23 - Duplicate ID issue
**Learning:** Having duplicate element IDs (`labSearch` was used twice in `index.html`) can cause the `getElementById` API to retrieve only the first match, resulting in bugs such as a secondary search bar not functioning properly. This violates basic HTML and accessibility rules regarding duplicate IDs.
**Action:** Always verify that input elements have unique IDs and ensure that event listeners are correctly assigned to each instance.
## 2024-04-18 - [Add aria-current to active lab buttons]
**Learning:** For interactive list elements that use custom `.active` classes to denote the current selection (like the lab buttons in `qulab.aios.is/index.html`), screen readers miss the state change unless `aria-current="true"` is synchronized.
**Action:** Always map `.active` class toggles to `aria-current="true"`/`"false"` programmatic updates in the JS render loop.
