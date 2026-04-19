## 2026-02-23 - Missing Focus States in Custom Navigation
**Learning:** The custom button implementation for lab items completely lacks keyboard focus indicators, making navigation impossible for keyboard users. This is a critical pattern failure in this design system.
**Action:** Always verify custom interactive elements like `.lab-item` have explicit `:focus-visible` styles matching their hover states.

## 2026-02-23 - Duplicate ID issue
**Learning:** Having duplicate element IDs (`labSearch` was used twice in `index.html`) can cause the `getElementById` API to retrieve only the first match, resulting in bugs such as a secondary search bar not functioning properly. This violates basic HTML and accessibility rules regarding duplicate IDs.
**Action:** Always verify that input elements have unique IDs and ensure that event listeners are correctly assigned to each instance.

## 2026-02-23 - Missing ARIA Labels on Implicitly-Labeled Inputs
**Learning:** Adding inputs without associated labels or `aria-label` attributes breaks screen reader accessibility, even if a placeholder text is present. Additionally, toggleable `.active` classes on buttons for active selection states must be synchronized with the `aria-current="true"` property to convey the selection to assistive technologies.
**Action:** Always ensure any icon-only buttons or standalone input fields use explicit `aria-label`s, and coordinate `.active` state toggling with `aria-current="true"` updates on dynamic JS menus.
