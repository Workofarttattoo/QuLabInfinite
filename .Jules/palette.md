## 2026-02-23 - Missing Focus States in Custom Navigation
**Learning:** The custom button implementation for lab items completely lacks keyboard focus indicators, making navigation impossible for keyboard users. This is a critical pattern failure in this design system.
**Action:** Always verify custom interactive elements like `.lab-item` have explicit `:focus-visible` styles matching their hover states.

## 2026-03-12 - Preserving Keyboard Focus in Custom Input Wrappers
**Learning:** Custom UI wrappers for inputs (like `.lab-search` and `.prompt-bar`) often strip the native browser focus ring on the inner `<input>` to look seamless, which completely breaks keyboard navigation visibility.
**Action:** When styling custom input wrappers that hide the native outline, ALWAYS apply `:focus-within` to the wrapper element itself so that it visually indicates when its internal input has focus. Combine with `:focus-visible` and `:focus:not(:focus-visible)` to distinguish keyboard from mouse users without breaking accessibility.
