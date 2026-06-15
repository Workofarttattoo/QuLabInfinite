
## 2025-03-05 - Missing Focus Outlines on Marketing Site Elements
**Learning:** Default browser focus rings are frequently suppressed by CSS resets or custom button stylings, hindering keyboard accessibility. For the `website/` marketing site, using the CSS `:focus-visible` pseudo-class allows providing clear, custom focus outlines (e.g., using `--primary` brand colors) specifically to keyboard users without degrading the visual experience for mouse users.
**Action:** When creating or modifying interactive elements (like `a`, `button`, `.btn-primary`, `.doc-card`), always define corresponding `:focus-visible` states. Ensure to suppress the fallback outline on `:focus:not(:focus-visible)` to prevent double outlines.

## 2025-04-25 - ContentEditable Keyboard Navigation
**Learning:** Elements that use JavaScript `contentEditable = true` (like the `.prompt-input` span used for chat) are not naturally keyboard focusable in HTML until the JS runs or unless they have a `tabindex`. This breaks screen reader and keyboard navigation expectations before interaction. Adding `tabindex="0"` to the HTML ensures it's in the document tab order. Additionally, ensuring these custom input elements have a `:focus-visible` outline makes the focus state accessible for keyboard users without affecting mouse users.
**Action:** When building custom text inputs using `contenteditable` on non-input elements (like `span` or `div`), always include `tabindex="0"` in the HTML markup and define a `:focus-visible` CSS rule for an accessible focus ring.
