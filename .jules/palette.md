## 2025-10-30 - Dynamic Content Accessibility
**Learning:** Screen readers often miss dynamic content updates in custom UI like the lab chat interface. Similarly, custom CSS wrappers around inputs frequently strip or obscure the native focus ring, breaking keyboard navigation cues.
**Action:** Always add `aria-live="polite"` to dynamic output containers. Ensure input wrappers explicitly use `:focus-within` combined with `:focus-visible` logic to restore accessible focus indicators.
