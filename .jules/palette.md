## 2024-05-24 - Dynamic Chat Interfaces & Custom Input Containers

**Learning:** Custom styled search/prompt components (like `.prompt-bar` and `.lab-search`) often strip native focus rings. Additionally, when using a dynamic output container (like `#labChat`) for AI responses, screen readers will be completely unaware of new messages unless the container is explicitly marked as a live region.

**Action:** Always ensure that custom wrapper elements containing inputs use `:focus-within` to indicate focus, use `:focus-visible` on interactive elements, and wrap dynamic message containers with `aria-live="polite"` so screen readers can announce AI responses automatically.
