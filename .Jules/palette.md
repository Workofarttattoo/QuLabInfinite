## 2024-05-22 - Handling Long Lists with Client-Side Filtering
**Learning:** Users struggle to scan long lists of items (like 100+ labs) without search functionality. Client-side filtering provides instant feedback and significantly improves usability for static datasets.
**Action:** Always include a search/filter input for lists with more than 20 items to allow quick access via typing.

## 2024-05-22 - Accessible Search Inputs without Visible Labels
**Learning:** Search inputs often rely on placeholders for design minimalism, which can disappear or be insufficient for screen readers.
**Action:** Ensure inputs without visible labels have a clear `aria-label` (e.g., `aria-label="Filter labs list"`) to provide context to assistive technologies.
