#!/usr/bin/env python3
import subprocess

title = "🧪 Add tests for create_lab_documentation"
body = """🎯 **What:** The testing gap addressed is the lack of unit tests for the `create_lab_documentation` function in `create_mcp_docs.py`, which is responsible for generating HTML documentation from tool configurations.

📊 **Coverage:** The new tests cover:
- Happy paths with fully populated tool attributes.
- Edge cases where optional properties (like `parameters`, `example_params`, `example_response`, `scientific_basis`, `references`, `status`) are missing, ensuring they fall back to expected defaults properly.
- All conditional formatting for tool statuses (`working`, `partial`, `placeholder`, and unknown values).
- An empty list of tools being passed.

✨ **Result:** A significant improvement in test coverage that acts as a safety net, allowing confident refactoring without fear of regressions to the `create_mcp_docs.py` core functionality."""

# Fallback method if submit tool fails to load directly
print(f"Submitting PR: {title}\n\n{body}")
