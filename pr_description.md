🧹 [Code Health] Group risk factors into RiskFactors dataclass

🎯 **What:** The `calculate_composite_risk_score` function in `alzheimers_early_detection.py` had an excessive number of parameters (8 parameters). It was refactored to accept a single `RiskFactors` dataclass object instead.

💡 **Why:** Having too many parameters in a function signature makes the code harder to read, harder to maintain, and more prone to errors when passing arguments (e.g. swapping argument positions). Grouping related risk factors into a single, cohesive `RiskFactors` dataclass significantly improves the maintainability and readability of the codebase without changing the underlying business logic.

✅ **Verification:** A mock verification script was written to bypass missing dependencies (like numpy and fastapi) and was executed to ensure the engine runs successfully without throwing `TypeError` parameter issues or changing the computed composite risk score. The output score remained exactly the same (`61.0` in the mock case).

✨ **Result:** The function signature is drastically simplified, improving maintainability and aligning better with modern python coding practices (dataclasses).
