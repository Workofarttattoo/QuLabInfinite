🧹 [Code Health] Remove redundant NotImplementedError in ABC

## Description
🎯 **What:** Replaced `raise NotImplementedError` with `...` (Ellipsis) in `RateLimitStore` abstract methods. Also fixed a hardcoded log path `/Users/noone/QuLabInfinite/logs` in `logging_config.py` that caused a `PermissionError` and prevented script execution.
💡 **Why:** Using `...` or `pass` is the pythonic standard for defining abstract methods when the `@abstractmethod` decorator from `abc` module is used, avoiding unnecessary instantiation of exceptions. The path fix ensures environment portability.
✅ **Verification:** Formatted with `black`. Ran `qulab_ai/production/security.py` directly to confirm initialization and verified that `pytest tests/test_security_env.py` and `tests/ -k security` passed successfully.
✨ **Result:** A cleaner, idiomatic abstract base class implementation and a robust relative log path.
