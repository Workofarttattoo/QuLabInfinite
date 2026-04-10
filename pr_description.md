🧪 Add tests for create_market_domination_inventions

🎯 **What**
Added a unit test suite for the `create_market_domination_inventions` function located in `ech0_market_domination_displays.py`. In addition, I also had to fix syntax errors inside `ech0_invention_accelerator.py` and `ech0_invention_poc_pipeline.py` which prevented the files from being compiled in python, preventing execution.

📊 **Coverage**
The new test file (`tests/test_ech0_market_domination_displays.py`) covers:
- Verification that exactly 25 inventions are returned
- Verification that all elements returned are instances of `InventionConcept`
- Verification that each concept has a valid string name and description
- Specific verification that a subset of expected inventions exist in the payload

✨ **Result**
The `create_market_domination_inventions` public function is now tested, and its core behavior will be protected from regressions. In addition, the pipeline syntax errors are fixed allowing successful execution.
