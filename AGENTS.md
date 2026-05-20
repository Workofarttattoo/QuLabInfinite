# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

QuLabInfinite is a Python (FastAPI) scientific simulation platform with 100+ virtual laboratories. The XGBoost hail prediction model lives in `hail_model/`.

### Lab Console GUI

```bash
bash scripts/start-qulab-gui.sh   # http://127.0.0.1:5173 (not :3000 — Grafana uses 3000 in docker-compose)
PYTHONPATH=. python3 unified_mcp_server.py   # MCP :8102
```

### Running the hail model

```bash
# Train on synthetic data
PYTHONPATH=/workspace python3 -m hail_model.train

# Run inference
PYTHONPATH=/workspace python3 -m hail_model.predict

# Validate a saved model
PYTHONPATH=/workspace python3 -m hail_model.validate
```

### Key gotchas

- **`pyproject.toml` build-backend typo**: The build-backend is set to `hatchling.backends` but the correct module is `hatchling.build`. Editable installs (`pip install -e .`) will fail. Install dependencies manually via `pip install` instead.
- **`PYTHONPATH` required**: Since editable install doesn't work, always set `PYTHONPATH=/workspace` when running Python commands or tests.
- **Auto-discovery crashes the API server**: `qulab/labs/medical/oncology_lab/quick_sanity_check.py` calls `sys.exit(0)` at module level. The registry's `except Exception` doesn't catch `SystemExit` (which inherits from `BaseException`), so `uvicorn qulab.api.main:app` fails during startup. Individual labs can still be tested by importing them directly.
- **No `@register_lab` decorators on labs**: The newer labs (genomics, cardiology, etc.) do not use the `@register_lab` decorator from `qulab.core.base_lab`, so they don't appear in the registry. Import and use them directly.
- **API auth in dev mode**: When `QULAB_API_KEY` is unset, the API allows unauthenticated access.

### Testing

```bash
# Hail model tests (fast, self-contained)
PYTHONPATH=/workspace pytest tests/test_hail_model.py -v

# Core framework tests
PYTHONPATH=/workspace pytest tests/test_core.py tests/test_runtime_core.py -v
```

### Linting

```bash
ruff check hail_model/ tests/test_hail_model.py
ruff check qulab/
```
