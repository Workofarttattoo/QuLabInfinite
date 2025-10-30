# Benchmark Registry

This directory holds YAML definitions that describe calibration / validation
benchmarks. Each benchmark links to raw data, canonical summaries, the engine
version, and acceptance criteria. The goal is to make every engine change
repeatable and auditable.

Layout:
- `mechanics/` – solid mechanics benchmarks (e.g., Johnson-Cook fits).
- `quantum/` – quantum chemistry / algorithm validation.

Each YAML file should contain:
- `id`: unique identifier.
- `summary`: short description of the physics/chemistry scenario.
- `inputs`: list of dimensional inputs.
- `criteria`: pass/fail gates.
- `data_ref`: pointers to raw + canonical datasets.
- `engine`: implementation version & calibration script.
- `reports`: markdown/nb reports generated after calibration.
- `metadata`: owner, last calibration date, notes.

Use `bench/run_golden_paths.py` to execute all registered benchmarks.
