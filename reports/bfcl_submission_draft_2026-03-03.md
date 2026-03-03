# BFCL Submission Draft (March 3, 2026)

Hello BFCL maintainers,

I would like to submit an Echo model run for leaderboard inclusion.

## Model
- Name: `ech0:latest`
- Provider/Org: Workofarttattoo / QuLabInfinite
- Runtime: local Ollama-compatible endpoint

## Benchmark Slice
- Dataset file: `BFCL_v3_exec_simple.json`
- Evaluator mode: exact-match against `ground_truth` function calls
- Samples: 12
- Correct: 8
- Accuracy: 0.6667

## Reproducibility Artifacts
- Summary JSON: `benchmark_runs/leaderboard/leaderboard_20260303T203243Z/summary.json`
- Gorilla task summary: `benchmark_runs/leaderboard/leaderboard_20260303T203243Z/repro_runs/gorilla_exec_simple_20260303_203350/summary.json`
- Predictions JSONL: `benchmark_runs/leaderboard/leaderboard_20260303T203243Z/repro_runs/gorilla_exec_simple_20260303_203350/predictions.jsonl`
- Hardware log: `benchmark_runs/leaderboard/leaderboard_20260303T203243Z/repro_runs/gorilla_exec_simple_20260303_203350/hardware.json`
- Run manifest: `benchmark_runs/leaderboard/leaderboard_20260303T203243Z/repro_runs/gorilla_exec_simple_20260303_203350/run_manifest.json`

## Command Used
`python benchmarks/reproducibility/run_leaderboard_harness.py --model ech0:latest --tasks winnogrande,truthfulqa,mmlu_college_physics,gorilla --limit 12 --mode repro --timeout 120`

Please let me know the exact submission format you prefer (JSON upload, repo link, or endpoint evaluation), and I will provide it immediately.
