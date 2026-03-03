# Phase 4: Benchmark Clean-Up (Reproducibility First)

This workflow intentionally prioritizes reproducibility over leaderboard submission.

## What this runner logs

For every run it records:
- exact prompt sent to the model
- exact raw model output
- normalized prediction
- gold label
- per-example correctness
- hardware/runtime metadata
- full command/arguments used

## Datasets

- Full MMLU subject split via `cais/mmlu` (e.g. `college_physics`)
- ARC-Challenge subset via `allenai/ai2_arc`
- ARC-Easy subset via `allenai/ai2_arc`
- WinoGrande via `winogrande/winogrande_xl`
- TruthfulQA MC1 via `truthfulqa/truthful_qa`
- Gorilla/BFCL (`BFCL_v3_exec_simple.json`) via `gorilla-llm/Berkeley-Function-Calling-Leaderboard`

## Usage

```bash
python benchmarks/reproducibility/run_llm_repro_benchmark.py \
  --dataset mmlu \
  --mmlu-subject college_physics \
  --split test \
  --limit 50 \
  --model ech0-polymath-14b \
  --backend ollama
```

Custom backend command:

```bash
python benchmarks/reproducibility/run_llm_repro_benchmark.py \
  --dataset arc_challenge \
  --split test \
  --limit 100 \
  --model my-model-id \
  --backend command \
  --command-template "python local_infer.py --model my-model-id --prompt {prompt}"
```

## Leaderboard Harness (lm-eval + fallback)

Use this when you want a leaderboard-oriented command that still preserves
reproducibility artifacts:

```bash
python benchmarks/reproducibility/run_leaderboard_harness.py \
  --model ech0:latest \
  --tasks winnogrande,truthfulqa,mmlu_college_physics,gorilla_exec_simple \
  --limit 25 \
  --mode auto
```

`--mode auto` behavior:
- tries `lm_eval` first (local OpenAI-compatible API at `http://127.0.0.1:11434/v1`)
- if `lm_eval` fails/times out, falls back to `run_llm_repro_benchmark.py`

Artifacts are written under:
- `benchmark_runs/leaderboard/leaderboard_<timestamp>/summary.json`
- plus per-run logs, hardware metadata, and runner-specific outputs

## Output

Each run creates `benchmark_runs/<dataset>_<timestamp>/` with:

- `hardware.json`
- `run_manifest.json`
- `predictions.jsonl`
- `summary.json`
- `repro_repo/` (GitHub-ready reproducibility folder)

`repro_repo/` can be committed directly to a public GitHub repository so anyone can rerun with the exact command and metadata.
