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

## Output

Each run creates `benchmark_runs/<dataset>_<timestamp>/` with:

- `hardware.json`
- `run_manifest.json`
- `predictions.jsonl`
- `summary.json`
- `repro_repo/` (GitHub-ready reproducibility folder)

`repro_repo/` can be committed directly to a public GitHub repository so anyone can rerun with the exact command and metadata.
