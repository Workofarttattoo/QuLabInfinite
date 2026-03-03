# Benchmark Reproducibility Pack (Phase 4)

This pack prioritizes **reproducibility over leaderboard reporting**.

## What gets logged

Each benchmark run logs:
- exact prompts (`prompts.jsonl`)
- exact model outputs (`outputs.jsonl`)
- hardware snapshot (`hardware.json`)
- run configuration (`config.json`)
- summary + full record set (`summary.json`, `full_results.json`)

## Run ARC subset

```bash
python bench/repro_benchmark.py --benchmark arc --limit 200 --model-command "ollama run llama3.1"
```

## Run MMLU subset

```bash
python bench/repro_benchmark.py --benchmark mmlu --limit 200 --mmlu-subject high_school_physics --model-command "ollama run llama3.1"
```

## Run full benchmark split

Use a high enough `--limit` to include the full split available from the source dataset.

## Publish a GitHub reproducibility repo

1. Run benchmark(s) and collect artifacts under `artifacts/repro_benchmark/<run_id>/`.
2. Create a new repository (e.g. `qulab-benchmark-repro`).
3. Copy this folder structure into the new repository:
   - `bench/repro_benchmark.py`
   - `reproducibility/README.md`
   - `artifacts/repro_benchmark/<run_id>/`
4. Add a short run manifest with:
   - commit SHA of QuLab Infinite
   - model command used
   - benchmark command(s) used
5. Push to GitHub.

Anyone can now rerun with the same prompt/output/hardware visibility.
