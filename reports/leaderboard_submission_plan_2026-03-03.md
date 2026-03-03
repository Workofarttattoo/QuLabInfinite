# Leaderboard Submission Kickoff Plan (March 3, 2026)

## Objective
Get Echo benchmarked on leaderboard-relevant tasks and start the external submission path immediately.

## Benchmarks Run Today
- Run directory: `benchmark_runs/leaderboard/leaderboard_20260303T203243Z`
- Model: `ech0:latest`
- Tasks: `winnogrande`, `truthfulqa`, `mmlu_college_physics`, `gorilla`
- Limit: 12 examples per task

### Results
- WinoGrande: 10/12 = 0.8333
- TruthfulQA MC1: 7/12 = 0.5833
- MMLU (college_physics): 7/12 = 0.5833
- Gorilla/BFCL exec simple exact-match: 8/12 = 0.6667
- Macro accuracy (4 tasks): 0.6667

## What Is Open Right Now
1. LMSYS Chatbot Arena (model-provider onboarding path exists)
2. Gorilla/BFCL leaderboard (submission path exists via maintainers)
3. Open benchmark directories/aggregators (OpenEvals + HF leaderboard docs)

## Submission Workstreams Started
1. Reproducible runner now supports:
- `winnogrande` alias -> `winogrande`
- `truthfulqa` alias -> `truthfulqa_mc1`
- `mmlu_<subject>`
- `gorilla` alias -> `gorilla_exec_simple` (`BFCL_v3_exec_simple.json`)

2. Leaderboard orchestration:
- `run_leaderboard_harness.py` can run mixed tasks and preserve exact prompts/outputs/hardware metadata.

3. Artifact layout for publishing:
- `benchmark_runs/leaderboard/leaderboard_<timestamp>/summary.json`
- per-task `predictions.jsonl` logs with exact prompts + raw outputs

## External Submission Path (Actionable)
1. BFCL/Gorilla submission packet
- Use the Gorilla run artifacts in `benchmark_runs/leaderboard/leaderboard_20260303T203243Z/repro_runs/gorilla_exec_simple_20260303_203350`
- Include exact command, model tag, and raw JSONL predictions.
- Submit through BFCL maintainers contact channel and request listing.

2. Arena provider onboarding
- Prepare production model endpoint metadata (model name, rate limits, uptime, safety settings, attribution).
- Start model-provider onboarding with Arena maintainers for listing.

3. Public reproducibility repo
- Publish a dedicated repo containing:
  - runner scripts
  - pinned commands
  - all benchmark artifacts for this run
  - rerun instructions
- Link this repo in all submission requests.

4. Repeat full-scale runs before final submission
- Increase from 12-shot quick validation to full benchmark slice per board requirements.
- Freeze model hash/tag and rerun once before final submission package.

## Current Blockers
- Hugging Face Open LLM Leaderboard legacy space was retired, so old direct submission flow is not available.
- External leaderboard listing still requires maintainer-side acceptance for Arena/BFCL.
- HF account auth/token is not configured in this environment, so automated HF publish is blocked until token is provided.

## Immediate Next Execution Steps
1. Push this PR with new harness support and today’s result links.
2. Prepare BFCL submission message + payload from current Gorilla run.
3. Prepare Arena onboarding message with endpoint/model metadata.
4. Run larger validation (same tasks, higher limit) and attach updated artifacts.
