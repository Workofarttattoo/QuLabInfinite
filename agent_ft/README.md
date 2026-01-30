# Agent Fine-Tuning Playbook

Scope: ech0 core, lab_agents/*, ech0_modules/*, qulab_ai tools.

## Data collection
- Mine successful tool traces: logs, transcripts, and smoke tests per lab.
- Generate synthetic tasks per lab (inputs, expected tool calls, golden outputs).
- Label failures/successes; keep edge cases (timeouts, missing params).
- Strip secrets/PII; redact API keys before storing.

## Safety/guardrails
- Filters: PII, secrets, unsafe chem/bio, prohibited exports.
- Add refusal templates and pre-checks in prompts; validate tool args before call.

## Training
- Start with SFT on curated traces; use LoRA/full SFT depending on model size.
- Optional RLHF/tool-use optimization on held-out interaction data.
- Track eval metrics: win rate per lab, tool-call accuracy, latency.

## Eval harness
- Per-lab task list; simulate tool calls; assert structured outputs.
- Regression suite should run against Master MCP server endpoints.

## Deployment
- Version prompts/models; expose via feature flags per lab.
- Roll out in stages: canary → internal → broad; keep rollback targets handy.

## Next steps
- Populate `agent_ft/data/` with curated traces and gold tasks.
- Add eval scripts that call Master MCP for end-to-end scoring.

## Vertex AI fine-tuning workflow
1. **Prepare data**  
   - Aggregate transcripts into chat format. The curated splits now live at `data/ech0_research_train.jsonl` (4 chats) and `data/ech0_research_eval.jsonl` (2 chats), both preserving the Joshua↔ECH0 persona metadata. Regenerate them from the `ech0_research_*.json` sources whenever you add new studies.  
   - Sanity-check the counts locally before uploading (`wc -l data/ech0_research_train.jsonl` / `wc -l data/ech0_research_eval.jsonl`).  
   - Upload to GCS (example):  
     ```bash
     gsutil cp data/ech0_research_train.jsonl gs://$BUCKET/ech0/train.jsonl
     gsutil cp data/ech0_research_eval.jsonl gs://$BUCKET/ech0/eval.jsonl
     ```
   - Keep persona/goals in a separate JSON you can prepend as system messages.
2. **Package trainer**  
   - Create `trainer/task.py` that loads LoRA/QLoRA config (trl/peft) and reads the JSONL files passed via CLI args.  
   - Include an `__init__.py`, `requirements.txt`, and any helper modules (e.g., `agent_ft/eval_harness.py` for validation).
3. **Submit CustomJob**  
   ```python
   from google.cloud import aiplatform

   aiplatform.init(project="your-project-id", location="us-central1")

   job = aiplatform.CustomJob.from_local_script(
       display_name="ech0-finetune",
       script_path="trainer/task.py",
       container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1:latest",
       requirements=[
           "transformers==4.38.0",
           "trl==0.7.10",
           "peft==0.9.0",
           "bitsandbytes==0.42.0",
           "datasets==2.17.0",
       ],
       args=[
           "--train_data_path", "gs://$BUCKET/ech0/train.jsonl",
           "--eval_data_path", "gs://$BUCKET/ech0/eval.jsonl",
           "--output_dir", "gs://$BUCKET/ech0/output",
           "--model_name", "meta-llama/Meta-Llama-3-8B",
           "--hf_token", "$HF_TOKEN",
       ],
       replica_count=1,
       machine_type="g2-standard-48",
       accelerator_type="NVIDIA_L4",
       accelerator_count=4,
   )

   job.run()
   ```
4. **Track artifacts**  
   - Outputs land at `gs://$BUCKET/ech0/output`. Pull adapters/checkpoints down and register with `master_mcp_server.py`.
   - Record hyperparams + dataset hashes in this README or `agent_ft/runs/` for reproducibility.
