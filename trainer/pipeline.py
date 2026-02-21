import json
import logging
from pathlib import Path
from typing import Optional, Tuple

from google.cloud import storage

from trainer.data import push_to_gcs
from trainer.vertex import launch_vertex_job

logger = logging.getLogger(__name__)


def stage_curated_data(train_local: str, eval_local: str, bucket: str) -> Tuple[str, str]:
    """Push curated JSONL datasets to the canonical GCS locations."""

    train_uri = f"gs://{bucket}/ech0/train.jsonl"
    eval_uri = f"gs://{bucket}/ech0/eval.jsonl"
    push_to_gcs(train_local, train_uri)
    push_to_gcs(eval_local, eval_uri)
    return train_uri, eval_uri


def download_job_artifacts(bucket: str, gcs_prefix: str, local_dir: str) -> Path:
    """Copy model artifacts from GCS to a local folder."""

    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    prefix = gcs_prefix.rstrip("/")
    target_dir = Path(local_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for blob in bucket_obj.list_blobs(prefix=prefix):
        relative = blob.name.replace(prefix, "", 1).lstrip("/")
        dest = target_dir / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s to %s", blob.name, dest)
        blob.download_to_filename(dest)
    return target_dir


def register_with_mcp(artifact_dir: str, registry_path: str = "ech0_mcp_weights.json") -> Path:
    """Record the latest fine-tune artifacts for downstream MCP consumers."""

    payload = {"active_weights_path": artifact_dir}
    registry = Path(registry_path)
    registry.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Registered new weights at %s", registry)
    return registry


def run_end_to_end(
    display_name: str,
    python_package_gcs_uri: str,
    bucket: str,
    train_local: str,
    eval_local: str,
    persona_path: str,
    hf_token: str,
    adapter_args: Optional[list[str]] = None,
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    accelerator_type: str = "NVIDIA_L4",
    accelerator_count: int = 1,
    machine_type: str = "g2-standard-12",
):
    train_uri, eval_uri = stage_curated_data(train_local, eval_local, bucket)

    adapter_args = adapter_args or []

    job = launch_vertex_job(
        display_name=display_name,
        python_package_gcs_uri=python_package_gcs_uri,
        bucket=bucket,
        train_data_path=train_uri,
        eval_data_path=eval_uri,
        persona_path=persona_path,
        hf_token=hf_token,
        model_name=model_name,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        adapter_args=adapter_args,
    )

    # Wait for job completion implicitly handled by job.run()
    artifact_dir = download_job_artifacts(bucket, "llama-output", f"/gcs/{bucket}/llama-output")
    register_with_mcp(str(artifact_dir))
    return job, artifact_dir
