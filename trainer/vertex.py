import logging
from typing import List, Optional

from google.cloud import aiplatform

logger = logging.getLogger(__name__)


def launch_vertex_job(
    display_name: str,
    python_package_gcs_uri: str,
    bucket: str,
    train_data_path: str,
    eval_data_path: str,
    persona_path: str,
    hf_token: str,
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    output_dir: str = "/gcs/your-bucket/llama-output",
    machine_type: str = "g2-standard-12",
    accelerator_type: str = "NVIDIA_L4",
    accelerator_count: int = 1,
    adapter_args: Optional[List[str]] = None,
    project: Optional[str] = None,
    location: str = "us-central1",
) -> aiplatform.CustomPythonPackageTrainingJob:
    """Submit a Vertex AI custom training job for the packaged trainer."""

    aiplatform.init(project=project, location=location)
    job = aiplatform.CustomPythonPackageTrainingJob(
        display_name=display_name,
        python_package_gcs_uri=python_package_gcs_uri,
        python_module_name="trainer.task",
        container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1:latest",
    )

    base_args = [
        "--train_data_path",
        train_data_path,
        "--eval_data_path",
        eval_data_path,
        "--persona_path",
        persona_path,
        "--hf_token",
        hf_token,
        "--model_name",
        model_name,
        "--output_dir",
        output_dir,
    ]

    full_args = base_args + (adapter_args or [])

    logger.info("Launching Vertex job %s with args: %s", display_name, full_args)
    job.run(
        args=full_args,
        replica_count=1,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        base_output_dir=f"gs://{bucket}/llama-output",
    )
    return job
