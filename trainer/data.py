import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from datasets import Dataset
from google.cloud import storage

logger = logging.getLogger(__name__)


@dataclass
class Persona:
    """Container for persona/system prompts."""

    system_prompt: str

    @classmethod
    def load(cls, path: str) -> "Persona":
        uri = Path(path)
        logger.info("Loading persona from %s", uri)
        with open(uri, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        system_prompt = raw.get("system") or raw.get("system_prompt")
        if not system_prompt:
            raise ValueError("Persona JSON must include a 'system' or 'system_prompt' key")
        return cls(system_prompt=system_prompt)


def load_jsonl(path: str) -> List[Dict[str, str]]:
    """Load a JSONL file from local disk.

    GCS paths should be synced locally before calling this helper.
    """

    records: List[Dict[str, str]] = []
    logger.info("Loading dataset from %s", path)
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    logger.info("Loaded %d records", len(records))
    return records


def format_chat_samples(records: Iterable[Dict[str, str]], persona: Persona) -> Dataset:
    """Format chat examples into a TRL-ready Hugging Face dataset."""

    prompts: List[Dict[str, str]] = []
    for item in records:
        user = item.get("user") or item.get("prompt")
        assistant = item.get("assistant") or item.get("response")
        if not user or not assistant:
            raise ValueError("Each training record must include 'user'/'assistant' text")
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{persona.system_prompt}\n<|eot_id|>\n"
            f"<|start_header_id|>user<|end_header_id|>\n{user}\n<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>"
        )
        prompts.append(
            {
                "prompt": prompt,
                "labels": assistant,
                "text": f"{prompt}{assistant}",
            }
        )
    dataset = Dataset.from_list(prompts)
    logger.info("Prepared %d formatted chat prompts", len(dataset))
    return dataset


def push_to_gcs(local_path: str, gcs_uri: str) -> None:
    """Upload a local file to the configured GCS bucket."""

    path_obj = Path(local_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Local path {local_path} does not exist")
    if not gcs_uri.startswith("gs://"):
        raise ValueError("gcs_uri must start with gs://")

    bucket_name, blob_path = _split_gs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    logger.info("Uploading %s to %s", local_path, gcs_uri)
    blob.upload_from_filename(local_path)
    logger.info("Upload complete")


def _split_gs_uri(gcs_uri: str) -> Tuple[str, str]:
    trimmed = gcs_uri.replace("gs://", "", 1)
    parts = trimmed.split("/", 1)
    if len(parts) != 2:
        raise ValueError("GCS URI must include bucket and path")
    return parts[0], parts[1]
