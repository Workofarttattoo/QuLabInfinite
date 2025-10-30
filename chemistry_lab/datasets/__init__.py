"""Machine-learning dataset registry for the chemistry laboratory."""

from __future__ import annotations
import pandas as pd
from typing import Dict, Iterable, List, Optional
from .base import DatasetDescriptor
from .registry import DATASET_REGISTRY

__all__ = [
    "DATASET_REGISTRY",
    "get_dataset",
    "list_datasets",
]
