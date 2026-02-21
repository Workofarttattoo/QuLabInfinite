"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

__init__ - Part of Chemistry Lab
"""

"""Machine-learning dataset registry for the chemistry laboratory."""

from __future__ import annotations
import pandas as pd
from typing import Dict, Iterable, List, Optional
from .base import DatasetDescriptor
from .registry import DATASET_REGISTRY, get_dataset, list_datasets

__all__ = [
    "DATASET_REGISTRY",
    "get_dataset",
    "list_datasets",
]
