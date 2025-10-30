"""
Specialized dataset descriptor for the Open Quantum Data Commons (OpenQDC).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .base import DatasetDescriptor


@dataclass
class OpenQDCDescriptor(DatasetDescriptor):
    """
    A DatasetDescriptor that dynamically discovers sub-datasets from a YAML registry.
    """
    registry_path: str = "data/raw/quantum/openqdc_registry.yaml"
    sub_datasets: Dict[str, DatasetDescriptor] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """
        Load sub-datasets from the YAML registry after the object is initialized.
        """
        self.load_registry()

    def load_registry(self):
        """
        Parses the OpenQDC YAML registry and creates DatasetDescriptors for each entry.
        """
        registry_file = Path(self.registry_path)
        if not registry_file.exists():
            return

        with registry_file.open("r", encoding="utf-8") as handle:
            registry = yaml.safe_load(handle)

        if not registry:
            return

        for name, data in registry.items():
            sub_dataset = DatasetDescriptor(
                name=f"openqdc_{name}",
                category="quantum-chemistry",
                description=data.get("notes", f"OpenQDC dataset: {name}"),
                url="https://github.com/OpenQDC/open-quantum-data-commons",
                citation="OpenQDC contributors (ongoing).",
                local_hint=data.get("cache_dir"),
                file_extensions=[".csv", ".json", ".h5"],
                notes=f"~{data.get('approx_size_gb', 'N/A')} GB. Download with: {data.get('download_hint')}"
            )
            self.sub_datasets[name] = sub_dataset

    def list_sub_datasets(self) -> List[str]:
        """
        Return a list of the names of the discovered sub-datasets.
        """
        return sorted(self.sub_datasets.keys())

    def get_sub_dataset(self, name: str) -> Optional[DatasetDescriptor]:
        """
        Return the DatasetDescriptor for a specific sub-dataset.
        """
        return self.sub_datasets.get(name)
