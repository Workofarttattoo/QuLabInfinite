from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable
import pandas as pd


class DataSource(ABC):
    """Abstract interface for a data source that returns a canonical dataframe.

    The dataframe must include at minimum the following columns:
    - material_id (str)
    - formula (str, optional)
    - property_name (str)
    - value (float)
    - units (str)
    - temperature_k (float, optional)
    - source (str)
    - source_ref (str, optional)
    """

    @abstractmethod
    def fetch(self, identifiers: Iterable[str]) -> pd.DataFrame:
        """Fetch data for the given identifiers and return a canonical dataframe."""
        raise NotImplementedError
