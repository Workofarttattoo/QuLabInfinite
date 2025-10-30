from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable, Dict, Any, List
from pydantic import BaseModel
import argparse

class DataSource(ABC):
    """Abstract base class for all data source plugins."""

    name: str = "UnnamedDataSource"
    description: str = "No description available."

    @classmethod
    @abstractmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Add source-specific arguments to the command-line parser."""
        pass

    @abstractmethod
    def load(self, args: argparse.Namespace) -> Iterable[BaseModel]:
        """
        Load data from the source.

        Args:
            args: The parsed command-line arguments.

        Returns:
            An iterable of Pydantic models representing the loaded data.
        """
        pass
