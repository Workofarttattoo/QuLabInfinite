import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

logger = logging.getLogger(__name__)

class BaseLab(ABC):
    """
    Abstract base class for all laboratories in QuLabInfinite.

    This class defines the common interface that all specialized labs
    (e.g., MaterialsLab, ChemistryLaboratory, QuantumLabSimulator) must implement.
    This ensures a consistent API for initialization, configuration,
    and execution of simulations across the platform.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the laboratory with a given configuration.

        Args:
            config: A dictionary of configuration parameters.
        """
        self.config = config or {}
        logger.info(f"Initializing {self.__class__.__name__}...")

    @abstractmethod
    def run_experiment(self, experiment_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a specific experiment based on the provided specification.

        Args:
            experiment_spec: A dictionary defining the experiment to be run,
                             including parameters and desired outputs.

        Returns:
            A dictionary containing the results of the experiment.
        """
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the laboratory.

        Returns:
            A dictionary containing status information, such as backend state,
            loaded data, and available capabilities.
        """
        pass

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get the capabilities of the laboratory.

        Returns:
            A dictionary describing the capabilities of the lab.
        """
        return {"name": self.__class__.__name__, "capabilities": "N/A"}
