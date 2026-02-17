import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any

logger = logging.getLogger(__name__)

from materials_lab.materials_lab import MaterialsLab
from quantum_lab.quantum_lab import QuantumLabSimulator
from chemistry_lab.chemistry_lab import ChemistryLaboratory
from core.base_lab import BaseLab
from core.config import ConfigManager

class UnifiedSimulator:
    """
    A unified interface for managing and running simulations across all
    available laboratories in QuLabInfinite.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the UnifiedSimulator and loads all available labs.
        """
        self.config_manager = ConfigManager(config_path)
        self.labs: Dict[str, BaseLab] = self._load_labs()

    def _load_labs(self) -> Dict[str, BaseLab]:
        """
        Load all available laboratories.
        """
        logger.info("Loading all available labs...")
        
        # Get lab-specific configs
        mat_config = self.config_manager.get_lab_config("materials_lab")
        qnt_config = self.config_manager.get_lab_config("quantum_lab")
        chm_config = self.config_manager.get_lab_config("chemistry_lab")

        labs = {
            "materials": MaterialsLab(config=mat_config),
            "quantum": QuantumLabSimulator(
                verbose=False,
                config=qnt_config,
                num_qubits=qnt_config.get("default_qubits", 5),
                backend=qnt_config.get("default_backend", "statevector")
            ),
            "chemistry": ChemistryLaboratory(config=chm_config),
        }
        logger.info("All labs loaded.")
        return labs

    def run_simulation(self, lab_name: str, experiment_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a simulation in a specified lab.

        Args:
            lab_name: The name of the lab to use (e.g., 'materials', 'quantum', 'chemistry').
            experiment_spec: The specification for the experiment to run.

        Returns:
            The results of the simulation.
        """
        if lab_name not in self.labs:
            raise ValueError(f"Unknown lab: {lab_name}. Available labs are: {list(self.labs.keys())}")
        
        lab = self.labs[lab_name]
        return lab.run_experiment(experiment_spec)

    def get_lab_status(self, lab_name: str) -> Dict[str, Any]:
        """
        Get the status of a specific lab.

        Args:
            lab_name: The name of the lab.

        Returns:
            The status of the lab.
        """
        if lab_name not in self.labs:
            raise ValueError(f"Unknown lab: {lab_name}")
        
        return self.labs[lab_name].get_status()

    def list_labs(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available labs and their capabilities.
        """
        return {name: lab.get_capabilities() for name, lab in self.labs.items()}

if __name__ == '__main__':
    simulator = UnifiedSimulator()

    print("\nListing available labs and their capabilities:")
    print(simulator.list_labs())

    # Example: Run a materials science experiment
    print("\nRunning a materials science experiment...")
    material_experiment = {
        "experiment_type": "tensile",
        "material_name": "Ti-6Al-4V",
        "max_strain": 0.15
    }
    material_results = simulator.run_simulation("materials", material_experiment)
    print("Material experiment results:", material_results)

    # Example: Run a quantum experiment
    print("\nRunning a quantum experiment...")
    quantum_experiment = {
        "experiment_type": "bell_pair"
    }
    quantum_results = simulator.run_simulation("quantum", quantum_experiment)
    print("Quantum experiment results:", quantum_results)
