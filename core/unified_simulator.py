import logging
import os
import sys
from typing import Any, Dict, Mapping

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chemistry_lab.chemistry_lab import ChemistryLaboratory
from core.base_lab import BaseLab
from core.config import ConfigManager
from core.runtime import RuntimeRegistry, Tool
from materials_lab.materials_lab import MaterialsLab
from quantum_lab.quantum_lab import QuantumLabSimulator

logger = logging.getLogger(__name__)


class LabTool(Tool):
    """Adapter that exposes a lab as a runtime Tool."""

    def __init__(self, name: str, lab: BaseLab):
        self._name = name
        self._lab = lab

    @property
    def name(self) -> str:
        return self._name

    def describe(self) -> Mapping[str, Any]:
        return self._lab.get_capabilities()

    def run(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._lab.run_experiment(dict(payload))


class UnifiedSimulator:
    """Canonical runtime entrypoint for all QuLabInfinite simulation tools."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_manager = ConfigManager(config_path)
        self.runtime = RuntimeRegistry()
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        logger.info("Registering runtime tools...")

        mat_config = self.config_manager.get_lab_config("materials_lab")
        qnt_config = self.config_manager.get_lab_config("quantum_lab")
        chm_config = self.config_manager.get_lab_config("chemistry_lab")

        labs: Dict[str, BaseLab] = {
            "materials": MaterialsLab(config=mat_config),
            "quantum": QuantumLabSimulator(
                verbose=False,
                config=qnt_config,
                num_qubits=qnt_config.get("default_qubits", 5),
                backend=qnt_config.get("default_backend", "statevector"),
            ),
            "chemistry": ChemistryLaboratory(config=chm_config),
        }

        for tool_name, lab in labs.items():
            self.runtime.register(LabTool(tool_name, lab))

        logger.info("Runtime tools ready: %s", ", ".join(sorted(labs)))

    def run_simulation(self, lab_name: str, experiment_spec: Dict[str, Any]) -> Dict[str, Any]:
        artifact = self.runtime.run(lab_name, experiment_spec)
        return artifact.result

    def run_simulation_artifact(self, lab_name: str, experiment_spec: Dict[str, Any]) -> str:
        artifact = self.runtime.run(lab_name, experiment_spec)
        return artifact.to_json()

    def get_lab_status(self, lab_name: str) -> Dict[str, Any]:
        tool = self.runtime.get_tool(lab_name)
        if isinstance(tool, LabTool):
            return tool._lab.get_status()
        return {"status": "unknown"}

    def list_labs(self) -> Dict[str, Dict[str, Any]]:
        return self.runtime.list_tools()


_simulator_singleton: UnifiedSimulator | None = None


def get_simulator() -> UnifiedSimulator:
    global _simulator_singleton
    if _simulator_singleton is None:
        _simulator_singleton = UnifiedSimulator()
    return _simulator_singleton


if __name__ == "__main__":
    simulator = get_simulator()
    print(simulator.list_labs())
