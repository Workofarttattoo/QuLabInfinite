import logging
import yaml
from typing import Dict, Any

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "materials_lab": {
        "index_on_load": True,
        "default_test_strain": 0.15
    },
    "quantum_lab": {
        "default_backend": "statevector",
        "optimize_for_m4": True,
        "default_qubits": 5
    },
    "chemistry_lab": {
        "enable_md": True,
        "enable_reactions": True,
        "default_force_field": "AMBER",
        "default_qm_method": "DFT"
    }
}

class ConfigManager:
    """
    A unified configuration manager for QuLabInfinite.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from a YAML file, or create a default one.
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                # You might want to merge with defaults to handle missing keys
                return config
        except FileNotFoundError:
            logger.info(f"Config file not found at {self.config_path}. Creating a default one.")
            with open(self.config_path, 'w') as f:
                yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False)
            return DEFAULT_CONFIG
        except Exception as e:
            logger.exception("Failed to load config file")
            return DEFAULT_CONFIG

    def get_lab_config(self, lab_name: str) -> Dict[str, Any]:
        """
        Get the configuration for a specific lab.
        """
        return self._config.get(lab_name, {})

    def get_config(self) -> Dict[str, Any]:
        """
        Get the entire configuration dictionary.
        """
        return self._config

    def set(self, key: str, value: Any):
        """
        Set a configuration value. E.g., set("quantum_lab.default_backend", "tensor_network")
        """
        keys = key.split('.')
        d = self._config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
        self.save()

    def save(self):
        """
        Save the current configuration to the file.
        """
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False)
        except Exception as e:
            logger.exception("Failed to save config file")
