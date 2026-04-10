"""
Unified configuration manager for QuLabInfinite.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

DEFAULT_CONFIG: Dict[str, Any] = {
    "app": {
        "name": "QuLabInfinite",
        "version": "1.0.0",
        "debug": False,
        "log_level": "INFO",
    },
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 4,
        "cors_origins": ["*"],
    },
    "labs": {
        "auto_discover": True,
        "materials_lab": {
            "index_on_load": True,
            "default_test_strain": 0.15,
        },
        "quantum_lab": {
            "default_backend": "statevector",
            "optimize_for_m4": True,
            "default_qubits": 5,
        },
        "chemistry_lab": {
            "enable_md": True,
            "enable_reactions": True,
            "default_force_field": "AMBER",
            "default_qm_method": "DFT",
        },
    },
    "database": {
        "url": os.getenv("DATABASE_URL", "sqlite:///qulab.db"),
    },
    "monitoring": {
        "prometheus_port": 9090,
        "enable_metrics": True,
    },
}


class ConfigManager:
    """
    Unified configuration manager with YAML file support and env var overrides.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.getenv("QULAB_CONFIG", "config.yaml")
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load config from YAML file, falling back to defaults."""
        if YAML_AVAILABLE and Path(self.config_path).exists():
            try:
                with open(self.config_path) as f:
                    file_config = yaml.safe_load(f) or {}
                return self._deep_merge(DEFAULT_CONFIG, file_config)
            except Exception as exc:
                logger.warning("Failed to load config from %s: %s", self.config_path, exc)

        return DEFAULT_CONFIG.copy()

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """Deep merge two dicts, override wins."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigManager._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value using dot notation: 'api.port'."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def get_lab_config(self, lab_name: str) -> Dict[str, Any]:
        """Get config for a specific lab."""
        return self._config.get("labs", {}).get(lab_name, {})

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    def set(self, key: str, value: Any) -> None:
        """Set a config value using dot notation."""
        keys = key.split(".")
        d = self._config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    def save(self, path: Optional[str] = None) -> None:
        """Save current config to YAML file."""
        if not YAML_AVAILABLE:
            logger.warning("PyYAML not installed, cannot save config")
            return
        save_path = path or self.config_path
        with open(save_path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
