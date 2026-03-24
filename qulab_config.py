# QuLab Configuration File
# Central configuration for all QuLab modules

import os
from typing import Dict, Any

class QuLabConfig:
    """Central configuration management"""

    def __init__(self):
        self.settings = {
            'debug': os.getenv('QULAB_DEBUG', 'False').lower() == 'true',
            'log_level': os.getenv('QULAB_LOG_LEVEL', 'INFO'),
            'api_timeout': int(os.getenv('QULAB_API_TIMEOUT', '30')),
            'cache_enabled': os.getenv('QULAB_CACHE_ENABLED', 'True').lower() == 'true',
            'materials_db_path': os.getenv('QULAB_MATERIALS_DB', './data/materials.db'),
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.settings.get(key, default)

# Global config instance
config = QuLabConfig()
