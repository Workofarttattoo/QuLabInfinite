import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Mock dependencies to avoid import errors
sys.modules['materials_lab.materials_lab'] = MagicMock()
sys.modules['quantum_lab.quantum_simulator'] = MagicMock()
sys.modules['chemistry_lab.synthesis_optimizer'] = MagicMock()
sys.modules['oncology_lab.cancer_simulator'] = MagicMock()
sys.modules['chemistry_lab.qulab_ai_integration'] = MagicMock()
sys.modules['frequency_lab.qulab_ai_integration'] = MagicMock()
sys.modules['qulab_ai.production'] = MagicMock()
sys.modules['psutil'] = MagicMock()
sys.modules['fastapi'] = MagicMock()
sys.modules['fastapi.middleware.cors'] = MagicMock()
sys.modules['fastapi.responses'] = MagicMock()
sys.modules['pydantic'] = MagicMock()
sys.modules['numpy'] = MagicMock()

def reset_modules():
    for mod in ['api.unified_api', 'api.production_api', 'core.security']:
        if mod in sys.modules:
            del sys.modules[mod]

class TestCORSConfiguration(unittest.TestCase):
    def test_cors_custom_origins(self):
        reset_modules()
        custom_origins = "https://custom.com, https://another.com"
        with patch.dict(os.environ, {"QULAB_ALLOWED_ORIGINS": custom_origins, "QU_LAB_MASTER_KEYS": "securekey123"}, clear=True):
            from core.security import get_allowed_origins
            self.assertEqual(get_allowed_origins(), ["https://custom.com", "https://another.com"])

    def test_cors_disallows_wildcard(self):
        reset_modules()
        with patch.dict(os.environ, {"QULAB_ALLOWED_ORIGINS": "*", "QU_LAB_MASTER_KEYS": "securekey123"}, clear=True):
            from core.security import get_allowed_origins
            self.assertEqual(get_allowed_origins(), ["https://qulab.ai"])

    def test_a_unified_api_cors_defaults(self):
        reset_modules()
        with patch.dict(os.environ, {"QU_LAB_MASTER_KEYS": "securekey123"}, clear=True):
            from api.unified_api import app, ALLOWED_ORIGINS
            self.assertEqual(ALLOWED_ORIGINS, ["https://qulab.ai", "https://api.qulab.ai"])
            from fastapi.middleware.cors import CORSMiddleware
            calls = app.add_middleware.call_args_list
            cors_call = next(c for c in calls if c.args[0] == CORSMiddleware)
            self.assertEqual(cors_call.kwargs["allow_origins"], ["https://qulab.ai", "https://api.qulab.ai"])

    def test_b_production_api_cors_defaults(self):
        reset_modules()
        with patch.dict(os.environ, {"QU_LAB_MASTER_KEYS": "securekey123"}, clear=True):
            from api.production_api import app, ALLOWED_ORIGINS
            self.assertEqual(ALLOWED_ORIGINS, ["https://qulab.ai", "https://api.qulab.ai"])
            from fastapi.middleware.cors import CORSMiddleware
            calls = app.add_middleware.call_args_list
            cors_call = next(c for c in calls if c.args[0] == CORSMiddleware)
            self.assertEqual(cors_call.kwargs["allow_origins"], ["https://qulab.ai", "https://api.qulab.ai"])

if __name__ == "__main__":
    unittest.main()
