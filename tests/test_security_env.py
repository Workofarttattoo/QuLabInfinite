import os
import sys
import unittest
from unittest import mock
import importlib

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock missing dependencies
sys.modules["fastapi"] = mock.MagicMock()
sys.modules["fastapi.security"] = mock.MagicMock()
sys.modules["jose"] = mock.MagicMock()
sys.modules["passlib"] = mock.MagicMock()
sys.modules["passlib.context"] = mock.MagicMock()

# Mock logging config to avoid file creation
logging_mock = mock.MagicMock()
sys.modules["qulab_ai.production.logging_config"] = logging_mock
logging_mock.get_logger.return_value = mock.MagicMock()

# Import the module to test
try:
    from qulab_ai.production import security
except ImportError as e:
    print(f"Import failed: {e}")
    security = None

class TestSecurityConfig(unittest.TestCase):
    def setUp(self):
        if security is None:
            self.skipTest("Could not import qulab_ai.production.security due to missing dependencies")

    def test_secret_key_from_env(self):
        """Test that SECRET_KEY is loaded from QULAB_SECRET_KEY environment variable."""
        test_key = "test_secret_key_12345"
        with mock.patch.dict(os.environ, {"QULAB_SECRET_KEY": test_key}):
            importlib.reload(security)
            self.assertEqual(security.SECRET_KEY, test_key)

    def test_secret_key_default(self):
        """Test that SECRET_KEY is generated randomly if QULAB_SECRET_KEY is not set."""
        # Ensure the env var is NOT set for this test
        with mock.patch.dict(os.environ):
            if "QULAB_SECRET_KEY" in os.environ:
                del os.environ["QULAB_SECRET_KEY"]

            importlib.reload(security)
            key1 = security.SECRET_KEY

            # It should be a string and not empty
            self.assertIsInstance(key1, str)
            self.assertTrue(len(key1) > 0)

            # Reload again to verify it changes (random)
            importlib.reload(security)
            key2 = security.SECRET_KEY
            self.assertNotEqual(key1, key2)

if __name__ == "__main__":
    unittest.main()
