"""Test configuration and shared fixtures."""

import os
import sys

import pytest

# Ensure qulab package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("QULAB_AUTH_ENABLED", "false")
os.environ.setdefault("QULAB_API_KEYS", "test-key")


@pytest.fixture
def config():
    from qulab.core.config import ConfigManager
    return ConfigManager()


@pytest.fixture
def registry():
    from qulab.core.registry import LabRegistry
    reg = LabRegistry()
    return reg
