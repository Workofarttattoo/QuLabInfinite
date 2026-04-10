"""Tests for core infrastructure: BaseLab, Registry, Config."""

import pytest
from qulab.core.base_lab import BaseLab, register_lab, get_registered_labs
from qulab.core.config import ConfigManager
from qulab.core.registry import LabRegistry


class TestBaseLab:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseLab()

    def test_register_lab_decorator(self):
        @register_lab("test_lab_1", category="test", description="A test lab")
        class TestLab(BaseLab):
            def run_experiment(self, spec):
                return {"sum": spec.get("a", 0) + spec.get("b", 0)}
            def get_status(self):
                return {"ready": True}

        assert "test_lab_1" in get_registered_labs()
        lab = TestLab()
        result = lab.run_experiment({"a": 3, "b": 5})
        assert result["sum"] == 8
        caps = lab.get_capabilities()
        assert caps["category"] == "test"

    def test_lab_default_capabilities(self):
        @register_lab("test_lab_2", category="demo")
        class DemoLab(BaseLab):
            def run_experiment(self, spec):
                return {}
            def get_status(self):
                return {}

        lab = DemoLab()
        caps = lab.get_capabilities()
        assert caps["name"] == "test_lab_2"


class TestConfig:
    def test_default_config(self):
        config = ConfigManager(config_path="/nonexistent.yaml")
        assert config.get("server.port") == 8000
        assert config.get("server.host") == "0.0.0.0"

    def test_get_lab_config(self):
        config = ConfigManager(config_path="/nonexistent.yaml")
        qlab = config.get_lab_config("quantum_lab")
        assert qlab["default_qubits"] == 5

    def test_get_missing_key(self):
        config = ConfigManager(config_path="/nonexistent.yaml")
        assert config.get("nonexistent.key") is None
        assert config.get("nonexistent.key", "default") == "default"


class TestRegistry:
    def test_registry_creation(self):
        reg = LabRegistry()
        assert reg.lab_count >= 0

    def test_get_unknown_lab_raises(self):
        reg = LabRegistry()
        reg._discovered = True  # Skip discovery
        with pytest.raises(ValueError, match="Unknown lab"):
            reg.get_lab("nonexistent_lab")
