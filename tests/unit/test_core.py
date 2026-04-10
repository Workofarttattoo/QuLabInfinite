"""
Tests for QuLabInfinite core: BaseLab, Registry, Config, Simulator.
"""

import pytest
from typing import Dict, Any

from qulab.core.base_lab import BaseLab, register_lab, LabMetadata, _LAB_REGISTRY
from qulab.core.config import ConfigManager
from qulab.core.registry import LabRegistry


# ------------------------------------------------------------------
# BaseLab tests
# ------------------------------------------------------------------


class ConcreteTestLab(BaseLab):
    """Minimal BaseLab implementation for testing."""

    def run_experiment(self, experiment_spec: Dict[str, Any]) -> Dict[str, Any]:
        self._track_experiment()
        return {"result": "ok", **experiment_spec}

    def get_status(self) -> Dict[str, Any]:
        return {"status": "running", "experiments": self._experiment_count}


class TestBaseLab:
    def test_init_with_config(self):
        lab = ConcreteTestLab(config={"key": "value"})
        assert lab.config["key"] == "value"

    def test_init_without_config(self):
        lab = ConcreteTestLab()
        assert lab.config == {}

    def test_run_experiment(self):
        lab = ConcreteTestLab()
        result = lab.run_experiment({"type": "test"})
        assert result["result"] == "ok"
        assert result["type"] == "test"

    def test_experiment_tracking(self):
        lab = ConcreteTestLab()
        assert lab._experiment_count == 0
        lab.run_experiment({})
        assert lab._experiment_count == 1
        lab.run_experiment({})
        assert lab._experiment_count == 2

    def test_get_status(self):
        lab = ConcreteTestLab()
        status = lab.get_status()
        assert status["status"] == "running"

    def test_get_capabilities_default(self):
        lab = ConcreteTestLab()
        caps = lab.get_capabilities()
        assert caps["name"] == "ConcreteTestLab"

    def test_uptime(self):
        lab = ConcreteTestLab()
        assert lab.uptime_seconds >= 0

    def test_validate_experiment_default(self):
        lab = ConcreteTestLab()
        assert lab.validate_experiment({"any": "spec"}) is True


# ------------------------------------------------------------------
# register_lab decorator tests
# ------------------------------------------------------------------


class TestRegisterLab:
    def test_register_lab_decorator(self):
        @register_lab(
            name="test_lab_unit",
            category="test",
            description="A test lab",
            is_medical=False,
        )
        class TestLabUnit(BaseLab):
            def run_experiment(self, spec):
                return {}

            def get_status(self):
                return {}

        assert "test_lab_unit" in _LAB_REGISTRY
        assert TestLabUnit._lab_metadata is not None
        assert TestLabUnit._lab_metadata.name == "test_lab_unit"
        assert TestLabUnit._lab_metadata.category == "test"

    def test_register_medical_lab(self):
        @register_lab(
            name="test_medical_unit",
            category="medical",
            description="A test medical lab",
            is_medical=True,
        )
        class TestMedicalUnit(BaseLab):
            def run_experiment(self, spec):
                return {}

            def get_status(self):
                return {}

        assert TestMedicalUnit._lab_metadata.is_medical is True


# ------------------------------------------------------------------
# ConfigManager tests
# ------------------------------------------------------------------


class TestConfigManager:
    def test_default_config(self):
        config = ConfigManager(config_path="/nonexistent/path.yaml")
        assert config.get("app.name") == "QuLabInfinite"

    def test_get_nested(self):
        config = ConfigManager(config_path="/nonexistent/path.yaml")
        assert config.get("api.port") == 8000

    def test_get_missing_key(self):
        config = ConfigManager(config_path="/nonexistent/path.yaml")
        assert config.get("nonexistent.key") is None
        assert config.get("nonexistent.key", "default") == "default"

    def test_get_lab_config(self):
        config = ConfigManager(config_path="/nonexistent/path.yaml")
        qlab = config.get_lab_config("quantum_lab")
        assert qlab.get("default_backend") == "statevector"

    def test_set_value(self):
        config = ConfigManager(config_path="/nonexistent/path.yaml")
        config.set("app.debug", True)
        assert config.get("app.debug") is True


# ------------------------------------------------------------------
# LabRegistry tests
# ------------------------------------------------------------------


class TestLabRegistry:
    def test_manual_registration(self):
        registry = LabRegistry()
        meta = LabMetadata(name="manual_lab", category="test", description="Manual")
        registry.register("manual_lab", ConcreteTestLab, metadata=meta)
        assert "manual_lab" in registry.list_labs()

    def test_get_instance(self):
        registry = LabRegistry()
        meta = LabMetadata(name="inst_lab", category="test")
        registry.register("inst_lab", ConcreteTestLab, metadata=meta)
        lab = registry.get("inst_lab")
        assert isinstance(lab, ConcreteTestLab)

    def test_get_unknown_raises(self):
        registry = LabRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent_lab")

    def test_list_by_category(self):
        registry = LabRegistry()
        for name, cat in [("a", "physics"), ("b", "physics"), ("c", "medical")]:
            meta = LabMetadata(name=name, category=cat)
            registry.register(name, ConcreteTestLab, metadata=meta)
        assert registry.list_by_category("physics") == ["a", "b"]
        assert registry.list_by_category("medical") == ["c"]

    def test_list_medical(self):
        registry = LabRegistry()
        registry.register(
            "med1",
            ConcreteTestLab,
            metadata=LabMetadata(name="med1", category="medical", is_medical=True),
        )
        registry.register(
            "phys1",
            ConcreteTestLab,
            metadata=LabMetadata(name="phys1", category="physics", is_medical=False),
        )
        assert registry.list_medical() == ["med1"]

    def test_count(self):
        registry = LabRegistry()
        assert registry.count == 0
        registry.register("x", ConcreteTestLab, LabMetadata(name="x", category="t"))
        assert registry.count == 1

    def test_summary(self):
        registry = LabRegistry()
        registry.register("a", ConcreteTestLab, LabMetadata(name="a", category="physics"))
        registry.register(
            "b",
            ConcreteTestLab,
            LabMetadata(name="b", category="medical", is_medical=True),
        )
        summary = registry.summary()
        assert summary["total_labs"] == 2
        assert "physics" in summary["categories"]
        assert "medical" in summary["categories"]
        assert "b" in summary["medical_labs"]
