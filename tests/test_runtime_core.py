from pathlib import Path

from core.runtime import ArtifactWriter, Tool, ToolRegistry


def add(x: int, y: int) -> int:
    return x + y


def test_registry_discovers_tool_by_name() -> None:
    registry = ToolRegistry()
    registry.register(Tool(name="math.add", module="math", description="add", func=add))

    discovered = registry.discover("math.add")

    assert discovered["name"] == "math.add"
    assert discovered["module"] == "math"


def test_artifacts_are_reproducible_and_sorted(tmp_path: Path) -> None:
    payload_a = ArtifactWriter.canonical_payload("math.add", {"x": 1, "y": 2}, 3)
    payload_b = ArtifactWriter.canonical_payload("math.add", {"x": 1, "y": 2}, 3)

    assert payload_a == payload_b

    artifact_path = tmp_path / "artifact.json"
    ArtifactWriter.write(artifact_path, payload_a)

    written = artifact_path.read_text(encoding="utf-8")
    assert '"artifact_id"' in written
    assert written.endswith("\n")
