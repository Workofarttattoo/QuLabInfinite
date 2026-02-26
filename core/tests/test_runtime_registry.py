from core.runtime import RuntimeRegistry


class EchoTool:
    name = "echo"

    def describe(self):
        return {"name": "Echo", "capabilities": ["mirror"]}

    def run(self, payload):
        return {"echo": payload.get("value", None)}


def test_tool_discovery_by_name():
    runtime = RuntimeRegistry()
    runtime.register(EchoTool())

    assert "echo" in runtime.list_tools()


def test_reproducible_json_artifact_ordering():
    runtime = RuntimeRegistry()
    runtime.register(EchoTool())

    artifact_a = runtime.run("echo", {"value": 42, "z": 0}).to_json()
    artifact_b = runtime.run("echo", {"z": 0, "value": 42}).to_json()

    assert artifact_a == artifact_b
    assert artifact_a == '{"payload":{"value":42,"z":0},"result":{"echo":42},"tool":"echo"}'
