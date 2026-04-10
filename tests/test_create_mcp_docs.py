import pytest
import sys
from pathlib import Path

# Add project root to path to allow importing create_mcp_docs
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from create_mcp_docs import create_lab_documentation

def test_create_lab_documentation_basic():
    """Test standard HTML generation with fully populated tool"""
    lab_name = "Test Lab"
    tools = [
        {
            "id": "test_tool_1",
            "name": "Full Tool",
            "status": "working",
            "description": "A very nice tool",
            "parameters": [
                {"name": "param1", "type": "string", "required": True, "description": "Param 1 desc"}
            ],
            "example_params": {"param1": "value1"},
            "example_response": {"result": "success"},
            "scientific_basis": "Science works",
            "references": ["Ref 1"]
        }
    ]
    html = create_lab_documentation(lab_name, tools)

    # Check title and lab name
    assert f"<title>{lab_name} - MCP Tool Documentation | QuLab Infinite</title>" in html
    assert f"<h1>{lab_name} - MCP Tool Documentation</h1>" in html

    # Check tool contents
    assert "Full Tool" in html
    assert "test_tool_1" in html
    assert "working" in html
    assert "✅ Working" in html
    assert "A very nice tool" in html
    assert "param1" in html
    assert "value1" in html
    assert "result" in html
    assert "Science works" in html
    assert "Ref 1" in html

def test_create_lab_documentation_missing_optional_fields():
    """Test generating documentation when optional fields are missing"""
    tools = [
        {
            "id": "minimal_tool",
            "name": "Minimal Tool",
            "description": "Tool without optional fields"
            # Missing status, parameters, example_params, example_response, scientific_basis, references
        }
    ]
    html = create_lab_documentation("Minimal Lab", tools)

    assert "Minimal Lab" in html
    assert "Minimal Tool" in html
    assert "Tool without optional fields" in html

    # Check defaults
    assert "placeholder" in html  # Default status logic
    assert "❌ Placeholder" in html
    assert "Under development" in html # Default scientific basis

def test_create_lab_documentation_statuses():
    """Test the different possible statuses"""
    tools = [
        {"id": "t1", "name": "T1", "description": "D1", "status": "working"},
        {"id": "t2", "name": "T2", "description": "D2", "status": "partial"},
        {"id": "t3", "name": "T3", "description": "D3", "status": "placeholder"},
        {"id": "t4", "name": "T4", "description": "D4", "status": "unknown_status"}
    ]
    html = create_lab_documentation("Status Lab", tools)

    # Working
    assert "status working" in html
    assert "✅ Working" in html

    # Partial
    assert "status partial" in html
    assert "🔶 Partial" in html

    # Placeholder
    assert "status placeholder" in html
    assert "❌ Placeholder" in html

    # Unknown gets placeholder class but displays '❓ Unknown'
    assert "status unknown_status" in html
    assert "❓ Unknown" in html

def test_create_lab_documentation_empty_tools():
    """Test HTML generation when no tools are provided"""
    html = create_lab_documentation("Empty Lab", [])

    assert "Empty Lab" in html
    assert "Available MCP Tools" in html
    assert "Python Client Example" in html # Global footer/example code should still be there

    # Shouldn't contain any tool markup
    assert "<div class=\"tool\">" not in html
