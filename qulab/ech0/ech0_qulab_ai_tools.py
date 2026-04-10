#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

ECH0 QuLab AI Tool Extension
Adds tool-calling capabilities from QuLab AI scaffold to ECH0 interface
"""
import subprocess
from typing import Dict, Any, List
from chemistry_lab.qulab_ai_integration import analyze_molecule_with_provenance
from materials_lab.qulab_ai_integration import analyze_structure_with_provenance
from frequency_lab.qulab_ai_integration import encode_spectrum_array
from qulab_ai.tools import calc
from qulab_ai.units import convert, HAVE_PINT
from qulab_ai.answer_mode import build_answer
from qulab_ai.provenance import citation_block

class ECH0_ToolRegistry:
    """Registry of tools available to ECH0 14B"""

    @staticmethod
    def calc_tool(expr: str) -> Dict[str, Any]:
        """
        Calculator tool for mathematical expressions

        Args:
            expr: Mathematical expression (e.g., "2 + 2", "10 * 5")

        Returns:
            Dict with result and provenance
        """
        try:
            result = calc(expr)
            return build_answer(
                payload={"expression": expr, "result": result},
                citations=[citation_block(source="QuLab AI Calculator")]
            )
        except Exception as e:
            return {"error": str(e), "expression": expr}

    @staticmethod
    def units_tool(value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
        """
        Unit conversion tool

        Args:
            value: Numerical value
            from_unit: Source unit (e.g., "eV", "cm", "kg")
            to_unit: Target unit (e.g., "J", "m", "g")

        Returns:
            Dict with converted value and provenance
        """
        try:
            result = convert(value, from_unit, to_unit)
            return build_answer(
                payload={
                    "input": {"value": value, "unit": from_unit},
                    "output": {"value": result, "unit": to_unit},
                    "conversion": f"{value} {from_unit} = {result} {to_unit}"
                },
                citations=[citation_block(
                    source="QuLab AI Units",
                    backend="pint" if HAVE_PINT else "minimal"
                )]
            )
        except Exception as e:
            return {"error": str(e), "input": f"{value} {from_unit} to {to_unit}"}

    @staticmethod
    def parse_smiles_tool(smiles: str) -> Dict[str, Any]:
        """
        SMILES molecule parser tool

        Args:
            smiles: SMILES notation string

        Returns:
            Dict with parsed molecule and provenance
        """
        try:
            return analyze_molecule_with_provenance(smiles)
        except Exception as e:
            return {"error": str(e), "smiles": smiles}

    @staticmethod
    def encode_spectrum_tool(x: List[float], y: List[float], caption: str = "") -> Dict[str, Any]:
        """
        Spectrum encoder tool for ML

        Args:
            x: X-axis values
            y: Y-axis values
            caption: Optional caption

        Returns:
            Dict with encoding and provenance
        """
        try:
            result = encode_spectrum_array(x, y, caption)
            return build_answer(
                payload=result,
                citations=[citation_block(source="QuLab AI Spectra Encoder")]
            )
        except Exception as e:
            return {"error": str(e)}

    # Tool registry mapping
    TOOLS = {
        "calc": calc_tool.__func__,
        "units": units_tool.__func__,
        "parse_smiles": parse_smiles_tool.__func__,
        "encode_spectrum": encode_spectrum_tool.__func__
    }

def call_ech0_with_tools(prompt: str, model: str = "ech0-qulab-14b", timeout: int = 60) -> Dict[str, Any]:
    """
    Call ECH0 with tool-calling support

    Args:
        prompt: User prompt
        model: ECH0 model to use (default: ech0-qulab-14b finetuned)
        timeout: Timeout in seconds

    Returns:
        Dict with response and any tool calls executed
    """
    # Format prompt with tool instructions
    tool_prompt = f"""You have access to these tools:
- calc(expr): Evaluate math expressions
- units(value, from_unit, to_unit): Convert units
- parse_smiles(smiles): Parse molecules
- encode_spectrum(x, y, caption): Encode spectra

When you need a tool, respond with: #TOOL: <name> <args>

User request:
{prompt}
"""

    # Call ECH0
    try:
        result = subprocess.run(
            ["ollama", "run", model, tool_prompt],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        response = result.stdout.strip()

        # Parse tool calls from response
        tool_calls = []
        lines = response.split('\n')
        for line in lines:
            if line.strip().startswith('#TOOL:'):
                tool_calls.append(line.strip())

        return {
            "response": response,
            "tool_calls": tool_calls,
            "model": model,
            "success": True
        }

    except subprocess.TimeoutExpired:
        return {
            "error": "ECH0 timeout",
            "timeout": timeout,
            "success": False
        }
    except Exception as e:
        return {
            "error": str(e),
            "success": False
        }

def execute_tool_call(tool_call: str) -> Dict[str, Any]:
    """
    Execute a tool call from ECH0 response

    Args:
        tool_call: Tool call string (e.g., "#TOOL: calc 2 + 2")

    Returns:
        Dict with tool execution result
    """
    # Parse tool call
    if not tool_call.startswith('#TOOL:'):
        return {"error": "Invalid tool call format"}

    parts = tool_call[7:].strip().split(maxsplit=1)
    if len(parts) < 1:
        return {"error": "No tool name specified"}

    tool_name = parts[0]
    tool_args = parts[1] if len(parts) > 1 else ""

    # Get tool from registry
    tool_fn = ECH0_ToolRegistry.TOOLS.get(tool_name)
    if not tool_fn:
        return {"error": f"Unknown tool: {tool_name}"}

    # Execute tool (simplified - in production, parse args properly)
    try:
        if tool_name == "calc":
            return tool_fn(tool_args)
        elif tool_name == "parse_smiles":
            return tool_fn(tool_args.strip())
        # Add more tool-specific parsing as needed
        else:
            return {"error": f"Tool execution not implemented: {tool_name}"}
    except Exception as e:
        return {"error": f"Tool execution failed: {e}"}

def ech0_interactive_session(model: str = "ech0-qulab-14b"):
    """
    Interactive session with ECH0 + QuLab AI tools

    Args:
        model: ECH0 model to use
    """
    print("=" * 70)
    print(f"ECH0 Interactive Session with QuLab AI Tools")
    print(f"Model: {model}")
    print("=" * 70)
    print("Available tools: calc, units, parse_smiles, encode_spectrum")
    print("Type 'exit' to quit\n")

    while True:
        try:
            prompt = input("You: ")
            if prompt.lower() in ['exit', 'quit']:
                break

            result = call_ech0_with_tools(prompt, model)

            if result["success"]:
                print(f"\nECH0: {result['response']}")

                if result["tool_calls"]:
                    print(f"\nTool calls detected: {len(result['tool_calls'])}")
                    for tool_call in result["tool_calls"]:
                        print(f"  Executing: {tool_call}")
                        tool_result = execute_tool_call(tool_call)
                        print(f"  Result: {tool_result}")
            else:
                print(f"\nError: {result['error']}")

            print()

        except KeyboardInterrupt:
            break

    print("\nSession ended.")

# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        # Interactive mode
        ech0_interactive_session()
    else:
        # Demo mode
        print("=" * 70)
        print("ECH0 + QuLab AI Tools Demo")
        print("=" * 70)

        # Test calculator
        print("\n[1] Testing calculator tool...")
        result = ECH0_ToolRegistry.calc_tool("2 + 2")
        print(f"Result: {result['result']['result']}")

        # Test units
        print("\n[2] Testing units tool...")
        result = ECH0_ToolRegistry.units_tool(100, "cm", "m")
        print(f"Conversion: {result['result']['conversion']}")

        # Test SMILES parser
        print("\n[3] Testing SMILES parser...")
        result = ECH0_ToolRegistry.parse_smiles_tool("CCO")
        print(f"Molecule: {result['result']['canonical_smiles']}")
        print(f"Atoms: {result['result']['n_atoms']}")

        # Test spectrum encoder
        print("\n[4] Testing spectrum encoder...")
        x = [1, 2, 3, 4, 5]
        y = [0.1, 0.9, 0.2, 0.8, 0.1]
        result = ECH0_ToolRegistry.encode_spectrum_tool(x, y, "test spectrum")
        print(f"Peaks detected: {result['result']['ml_encoding']['peaks']}")

        print("\n" + "=" * 70)
        print("All tools working! Run with 'interactive' arg for chat mode.")
        print("=" * 70)
