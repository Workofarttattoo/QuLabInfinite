"""
Lightweight eval harness that exercises Master MCP tools against canned tasks.
Extend `TASKS` per lab/tool to grow coverage. Intended for smoke/regression use.
"""

from __future__ import annotations

from typing import Any, Dict, List

from scripts.master_mcp_clients import invoke


class EvalResult:
    def __init__(self, name: str, passed: bool, detail: str = ""):
        self.name = name
        self.passed = passed
        self.detail = detail

    def to_dict(self):
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


TASKS: List[Dict[str, Any]] = [
    {
        "name": "calc_basic",
        "tool": "ai.calc",
        "args": {"expression": "2+3*4"},
        "check": lambda r: str(r.get("result") or r).strip() in {"14", "14.0"},
    },
    {
        "name": "physics_fe",
        "tool": "physics.get_element_properties",
        "args": {"element": "Fe"},
        "check": lambda r: bool(r),
    },
]


def run_tasks() -> List[EvalResult]:
    results: List[EvalResult] = []
    for task in TASKS:
        try:
            resp = invoke(task["tool"], task["args"])
            ok = task["check"](resp.get("result", resp))
            results.append(EvalResult(task["name"], ok, "" if ok else f"Got {resp}"))
        except Exception as exc:  # pragma: no cover
            results.append(EvalResult(task["name"], False, str(exc)))
    return results


def main():
    outcomes = run_tasks()
    passed = sum(r.passed for r in outcomes)
    print(f"[eval] {passed}/{len(outcomes)} passed")
    for r in outcomes:
        status = "PASS" if r.passed else "FAIL"
        print(f"[eval] {status} - {r.name} {r.detail}")


if __name__ == "__main__":
    main()
