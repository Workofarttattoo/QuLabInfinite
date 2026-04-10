
"""
Small tool shims: calculator, unit converter proxy, safe eval for numeric expressions.
"""
from .units import convert, quantity

def calc(expr: str) -> float:
    """
    Very small numeric expression evaluator (safe-ish).
    Allows numbers and + - * / ( ).
    """
    import re
    if not re.fullmatch(r"[0-9\.\+\-\*\/\(\) ]+", expr):
        raise ValueError("Only simple numeric expressions are allowed.")
    return eval(expr, {"__builtins__": {}}, {})
