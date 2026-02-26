"""
Small tool shims: calculator, unit converter proxy, safe eval for numeric expressions.
"""
import ast
import math
import operator
from .units import convert, quantity

# Whitelisted operators
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Whitelisted functions
FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
}

# Whitelisted constants
CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}

def _eval_node(node):
    if isinstance(node, ast.Constant):  # Python 3.8+
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Invalid constant type: {type(node.value)}")

    elif isinstance(node, ast.Num):  # Python < 3.8
        return node.n

    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type in OPERATORS:
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            return OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported binary operator: {op_type}")

    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type in OPERATORS:
            operand = _eval_node(node.operand)
            return OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type}")

    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in FUNCTIONS:
                args = [_eval_node(arg) for arg in node.args]
                return FUNCTIONS[func_name](*args)
            raise ValueError(f"Unsupported function: {func_name}")
        raise ValueError("Function calls must be by name")

    elif isinstance(node, ast.Name):
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        raise ValueError(f"Unknown variable: {node.id}")

    raise ValueError(f"Unsupported syntax: {type(node).__name__}")

def calc(expr: str) -> float:
    """
    Safe numeric expression evaluator using AST parsing.
    Supports basic arithmetic, math functions (sqrt, sin, cos, etc.), and constants (pi, e).
    """
    if len(expr) > 500:
        raise ValueError("Expression too long")

    try:
        # Parse the expression into an AST
        tree = ast.parse(expr, mode='eval')

        # Evaluate the AST
        return _eval_node(tree.body)

    except SyntaxError as e:
        raise ValueError(f"Invalid syntax: {e}")
    except Exception as e:
        raise ValueError(f"Calculation error: {e}")
