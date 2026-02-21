"""
ECH0 Symbolic Mathematics Module
Exact symbolic computation and verification using SymPy

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import re
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

try:
    import sympy as sp
    from sympy import symbols, simplify, solve, integrate, diff, limit, series
    from sympy.parsing.sympy_parser import parse_expr
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

@dataclass
class SymbolicResult:
    """Result from symbolic computation"""
    result: str
    latex: str
    simplified: bool
    verified: bool
    method: str  # 'solve', 'integrate', 'differentiate', etc.


class ECH0_Symbolic_Math:
    """
    Symbolic mathematics for exact computation and verification

    Uses SymPy for:
    - Solving equations exactly
    - Symbolic integration/differentiation
    - Expression simplification
    - Algebraic manipulation
    - Answer verification
    """

    def __init__(self):
        if not SYMPY_AVAILABLE:
            raise ImportError("SymPy not installed. Run: pip install sympy")

        # Common variables
        self.x, self.y, self.z = symbols('x y z')
        self.a, self.b, self.c = symbols('a b c')
        self.n = symbols('n', integer=True)

    def verify_answer(self, llm_answer: str, problem: str) -> Tuple[bool, str]:
        """
        Verify an LLM's answer using symbolic computation

        Args:
            llm_answer: Answer from LLM
            problem: Original problem statement

        Returns:
            (is_correct, explanation)
        """
        # Extract mathematical expressions from problem
        problem_type = self._detect_problem_type(problem)

        if problem_type == "equation":
            return self._verify_equation_solution(llm_answer, problem)
        elif problem_type == "derivative":
            return self._verify_derivative(llm_answer, problem)
        elif problem_type == "integral":
            return self._verify_integral(llm_answer, problem)
        elif problem_type == "simplification":
            return self._verify_simplification(llm_answer, problem)
        else:
            # Can't verify this type symbolically
            return False, f"Cannot verify {problem_type} symbolically"

    def solve_equation(self, equation_str: str) -> SymbolicResult:
        """
        Solve equation symbolically

        Example: "x^2 - 4 = 0" -> x = -2, x = 2
        """
        try:
            # Parse equation
            if '=' in equation_str:
                left, right = equation_str.split('=')
                eq = parse_expr(left) - parse_expr(right)
            else:
                eq = parse_expr(equation_str)

            # Solve for x (default)
            solutions = solve(eq, self.x)

            result_str = ', '.join(str(sol) for sol in solutions)
            latex_str = ', '.join(sp.latex(sol) for sol in solutions)

            return SymbolicResult(
                result=result_str,
                latex=latex_str,
                simplified=True,
                verified=True,
                method='solve'
            )

        except Exception as e:
            return SymbolicResult(
                result=f"Error: {e}",
                latex="",
                simplified=False,
                verified=False,
                method='solve'
            )

    def differentiate(self, expr_str: str, var: str = 'x') -> SymbolicResult:
        """
        Compute derivative symbolically

        Example: "x^3 + 2*x^2 - 5*x + 1" -> "3*x^2 + 4*x - 5"
        """
        try:
            expr = parse_expr(expr_str)
            var_symbol = symbols(var)
            derivative = diff(expr, var_symbol)
            simplified = simplify(derivative)

            return SymbolicResult(
                result=str(simplified),
                latex=sp.latex(simplified),
                simplified=True,
                verified=True,
                method='differentiate'
            )

        except Exception as e:
            return SymbolicResult(
                result=f"Error: {e}",
                latex="",
                simplified=False,
                verified=False,
                method='differentiate'
            )

    def integrate_expr(self, expr_str: str, var: str = 'x', limits: Optional[Tuple] = None) -> SymbolicResult:
        """
        Compute integral symbolically

        Example: "x^2" -> "x^3/3"
        """
        try:
            expr = parse_expr(expr_str)
            var_symbol = symbols(var)

            if limits:
                # Definite integral
                result = integrate(expr, (var_symbol, limits[0], limits[1]))
            else:
                # Indefinite integral
                result = integrate(expr, var_symbol)

            simplified = simplify(result)

            return SymbolicResult(
                result=str(simplified),
                latex=sp.latex(simplified),
                simplified=True,
                verified=True,
                method='integrate'
            )

        except Exception as e:
            return SymbolicResult(
                result=f"Error: {e}",
                latex="",
                simplified=False,
                verified=False,
                method='integrate'
            )

    def simplify_expr(self, expr_str: str) -> SymbolicResult:
        """Simplify expression symbolically"""
        try:
            expr = parse_expr(expr_str)
            simplified = simplify(expr)

            return SymbolicResult(
                result=str(simplified),
                latex=sp.latex(simplified),
                simplified=True,
                verified=True,
                method='simplify'
            )

        except Exception as e:
            return SymbolicResult(
                result=f"Error: {e}",
                latex="",
                simplified=False,
                verified=False,
                method='simplify'
            )

    def _detect_problem_type(self, problem: str) -> str:
        """Detect what type of mathematical problem this is"""
        problem_lower = problem.lower()

        if 'derivative' in problem_lower or 'd/dx' in problem_lower or 'differentiate' in problem_lower:
            return 'derivative'
        elif 'integral' in problem_lower or '∫' in problem or 'integrate' in problem_lower:
            return 'integral'
        elif 'solve' in problem_lower or '=' in problem:
            return 'equation'
        elif 'simplify' in problem_lower:
            return 'simplification'
        else:
            return 'unknown'

    def _verify_equation_solution(self, llm_answer: str, problem: str) -> Tuple[bool, str]:
        """Verify solution to an equation"""
        try:
            # Extract equation from problem
            equation_match = re.search(r'([^=]+=[^=]+)', problem)
            if not equation_match:
                return False, "Could not extract equation from problem"

            equation = equation_match.group(1)

            # Solve symbolically
            symbolic_result = self.solve_equation(equation)

            # Compare with LLM answer
            if symbolic_result.result.lower() in llm_answer.lower():
                return True, f"Verified: {symbolic_result.result}"
            else:
                return False, f"Expected {symbolic_result.result}, got {llm_answer}"

        except Exception as e:
            return False, f"Verification error: {e}"

    def _verify_derivative(self, llm_answer: str, problem: str) -> Tuple[bool, str]:
        """Verify derivative computation"""
        try:
            # Extract expression to differentiate
            # Look for patterns like "x^3 + 2x^2"
            expr_match = re.search(r'[a-z]\^?\d+[\s\+\-\*/\d\^a-z]*', problem)
            if not expr_match:
                return False, "Could not extract expression"

            expr = expr_match.group(0)

            # Compute derivative
            symbolic_result = self.differentiate(expr)

            # Normalize both for comparison
            llm_normalized = llm_answer.replace(' ', '').replace('*', '').lower()
            sym_normalized = symbolic_result.result.replace(' ', '').replace('*', '').lower()

            if sym_normalized in llm_normalized:
                return True, f"Verified: {symbolic_result.result}"
            else:
                return False, f"Expected {symbolic_result.result}, got {llm_answer}"

        except Exception as e:
            return False, f"Verification error: {e}"

    def _verify_integral(self, llm_answer: str, problem: str) -> Tuple[bool, str]:
        """Verify integral computation"""
        try:
            # Extract expression to integrate
            expr_match = re.search(r'[a-z]\^?\d+[\s\+\-\*/\d\^a-z]*', problem)
            if not expr_match:
                return False, "Could not extract expression"

            expr = expr_match.group(0)

            # Compute integral
            symbolic_result = self.integrate_expr(expr)

            # Normalize for comparison
            llm_normalized = llm_answer.replace(' ', '').replace('*', '').lower()
            sym_normalized = symbolic_result.result.replace(' ', '').replace('*', '').lower()

            # Allow for constant of integration
            if sym_normalized in llm_normalized or (sym_normalized + '+c') in llm_normalized:
                return True, f"Verified: {symbolic_result.result} + C"
            else:
                return False, f"Expected {symbolic_result.result}, got {llm_answer}"

        except Exception as e:
            return False, f"Verification error: {e}"

    def _verify_simplification(self, llm_answer: str, problem: str) -> Tuple[bool, str]:
        """Verify expression simplification"""
        try:
            # Extract expression
            expr_match = re.search(r'[a-z]\^?\d+[\s\+\-\*/\d\^a-z\(\)]*', problem)
            if not expr_match:
                return False, "Could not extract expression"

            expr = expr_match.group(0)

            # Simplify
            symbolic_result = self.simplify_expr(expr)

            # Compare
            llm_normalized = llm_answer.replace(' ', '').lower()
            sym_normalized = symbolic_result.result.replace(' ', '').lower()

            if sym_normalized == llm_normalized:
                return True, f"Verified: {symbolic_result.result}"
            else:
                return False, f"Expected {symbolic_result.result}, got {llm_answer}"

        except Exception as e:
            return False, f"Verification error: {e}"


# Global instance
_symbolic_math = None

def get_symbolic_math() -> ECH0_Symbolic_Math:
    """Get global symbolic math instance"""
    global _symbolic_math
    if _symbolic_math is None:
        _symbolic_math = ECH0_Symbolic_Math()
    return _symbolic_math


def verify_with_sympy(llm_answer: str, problem: str) -> Tuple[bool, str]:
    """Convenience function to verify answer with SymPy"""
    if not SYMPY_AVAILABLE:
        return False, "SymPy not available"

    sym = get_symbolic_math()
    return sym.verify_answer(llm_answer, problem)
