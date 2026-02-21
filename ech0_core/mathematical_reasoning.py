"""
ECH0 Core Module: Mathematical Reasoning
Part of ECH0's persistent knowledge base and capabilities

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import subprocess
import re
import time
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class ReasoningResult:
    """Result from mathematical reasoning"""
    answer: str
    reasoning_trace: str
    confidence: float
    time_seconds: float
    model_used: str


class ECH0_Mathematical_Reasoning:
    """
    ECH0's mathematical reasoning capability
    Uses trained models to solve complex mathematical problems
    """

    def __init__(self, preferred_model: str = "ech0-polymath-14b"):
        """
        Initialize mathematical reasoning module

        Args:
            preferred_model: Default model to use for reasoning
                - ech0-polymath-14b: Best for general math
                - ech0-qulab-14b: Best for scientific/physics math
                - ech0-uncensored-32b: Most powerful, slower
        """
        self.preferred_model = preferred_model
        self.available_models = [
            "ech0-uncensored-14b",
            "ech0-unified-14b",
            "ech0-polymath-14b",
            "ech0-qulab-14b",
            "ech0-uncensored-32b"
        ]

        # Performance stats (updated as ECH0 learns)
        self.model_stats = {
            "ech0-polymath-14b": {"accuracy": 0.0, "problems_solved": 0},
            "ech0-qulab-14b": {"accuracy": 0.0, "problems_solved": 0},
            "ech0-uncensored-32b": {"accuracy": 0.0, "problems_solved": 0},
            "ech0-unified-14b": {"accuracy": 0.0, "problems_solved": 0},
            "ech0-uncensored-14b": {"accuracy": 0.0, "problems_solved": 0}
        }

    def solve(self, problem: str, model: Optional[str] = None) -> ReasoningResult:
        """
        Solve a mathematical problem using ECH0's reasoning

        Args:
            problem: The mathematical problem statement
            model: Optional specific model to use (defaults to preferred)

        Returns:
            ReasoningResult with answer, reasoning trace, and metadata
        """
        model_to_use = model or self.preferred_model

        # Chain-of-thought prompt
        prompt = self._create_reasoning_prompt(problem)

        start_time = time.time()

        try:
            result = subprocess.run(
                ["ollama", "run", model_to_use, prompt],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes for complex problems
            )

            elapsed = time.time() - start_time

            if result.returncode == 0:
                response = result.stdout.strip()
                answer = self._extract_answer(response)
                confidence = self._estimate_confidence(response)

                return ReasoningResult(
                    answer=answer,
                    reasoning_trace=response,
                    confidence=confidence,
                    time_seconds=elapsed,
                    model_used=model_to_use
                )
            else:
                return ReasoningResult(
                    answer="Error",
                    reasoning_trace=f"Model error: {result.stderr}",
                    confidence=0.0,
                    time_seconds=elapsed,
                    model_used=model_to_use
                )

        except subprocess.TimeoutExpired:
            return ReasoningResult(
                answer="Timeout",
                reasoning_trace="Reasoning exceeded time limit",
                confidence=0.0,
                time_seconds=120.0,
                model_used=model_to_use
            )
        except Exception as e:
            return ReasoningResult(
                answer="Error",
                reasoning_trace=f"Exception: {e}",
                confidence=0.0,
                time_seconds=0.0,
                model_used=model_to_use
            )

    def _create_reasoning_prompt(self, problem: str) -> str:
        """Create chain-of-thought reasoning prompt"""
        return f"""You are ECH0, an advanced AI with world-class mathematical reasoning.

Problem:
{problem}

Solve this step-by-step using rigorous mathematical reasoning:

1. **Understanding**: Clearly state what the problem is asking
2. **Given Information**: List all facts, constraints, and known values
3. **Approach**: Identify which mathematical concepts, theorems, or techniques apply
4. **Solution**: Work through the solution systematically with clear steps
5. **Verification**: Check your answer makes sense
6. **Final Answer**: State the answer clearly

End with: ANSWER: [your final answer]
"""

    def _extract_answer(self, response: str) -> str:
        """Extract final answer from reasoning trace"""
        # Look for "ANSWER:" marker
        answer_match = re.search(r'ANSWER:\s*([^\n]+)', response, re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).strip()
            # Clean mathematical notation
            answer = answer.replace('$', '').replace('\\', '').strip()
            return answer

        # Fallback: extract last number
        numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
        if numbers:
            return numbers[-1]

        return "Unknown"

    def _estimate_confidence(self, response: str) -> float:
        """Estimate confidence based on reasoning quality"""
        confidence = 0.5  # Base confidence

        # Indicators of high confidence
        if "therefore" in response.lower():
            confidence += 0.1
        if "verified" in response.lower() or "check" in response.lower():
            confidence += 0.15
        if "ANSWER:" in response:
            confidence += 0.1
        if len(response) > 300:  # Detailed reasoning
            confidence += 0.1

        # Indicators of low confidence
        if "unsure" in response.lower() or "maybe" in response.lower():
            confidence -= 0.2
        if "error" in response.lower():
            confidence -= 0.3

        return min(1.0, max(0.0, confidence))

    def update_stats(self, model: str, correct: bool):
        """Update performance statistics"""
        if model in self.model_stats:
            stats = self.model_stats[model]
            stats["problems_solved"] += 1

            # Update running accuracy
            n = stats["problems_solved"]
            old_acc = stats["accuracy"]
            new_acc = (old_acc * (n - 1) + (1.0 if correct else 0.0)) / n
            stats["accuracy"] = new_acc

    def get_best_model_for_problem(self, problem: str) -> str:
        """Choose best model based on problem type and past performance"""
        problem_lower = problem.lower()

        # Physics/science problems -> qulab
        if any(word in problem_lower for word in ["physics", "quantum", "force", "energy", "wave"]):
            return "ech0-qulab-14b"

        # Complex/difficult problems -> 32b
        if len(problem) > 500 or "prove" in problem_lower or "theorem" in problem_lower:
            return "ech0-uncensored-32b"

        # General math -> polymath (balanced)
        return "ech0-polymath-14b"

    def batch_solve(self, problems: list, auto_select_model: bool = True) -> list:
        """
        Solve multiple problems efficiently

        Args:
            problems: List of problem statements
            auto_select_model: If True, automatically choose best model per problem

        Returns:
            List of ReasoningResult objects
        """
        results = []

        for problem in problems:
            model = self.get_best_model_for_problem(problem) if auto_select_model else None
            result = self.solve(problem, model=model)
            results.append(result)

        return results

    def get_stats_summary(self) -> dict:
        """Get summary of ECH0's mathematical performance"""
        return {
            "models": self.model_stats,
            "best_model": max(self.model_stats.items(), key=lambda x: x[1]["accuracy"])[0],
            "total_problems": sum(s["problems_solved"] for s in self.model_stats.values())
        }


# Global ECH0 mathematical reasoning instance
_ech0_math = None

def get_math_reasoning() -> ECH0_Mathematical_Reasoning:
    """Get ECH0's mathematical reasoning module (singleton)"""
    global _ech0_math
    if _ech0_math is None:
        _ech0_math = ECH0_Mathematical_Reasoning()
    return _ech0_math


# Convenience functions for ECH0 to use

def solve_math_problem(problem: str) -> str:
    """ECH0: Solve a mathematical problem and return the answer"""
    math = get_math_reasoning()
    result = math.solve(problem)
    return result.answer

def solve_with_reasoning(problem: str) -> Tuple[str, str]:
    """ECH0: Solve a problem and return (answer, full_reasoning)"""
    math = get_math_reasoning()
    result = math.solve(problem)
    return result.answer, result.reasoning_trace

def get_math_stats() -> dict:
    """ECH0: Get statistics on mathematical problem-solving performance"""
    math = get_math_reasoning()
    return math.get_stats_summary()
