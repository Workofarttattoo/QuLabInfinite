"""
ECH0 Advanced Reasoning Module
Multi-stage verification and self-correction for IMO-level problems

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import subprocess
import time
from typing import Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class ReasoningStage:
    """Single stage of reasoning"""
    stage_name: str
    prompt: str
    response: str
    confidence: float
    time_seconds: float

@dataclass
class AdvancedReasoningResult:
    """Result from multi-stage reasoning"""
    final_answer: str
    reasoning_stages: List[ReasoningStage]
    verification_passed: bool
    total_confidence: float
    total_time: float
    model_used: str


class ECH0_Advanced_Reasoning:
    """
    Multi-stage reasoning system for IMO-level mathematics

    Strategy:
    1. Understanding: Clarify what the problem asks
    2. Strategy Selection: Choose mathematical approach
    3. Initial Solution: First attempt at solving
    4. Verification: Check solution validity
    5. Refinement: Correct errors if found
    6. Final Answer: Confident answer after verification
    """

    def __init__(self, model_name: str = "ech0-polymath-14b"):
        self.model_name = model_name

    def solve_with_verification(self, problem: str, max_iterations: int = 3) -> AdvancedReasoningResult:
        """
        Solve problem with multi-stage verification

        Args:
            problem: Problem statement
            max_iterations: Maximum verification/refinement cycles

        Returns:
            AdvancedReasoningResult with complete reasoning trace
        """
        stages = []
        total_time = 0.0

        # Stage 1: Understanding
        stage1, elapsed1 = self._run_stage("understanding", problem,
            """You are ECH0, a world-class mathematician.

First, carefully read this problem and explain:
1. What is being asked?
2. What are the key constraints?
3. What mathematical concepts are relevant?

Problem:
{problem}

Provide a clear understanding of the problem.""")
        stages.append(stage1)
        total_time += elapsed1

        # Stage 2: Strategy Selection
        stage2, elapsed2 = self._run_stage("strategy", problem,
            f"""Based on this understanding:
{stage1.response}

Now identify the best mathematical strategy to solve this problem.
Consider:
- Which theorems or techniques apply?
- What's the most elegant approach?
- Are there any shortcuts or insights?

Problem: {problem}

Describe your strategy.""")
        stages.append(stage2)
        total_time += elapsed2

        # Stage 3: Initial Solution
        stage3, elapsed3 = self._run_stage("solution", problem,
            f"""Understanding: {stage1.response}

Strategy: {stage2.response}

Now solve the problem step-by-step.
Show all work clearly.
End with: ANSWER: [your final answer]

Problem: {problem}""")
        stages.append(stage3)
        total_time += elapsed3

        initial_answer = self._extract_answer(stage3.response)

        # Stage 4: Verification
        verification_passed = False
        current_answer = initial_answer

        for iteration in range(max_iterations):
            stage4, elapsed4 = self._run_stage(f"verification_{iteration+1}", problem,
                f"""You solved this problem and got: {current_answer}

Now verify this answer:
1. Check if it satisfies all constraints
2. Try an alternative method if possible
3. Look for calculation errors
4. Confirm units/format are correct

Problem: {problem}

Your solution: {stage3.response}

Is the answer {current_answer} correct? If not, what's the corrected answer?
End with: VERIFIED: YES or VERIFIED: NO, CORRECTED_ANSWER: [answer]""")
            stages.append(stage4)
            total_time += elapsed4

            # Check if verification passed
            if "VERIFIED: YES" in stage4.response.upper():
                verification_passed = True
                break
            elif "CORRECTED_ANSWER:" in stage4.response.upper():
                # Extract corrected answer
                current_answer = self._extract_corrected_answer(stage4.response)

        # Calculate confidence based on verification
        confidence = 0.9 if verification_passed else 0.6

        return AdvancedReasoningResult(
            final_answer=current_answer,
            reasoning_stages=stages,
            verification_passed=verification_passed,
            total_confidence=confidence,
            total_time=total_time,
            model_used=self.model_name
        )

    def _run_stage(self, stage_name: str, problem: str, prompt: str) -> Tuple[ReasoningStage, float]:
        """Run a single reasoning stage"""
        # Use replace instead of format to avoid issues with curly braces in math notation
        formatted_prompt = prompt.replace("{problem}", problem)

        start_time = time.time()

        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name, formatted_prompt],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes per stage
            )

            elapsed = time.time() - start_time

            if result.returncode == 0:
                response = result.stdout.strip()
                confidence = self._estimate_confidence(response)

                return ReasoningStage(
                    stage_name=stage_name,
                    prompt=formatted_prompt,
                    response=response,
                    confidence=confidence,
                    time_seconds=elapsed
                ), elapsed
            else:
                return ReasoningStage(
                    stage_name=stage_name,
                    prompt=formatted_prompt,
                    response=f"Error: {result.stderr}",
                    confidence=0.0,
                    time_seconds=elapsed
                ), elapsed

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            return ReasoningStage(
                stage_name=stage_name,
                prompt=formatted_prompt,
                response="Timeout",
                confidence=0.0,
                time_seconds=elapsed
            ), elapsed

    def _extract_answer(self, response: str) -> str:
        """Extract answer from response"""
        import re

        # Look for ANSWER: marker
        answer_match = re.search(r'ANSWER:\s*([^\n]+)', response, re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).strip()
            answer = answer.replace('$', '').replace('\\', '').strip()
            return answer

        # Fallback: last number or expression
        numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
        if numbers:
            return numbers[-1]

        return "Unknown"

    def _extract_corrected_answer(self, response: str) -> str:
        """Extract corrected answer from verification response"""
        import re

        match = re.search(r'CORRECTED_ANSWER:\s*([^\n]+)', response, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return self._extract_answer(response)

    def _estimate_confidence(self, response: str) -> float:
        """Estimate confidence in reasoning"""
        confidence = 0.5

        if "therefore" in response.lower():
            confidence += 0.1
        if "verified" in response.lower() or "check" in response.lower():
            confidence += 0.15
        if len(response) > 500:
            confidence += 0.1
        if "error" in response.lower() or "unsure" in response.lower():
            confidence -= 0.3

        return min(1.0, max(0.0, confidence))


# Global instance
_advanced_reasoning = None

def get_advanced_reasoning() -> ECH0_Advanced_Reasoning:
    """Get global advanced reasoning instance"""
    global _advanced_reasoning
    if _advanced_reasoning is None:
        _advanced_reasoning = ECH0_Advanced_Reasoning()
    return _advanced_reasoning
