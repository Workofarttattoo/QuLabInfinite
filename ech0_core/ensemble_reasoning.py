"""
ECH0 Ensemble Reasoning Module
Combines multiple 14B models for consensus-based answers (excludes 32B to avoid performance issues)

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import subprocess
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import Counter

@dataclass
class ModelVote:
    """Single model's vote on an answer"""
    model_name: str
    answer: str
    reasoning: str
    confidence: float
    time_seconds: float

@dataclass
class EnsembleResult:
    """Result from ensemble voting"""
    consensus_answer: str
    votes: List[ModelVote]
    agreement_score: float  # 0.0-1.0, how many models agreed
    total_confidence: float
    total_time: float
    winning_models: List[str]  # Models that voted for consensus answer


class ECH0_Ensemble_Reasoning:
    """
    Ensemble reasoning using multiple 14B models

    Strategy: Run problem through 3 different 14B models and vote on answer
    - ech0-polymath-14b: Mathematics specialist
    - ech0-qulab-14b: Scientific computing specialist
    - ech0-unified-14b: General multi-domain model

    NOTE: Excludes ech0-uncensored-32b to avoid grinding laptop to halt
    """

    def __init__(self):
        # Only use 14B models for ensemble voting (lighter on resources)
        self.ensemble_models = [
            "ech0-polymath-14b",   # Best for pure math
            "ech0-qulab-14b",      # Best for scientific/physics math
            "ech0-unified-14b"      # Balanced general reasoning
        ]

    def solve_with_ensemble(self, problem: str, timeout_per_model: int = 300) -> EnsembleResult:
        """
        Solve problem using ensemble voting

        Args:
            problem: Problem statement
            timeout_per_model: Timeout in seconds for each model (default 300s = 5min)

        Returns:
            EnsembleResult with consensus answer and vote breakdown
        """
        votes = []
        total_time = 0.0

        print(f"\n[ENSEMBLE VOTING]")
        print(f"Running problem through {len(self.ensemble_models)} models...")
        print()

        # Get vote from each model
        for i, model in enumerate(self.ensemble_models, 1):
            print(f"  [{i}/{len(self.ensemble_models)}] Querying {model}...")

            vote = self._get_model_vote(model, problem, timeout_per_model)
            votes.append(vote)
            total_time += vote.time_seconds

            print(f"      Answer: {vote.answer}")
            print(f"      Confidence: {vote.confidence:.0%}")
            print(f"      Time: {vote.time_seconds:.1f}s")
            print()

        # Count votes and determine consensus
        consensus_answer, agreement_score, winning_models = self._determine_consensus(votes)

        # Calculate average confidence of winning models
        winning_votes = [v for v in votes if v.answer == consensus_answer]
        total_confidence = sum(v.confidence for v in winning_votes) / len(winning_votes) if winning_votes else 0.0

        return EnsembleResult(
            consensus_answer=consensus_answer,
            votes=votes,
            agreement_score=agreement_score,
            total_confidence=total_confidence,
            total_time=total_time,
            winning_models=winning_models
        )

    def _get_model_vote(self, model: str, problem: str, timeout: int) -> ModelVote:
        """Get a single model's vote"""
        prompt = f"""You are ECH0, a world-class mathematician solving IMO problems.

Problem:
{problem}

Solve step-by-step with clear reasoning.
End with: ANSWER: [your final answer]
"""

        start_time = time.time()

        try:
            result = subprocess.run(
                ["ollama", "run", model, prompt],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            elapsed = time.time() - start_time

            if result.returncode == 0:
                response = result.stdout.strip()
                answer = self._extract_answer(response)
                confidence = self._estimate_confidence(response)

                return ModelVote(
                    model_name=model,
                    answer=answer,
                    reasoning=response,
                    confidence=confidence,
                    time_seconds=elapsed
                )
            else:
                return ModelVote(
                    model_name=model,
                    answer="Error",
                    reasoning=f"Error: {result.stderr}",
                    confidence=0.0,
                    time_seconds=elapsed
                )

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            return ModelVote(
                model_name=model,
                answer="Timeout",
                reasoning="Timeout expired",
                confidence=0.0,
                time_seconds=elapsed
            )
        except Exception as e:
            elapsed = time.time() - start_time
            return ModelVote(
                model_name=model,
                answer="Error",
                reasoning=f"Exception: {e}",
                confidence=0.0,
                time_seconds=elapsed
            )

    def _extract_answer(self, response: str) -> str:
        """Extract answer from model response"""
        import re

        # Look for ANSWER: marker
        answer_match = re.search(r'ANSWER:\\s*([^\\n]+)', response, re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).strip()
            answer = answer.replace('$', '').replace('\\\\', '').strip()
            return answer

        # Fallback: last number or expression
        numbers = re.findall(r'-?\\d+(?:\\.\\d+)?', response)
        if numbers:
            return numbers[-1]

        return "Unknown"

    def _estimate_confidence(self, response: str) -> float:
        """Estimate confidence based on reasoning quality"""
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

    def _determine_consensus(self, votes: List[ModelVote]) -> Tuple[str, float, List[str]]:
        """
        Determine consensus answer from votes

        Returns:
            (consensus_answer, agreement_score, winning_models)
        """
        # Count votes (excluding errors and timeouts)
        valid_votes = [v for v in votes if v.answer not in ["Error", "Timeout", "Unknown"]]

        if not valid_votes:
            # All models failed - return most common failure
            answer_counts = Counter(v.answer for v in votes)
            most_common = answer_counts.most_common(1)[0][0]
            return most_common, 0.0, []

        # Normalize answers for comparison (lowercase, strip whitespace)
        normalized_answers = {}
        for vote in valid_votes:
            normalized = vote.answer.lower().strip()
            if normalized not in normalized_answers:
                normalized_answers[normalized] = []
            normalized_answers[normalized].append(vote)

        # Find most common answer
        most_common_normalized = max(normalized_answers.items(), key=lambda x: len(x[1]))
        consensus_votes = most_common_normalized[1]

        # Use original (non-normalized) answer from first vote
        consensus_answer = consensus_votes[0].answer

        # Calculate agreement score
        agreement_score = len(consensus_votes) / len(votes)

        # Get winning model names
        winning_models = [v.model_name for v in consensus_votes]

        return consensus_answer, agreement_score, winning_models


# Global instance
_ensemble_reasoning = None

def get_ensemble_reasoning() -> ECH0_Ensemble_Reasoning:
    """Get global ensemble reasoning instance"""
    global _ensemble_reasoning
    if _ensemble_reasoning is None:
        _ensemble_reasoning = ECH0_Ensemble_Reasoning()
    return _ensemble_reasoning
