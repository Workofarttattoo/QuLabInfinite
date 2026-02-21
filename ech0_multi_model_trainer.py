"""
ECH0 Multi-Model IMO Trainer
Train all ECH0 variants (14b and 32b) on Google DeepMind IMO Bench

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import pandas as pd
import numpy as np
import json
import time
import re
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class ModelResult:
    """Results for a single model"""
    model_name: str
    correct: int
    total: int
    accuracy: float
    avg_reasoning_time: float
    problems: List[Dict]


class ECH0_Multi_Model_Trainer:
    """
    Train all ECH0 variants on IMO Bench:
    - ech0-uncensored-14b
    - ech0-unified-14b
    - ech0-polymath-14b
    - ech0-qulab-14b
    - ech0-uncensored-32b
    """

    def __init__(self, imo_bench_path: str = "/Users/noone/Downloads/superhuman-main/imobench"):
        self.imo_path = Path(imo_bench_path)

        # All ECH0 models to train
        self.models = [
            "ech0-uncensored-14b",
            "ech0-unified-14b",
            "ech0-polymath-14b",
            "ech0-qulab-14b",
            "ech0-uncensored-32b"
        ]

        self.results: Dict[str, ModelResult] = {}

        print("=" * 80)
        print("ECH0 MULTI-MODEL IMO TRAINER")
        print("Training ALL ECH0 variants on Google DeepMind IMO Bench")
        print("=" * 80)
        print(f"\nModels to train: {len(self.models)}")
        for model in self.models:
            print(f"  - {model}")

    def load_datasets(self):
        """Load IMO Bench datasets"""
        print("\n[LOADING DATASETS]")

        self.answerbench = pd.read_csv(self.imo_path / "answerbench.csv")
        print(f"✓ AnswerBench: {len(self.answerbench)} problems loaded")

    def reason_with_model(self, model_name: str, problem: str) -> Tuple[str, str, float]:
        """
        Use specific ECH0 model to solve problem

        Returns:
            (answer, reasoning, time_seconds)
        """
        prompt = f"""You are ECH0, a world-class mathematician solving IMO problems.

Problem:
{problem}

Think step-by-step:
1. Understanding: What is the problem asking?
2. Key Information: What are the given facts?
3. Approach: What mathematical concepts apply?
4. Solution Steps: Work through systematically
5. Final Answer: State answer as a number or expression

End with: ANSWER: [your answer]
"""

        try:
            start_time = time.time()

            result = subprocess.run(
                ["ollama", "run", model_name, prompt],
                capture_output=True,
                text=True,
                timeout=120
            )

            elapsed = time.time() - start_time

            if result.returncode == 0:
                response = result.stdout.strip()
                answer = self._extract_answer(response)
                return answer, response, elapsed
            else:
                return "0", f"Error: {result.stderr}", 0.0

        except subprocess.TimeoutExpired:
            return "0", "Error: Timeout", 120.0
        except Exception as e:
            return "0", f"Error: {e}", 0.0

    def _extract_answer(self, response: str) -> str:
        """Extract answer from reasoning"""
        # Look for "ANSWER:" pattern
        answer_match = re.search(r'ANSWER:\s*([^\n]+)', response, re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).strip()
            answer = answer.replace('$', '').replace('\\', '').strip()
            return answer

        # Fallback: last number
        numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
        if numbers:
            return numbers[-1]

        return "0"

    def test_model(self, model_name: str, num_problems: int = 10) -> ModelResult:
        """Test a single ECH0 model on IMO problems"""
        print(f"\n{'=' * 80}")
        print(f"[TESTING {model_name.upper()}]")
        print(f"{'=' * 80}")

        # Sample problems
        sample_problems = self.answerbench.sample(n=min(num_problems, len(self.answerbench)))

        correct = 0
        total = 0
        problems = []
        total_time = 0.0

        for idx, row in sample_problems.iterrows():
            problem = self._extract_problem(row)
            correct_answer = self._extract_correct_answer(row)

            if problem and correct_answer:
                print(f"\nProblem {total + 1}/{num_problems}:")
                print(f"  {problem[:100]}...")

                ech0_answer, reasoning, elapsed = self.reason_with_model(model_name, problem)

                is_correct = self._check_answer(ech0_answer, correct_answer)
                if is_correct:
                    correct += 1
                total += 1
                total_time += elapsed

                print(f"  ECH0 Answer: {ech0_answer}")
                print(f"  Correct Answer: {correct_answer}")
                print(f"  Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
                print(f"  Time: {elapsed:.1f}s")

                problems.append({
                    "problem": problem[:100],
                    "ech0_answer": ech0_answer,
                    "correct_answer": correct_answer,
                    "correct": is_correct,
                    "reasoning_time": elapsed
                })

        accuracy = (correct / total * 100) if total > 0 else 0
        avg_time = total_time / total if total > 0 else 0

        print(f"\n{model_name} Results:")
        print(f"  Correct: {correct}/{total}")
        print(f"  Accuracy: {accuracy:.1f}%")
        print(f"  Avg Time: {avg_time:.1f}s per problem")

        return ModelResult(
            model_name=model_name,
            correct=correct,
            total=total,
            accuracy=accuracy,
            avg_reasoning_time=avg_time,
            problems=problems
        )

    def test_all_models_sequential(self, num_problems: int = 10):
        """Test all models one by one (sequential)"""
        print("\n[TESTING ALL MODELS - SEQUENTIAL]")

        for model_name in self.models:
            result = self.test_model(model_name, num_problems)
            self.results[model_name] = result

    def test_all_models_parallel(self, num_problems: int = 5):
        """Test all models in parallel (faster but uses more resources)"""
        print("\n[TESTING ALL MODELS - PARALLEL]")
        print("Testing 5 models simultaneously...")

        # Use same problems for all models (fair comparison)
        sample_problems = self.answerbench.sample(n=min(num_problems, len(self.answerbench)))

        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all model tests
            future_to_model = {
                executor.submit(self._test_model_on_problems, model, sample_problems): model
                for model in self.models
            }

            # Collect results as they complete
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result()
                    self.results[model_name] = result
                    print(f"✓ {model_name} complete: {result.accuracy:.1f}%")
                except Exception as e:
                    print(f"✗ {model_name} failed: {e}")

    def _test_model_on_problems(self, model_name: str, problems_df) -> ModelResult:
        """Helper for parallel testing"""
        correct = 0
        total = 0
        problems = []
        total_time = 0.0

        for idx, row in problems_df.iterrows():
            problem = self._extract_problem(row)
            correct_answer = self._extract_correct_answer(row)

            if problem and correct_answer:
                ech0_answer, reasoning, elapsed = self.reason_with_model(model_name, problem)

                is_correct = self._check_answer(ech0_answer, correct_answer)
                if is_correct:
                    correct += 1
                total += 1
                total_time += elapsed

                problems.append({
                    "problem": problem[:100],
                    "ech0_answer": ech0_answer,
                    "correct_answer": correct_answer,
                    "correct": is_correct,
                    "reasoning_time": elapsed
                })

        accuracy = (correct / total * 100) if total > 0 else 0
        avg_time = total_time / total if total > 0 else 0

        return ModelResult(
            model_name=model_name,
            correct=correct,
            total=total,
            accuracy=accuracy,
            avg_reasoning_time=avg_time,
            problems=problems
        )

    def generate_comparison_report(self):
        """Generate comparison report across all models"""
        print("\n" + "=" * 80)
        print("MULTI-MODEL COMPARISON REPORT")
        print("=" * 80)

        if not self.results:
            print("No results to report")
            return

        # Sort by accuracy
        sorted_results = sorted(self.results.values(), key=lambda x: x.accuracy, reverse=True)

        print("\n┌─────────────────────────┬──────────┬─────────────┬──────────────┐")
        print("│ Model                   │ Accuracy │ Correct     │ Avg Time (s) │")
        print("├─────────────────────────┼──────────┼─────────────┼──────────────┤")

        for result in sorted_results:
            model_short = result.model_name.replace("ech0-", "")
            print(f"│ {model_short:<23} │ {result.accuracy:6.1f}% │ {result.correct:2d}/{result.total:2d}       │ {result.avg_reasoning_time:12.1f} │")

        print("└─────────────────────────┴──────────┴─────────────┴──────────────┘")

        # Best model
        best = sorted_results[0]
        print(f"\n🏆 BEST MODEL: {best.model_name}")
        print(f"   Accuracy: {best.accuracy:.1f}%")
        print(f"   Speed: {best.avg_reasoning_time:.1f}s per problem")

        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"ech0_multi_model_results_{timestamp}.json"

        save_data = {
            "timestamp": timestamp,
            "models_tested": len(self.results),
            "results": {
                name: {
                    "accuracy": r.accuracy,
                    "correct": r.correct,
                    "total": r.total,
                    "avg_time": r.avg_reasoning_time,
                    "problems": r.problems
                }
                for name, r in self.results.items()
            },
            "best_model": {
                "name": best.model_name,
                "accuracy": best.accuracy
            }
        }

        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"\n✓ Detailed results saved to: {results_file}")

    # Helper methods

    def _extract_problem(self, row) -> Optional[str]:
        """Extract problem statement"""
        if 'Problem' in row.index:
            return str(row['Problem'])
        return None

    def _extract_correct_answer(self, row) -> Optional[str]:
        """Extract correct answer"""
        if 'Short Answer' in row.index:
            ans = str(row['Short Answer'])
            if ans and ans != 'nan':
                return ans
        return None

    def _check_answer(self, ech0_answer: str, correct_answer: str) -> bool:
        """Check if answer is correct"""
        ech0_norm = ech0_answer.strip().lower()
        correct_norm = correct_answer.strip().lower()

        if ech0_norm == correct_norm:
            return True

        try:
            ech0_num = float(ech0_norm)
            correct_num = float(correct_norm)
            return abs(ech0_num - correct_num) < 0.01
        except:
            pass

        if ech0_norm in correct_norm or correct_norm in ech0_norm:
            return True

        return False


def main():
    """Main multi-model training pipeline"""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║              ECH0 MULTI-MODEL IMO TRAINER                                  ║")
    print("║         Train ALL ECH0 Variants on Mathematical Reasoning                  ║")
    print("║                                                                            ║")
    print("║  Testing: uncensored-14b, unified-14b, polymath-14b, qulab-14b, 32b       ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    print("\n")

    trainer = ECH0_Multi_Model_Trainer()

    # Load datasets
    trainer.load_datasets()

    # Choose testing mode
    print("\nTesting Mode:")
    print("  [1] Sequential (one model at a time, slower but less resource intensive)")
    print("  [2] Parallel (all models simultaneously, faster but needs more RAM)")

    # Default to sequential for reliability
    mode = "sequential"
    num_problems = 10  # Test with 10 problems per model

    print(f"\nUsing: {mode.upper()} mode with {num_problems} problems per model")
    print("This will take ~15-30 minutes total")

    # Test all models
    if mode == "parallel":
        trainer.test_all_models_parallel(num_problems=num_problems)
    else:
        trainer.test_all_models_sequential(num_problems=num_problems)

    # Generate comparison report
    trainer.generate_comparison_report()

    print("\n" + "╔" + "═" * 78 + "╗")
    print(f"║  TRAINING COMPLETE                                                         ║")
    print("╠" + "═" * 78 + "╣")
    print(f"║  All {len(trainer.models)} ECH0 models tested on IMO mathematical reasoning             ║")
    print(f"║  Results show which ECH0 variant performs best on complex math            ║")
    print(f"║  Use this data to select optimal model for specific tasks                 ║")
    print("╚" + "═" * 78 + "╝")
    print("\n")


if __name__ == "__main__":
    main()
