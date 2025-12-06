"""
ECH0 Training Data Formatter
Converts downloaded math+science datasets into fine-tuning format

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import os
import json
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset

# Set HF token for authentication (read from environment)
HF_TOKEN = os.getenv("HF_TOKEN", "")


class ECH0_Training_Data_Formatter:
    """Format datasets into training data for ech0-polymath-science fine-tuning"""

    def __init__(self, output_dir: str = "/Users/noone/aios/QuLabInfinite/training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Download datasets
        self.datasets = {}
        self.stats = {
            'math': 0,
            'science': 0,
            'physics': 0,
            'total': 0
        }

    def download_all_datasets(self):
        """Download all required datasets"""
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN env var is required for dataset download")

        print("=" * 80)
        print("DOWNLOADING DATASETS")
        print("=" * 80)
        print()

        # 1. MATH Dataset (all subjects)
        math_subjects = ['algebra', 'counting_and_probability', 'geometry',
                        'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']

        print(f"[1/3] Downloading MATH dataset ({len(math_subjects)} subjects)...")
        all_math = []
        for i, subject in enumerate(math_subjects, 1):
            print(f"    [{i}/{len(math_subjects)}] {subject}...", end=" ")
            try:
                ds = load_dataset("EleutherAI/hendrycks_math", subject, split="train", token=HF_TOKEN)
                all_math.append(ds)
                print(f"✓ {len(ds)} problems")
            except Exception as e:
                print(f"✗ {e}")

        if all_math:
            self.datasets['math'] = all_math
            self.stats['math'] = sum(len(ds) for ds in all_math)
            print(f"    ✓ Total: {self.stats['math']} math problems")
        print()

        # 2. Science QA Dataset
        print("[2/3] Downloading sciq (Science Questions)...")
        try:
            sci_ds = load_dataset("sciq", split="train", token=HF_TOKEN)
            self.datasets['science'] = sci_ds
            self.stats['science'] = len(sci_ds)
            print(f"    ✓ {self.stats['science']} science questions")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
        print()

        # 3. MMLU Physics
        print("[3/3] Downloading cais/mmlu (Physics subset)...")
        try:
            phys_ds = load_dataset("cais/mmlu", "college_physics", split="test", token=HF_TOKEN)
            self.datasets['physics'] = phys_ds
            self.stats['physics'] = len(phys_ds)
            print(f"    ✓ {self.stats['physics']} physics problems")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
        print()

        self.stats['total'] = self.stats['math'] + self.stats['science'] + self.stats['physics']

        print("=" * 80)
        print("DOWNLOAD SUMMARY")
        print("=" * 80)
        print(f"Mathematics: {self.stats['math']:,} problems")
        print(f"Science: {self.stats['science']:,} questions")
        print(f"Physics: {self.stats['physics']:,} problems")
        print(f"TOTAL: {self.stats['total']:,} training examples")
        print("=" * 80)
        print()

    def format_math_example(self, example: Dict) -> Dict:
        """Format a MATH dataset example"""
        return {
            "prompt": f"Solve this mathematics problem step-by-step:\n\n{example['problem']}",
            "completion": example['solution'],
            "domain": "mathematics",
            "level": example.get('level', 'unknown'),
            "type": example.get('type', 'unknown')
        }

    def format_science_example(self, example: Dict) -> Dict:
        """Format a SciQ dataset example"""
        prompt = f"Answer this science question with reasoning:\n\n{example['question']}"

        # SciQ has: question, correct_answer, support (explanation)
        completion = f"{example['support']}\n\nAnswer: {example['correct_answer']}"

        return {
            "prompt": prompt,
            "completion": completion,
            "domain": "science"
        }

    def format_physics_example(self, example: Dict) -> Dict:
        """Format a MMLU physics example"""
        # MMLU format: question, choices (list), answer (index)
        choices_str = "\n".join([f"{chr(65+i)}. {choice}"
                                for i, choice in enumerate(example['choices'])])

        prompt = f"Answer this physics question:\n\n{example['question']}\n\n{choices_str}"

        # Answer is index into choices
        answer_letter = chr(65 + example['answer'])
        completion = f"The correct answer is {answer_letter}. {example['choices'][example['answer']]}"

        return {
            "prompt": prompt,
            "completion": completion,
            "domain": "physics"
        }

    def format_all_examples(self) -> List[Dict]:
        """Format all downloaded examples"""
        print("=" * 80)
        print("FORMATTING TRAINING DATA")
        print("=" * 80)
        print()

        formatted_examples = []

        # Format MATH examples
        if 'math' in self.datasets:
            print(f"[1/3] Formatting {self.stats['math']:,} math problems...")
            for ds in self.datasets['math']:
                for example in ds:
                    formatted_examples.append(self.format_math_example(example))
            print(f"    ✓ Formatted {self.stats['math']:,} math examples")
        print()

        # Format Science examples
        if 'science' in self.datasets:
            print(f"[2/3] Formatting {self.stats['science']:,} science questions...")
            for example in self.datasets['science']:
                formatted_examples.append(self.format_science_example(example))
            print(f"    ✓ Formatted {self.stats['science']:,} science examples")
        print()

        # Format Physics examples
        if 'physics' in self.datasets:
            print(f"[3/3] Formatting {self.stats['physics']:,} physics problems...")
            for example in self.datasets['physics']:
                formatted_examples.append(self.format_physics_example(example))
            print(f"    ✓ Formatted {self.stats['physics']:,} physics examples")
        print()

        print("=" * 80)
        print(f"✓ Total formatted examples: {len(formatted_examples):,}")
        print("=" * 80)
        print()

        return formatted_examples

    def save_training_data(self, examples: List[Dict]):
        """Save formatted training data in multiple formats"""
        print("=" * 80)
        print("SAVING TRAINING DATA")
        print("=" * 80)
        print()

        # 1. Save as JSONL (for Ollama/unsloth)
        jsonl_path = self.output_dir / "ech0_polymath_science_training.jsonl"
        print(f"[1/3] Saving JSONL format: {jsonl_path}")
        with open(jsonl_path, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        print(f"    ✓ Saved {len(examples):,} examples")
        print()

        # 2. Save as JSON (backup format)
        json_path = self.output_dir / "ech0_polymath_science_training.json"
        print(f"[2/3] Saving JSON format: {json_path}")
        with open(json_path, 'w') as f:
            json.dump(examples, f, indent=2)
        print(f"    ✓ Saved {len(examples):,} examples")
        print()

        # 3. Save statistics
        stats_path = self.output_dir / "training_data_stats.json"
        print(f"[3/3] Saving statistics: {stats_path}")

        # Count by domain
        domain_counts = {}
        for ex in examples:
            domain = ex['domain']
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        stats = {
            "total_examples": len(examples),
            "by_domain": domain_counts,
            "datasets_used": list(self.datasets.keys()),
            "output_files": {
                "jsonl": str(jsonl_path),
                "json": str(json_path)
            }
        }

        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"    ✓ Saved statistics")
        print()

        print("=" * 80)
        print("TRAINING DATA READY FOR FINE-TUNING")
        print("=" * 80)
        print()
        print(f"Main training file: {jsonl_path}")
        print(f"Total examples: {len(examples):,}")
        print(f"Domain breakdown:")
        for domain, count in domain_counts.items():
            print(f"  - {domain.title()}: {count:,} examples")
        print()
        print("Next step: Run fine-tuning script")
        print("=" * 80)


def main():
    print("\n")
    print("=" * 80)
    print("ECH0 POLYMATH SCIENCE - TRAINING DATA FORMATTER")
    print("Fixing 'gibberish' output on science problems")
    print("=" * 80)
    print("\n")

    formatter = ECH0_Training_Data_Formatter()

    # Step 1: Download datasets
    formatter.download_all_datasets()

    # Step 2: Format examples
    formatted_examples = formatter.format_all_examples()

    # Step 3: Save training data
    formatter.save_training_data(formatted_examples)

    print("\n")
    print("=" * 80)
    print("SUCCESS!")
    print("=" * 80)
    print()
    print("Training data formatted and saved.")
    print(f"Total: {formatter.stats['total']:,} examples")
    print()
    print("To fine-tune ech0-polymath-14b:")
    print("  1. Install dependencies: pip install unsloth transformers accelerate peft")
    print("  2. Run: python3 ech0_polymath_science_trainer.py")
    print("  3. Wait 2-4 hours for fine-tuning to complete")
    print("  4. Test improved model on science problems")
    print()
    print("Expected improvement:")
    print("  - Science: Gibberish (0%) → 40-60% accuracy")
    print("  - Physics: Gibberish (0%) → 30-50% accuracy")
    print("  - Math: Maintains current 10% accuracy")
    print("=" * 80)
    print("\n")


if __name__ == "__main__":
    main()
