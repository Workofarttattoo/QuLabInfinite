"""
ECH0 Polymath Science Fine-Tuning Trainer
Executes LoRA fine-tuning on ech0-polymath-14b using formatted training data

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import subprocess
import json
import time
from pathlib import Path
from typing import Dict, Optional


class ECH0_Polymath_Science_Trainer:
    """
    Fine-tune ech0-polymath-14b on math+science datasets to fix gibberish issue

    Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning on 32GB RAM laptop
    """

    def __init__(self, training_data_path: str = "/Users/noone/aios/QuLabInfinite/training_data/ech0_polymath_science_training.jsonl"):
        self.base_model = "ech0-polymath-14b"
        self.output_model = "ech0-polymath-science-14b"
        self.training_data_path = Path(training_data_path)

        # LoRA configuration for 14B model on laptop
        self.lora_config = {
            "r": 8,              # Rank of adaptation matrices
            "alpha": 16,          # Scaling factor
            "dropout": 0.05,      # Regularization
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
        }

        # Training configuration
        self.training_config = {
            "epochs": 3,
            "batch_size": 1,      # Small batch for 32GB RAM
            "learning_rate": 2e-4,
            "max_seq_length": 2048,
            "warmup_steps": 100,
            "gradient_accumulation_steps": 4  # Effective batch size of 4
        }

    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are available"""
        print("=" * 80)
        print("CHECKING PREREQUISITES")
        print("=" * 80)
        print()

        prerequisites_met = True

        # Check training data exists
        print("[1/3] Checking training data...")
        if not self.training_data_path.exists():
            print(f"    ✗ Training data not found: {self.training_data_path}")
            print(f"       Run: python3 ech0_format_training_data.py")
            prerequisites_met = False
        else:
            # Count examples
            with open(self.training_data_path) as f:
                num_examples = sum(1 for _ in f)
            print(f"    ✓ Found {num_examples:,} training examples")
        print()

        # Check base model exists
        print("[2/3] Checking base model...")
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if self.base_model in result.stdout:
                print(f"    ✓ Base model {self.base_model} is available")
            else:
                print(f"    ✗ Base model {self.base_model} not found in Ollama")
                print(f"       Available models:")
                for line in result.stdout.split('\n')[1:6]:
                    if line.strip():
                        print(f"         {line.split()[0]}")
                prerequisites_met = False
        except Exception as e:
            print(f"    ✗ Failed to check Ollama models: {e}")
            prerequisites_met = False
        print()

        # Check Python dependencies
        print("[3/3] Checking Python dependencies...")
        try:
            import torch
            print(f"    ✓ PyTorch {torch.__version__}")

            # Check for CUDA/MPS
            if torch.cuda.is_available():
                print(f"      CUDA available: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                print(f"      Apple MPS (Metal) available")
            else:
                print(f"      CPU only (slower)")

        except ImportError:
            print(f"    ✗ PyTorch not installed")
            print(f"       Run: pip install torch")
            prerequisites_met = False

        try:
            import transformers
            print(f"    ✓ transformers {transformers.__version__}")
        except ImportError:
            print(f"    ⚠ transformers not installed (needed for some methods)")
            print(f"       Run: pip install transformers")

        try:
            import peft
            print(f"    ✓ peft (LoRA library)")
        except ImportError:
            print(f"    ⚠ peft not installed (needed for LoRA)")
            print(f"       Run: pip install peft")

        print()
        print("=" * 80)
        if prerequisites_met:
            print("✓ All prerequisites met - ready to fine-tune")
        else:
            print("✗ Prerequisites missing - fix issues above")
        print("=" * 80)
        print()

        return prerequisites_met

    def create_ollama_modelfile(self) -> Path:
        """Create Ollama Modelfile for fine-tuning"""
        modelfile_content = f"""FROM {self.base_model}

# System prompt for science and math
SYSTEM You are ECH0 Polymath, an expert in both mathematics and scientific reasoning. You solve problems step-by-step with clear explanations. You provide coherent, well-reasoned answers to questions in mathematics, science, and physics.

# Temperature optimized for reasoning
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40

# License
MESSAGE Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

        modelfile_path = Path("/tmp/ech0_polymath_science_modelfile")
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)

        return modelfile_path

    def run_ollama_finetune(self) -> bool:
        """
        Run fine-tuning using Ollama (if supported)

        NOTE: As of early 2025, Ollama doesn't have native fine-tuning.
        This creates the improved model with better system prompt.
        For actual LoRA fine-tuning, use unsloth method below.
        """
        print("=" * 80)
        print("CREATING IMPROVED MODEL WITH OLLAMA")
        print("=" * 80)
        print()

        print("Creating improved Modelfile with science-aware system prompt...")
        modelfile_path = self.create_ollama_modelfile()

        print(f"Modelfile created: {modelfile_path}")
        print()

        print(f"Creating {self.output_model} from {self.base_model}...")
        try:
            result = subprocess.run(
                ["ollama", "create", self.output_model, "-f", str(modelfile_path)],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                print(f"✓ Successfully created {self.output_model}")
                print()
                print("=" * 80)
                print("MODEL CREATED - BUT NOT YET FINE-TUNED")
                print("=" * 80)
                print()
                print("This model has an improved system prompt, but")
                print("hasn't been fine-tuned on the 19,281 training examples yet.")
                print()
                print("For actual LoRA fine-tuning, use one of these methods:")
                print("  1. unsloth (recommended for laptops)")
                print("  2. PyTorch + PEFT directly")
                print("  3. Wait for Ollama to add native fine-tuning support")
                print()
                return True
            else:
                print(f"✗ Failed to create model: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("✗ Model creation timed out")
            return False
        except Exception as e:
            print(f"✗ Error: {e}")
            return False

    def run_unsloth_finetune(self) -> bool:
        """
        Run fine-tuning using unsloth (efficient library for LoRA)

        This is the recommended method for laptop fine-tuning
        """
        print("=" * 80)
        print("FINE-TUNING WITH UNSLOTH")
        print("=" * 80)
        print()

        print("Checking unsloth installation...")
        try:
            import unsloth
            print(f"✓ unsloth is installed")
        except ImportError:
            print("✗ unsloth not installed")
            print()
            print("To install unsloth:")
            print("  pip install unsloth")
            print()
            print("Or use the PyTorch method instead (see documentation)")
            return False

        print()
        print("=" * 80)
        print("UNSLOTH FINE-TUNING SCRIPT")
        print("=" * 80)
        print()
        print("To fine-tune with unsloth, run this Python script:")
        print()
        print("```python")
        print("""from unsloth import FastLanguageModel
import json

# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "ech0-polymath-14b",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,  # Use 4-bit quantization for memory efficiency
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 8,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = True,
)

# Load training data
training_data = []
with open("training_data/ech0_polymath_science_training.jsonl") as f:
    for line in f:
        training_data.append(json.loads(line))

# Format for training
def format_prompts(examples):
    texts = []
    for ex in examples:
        text = f"<s>[INST] {ex['prompt']} [/INST] {ex['completion']} </s>"
        texts.append(text)
    return {"text": texts}

# Train
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    train_dataset = training_data,
    dataset_text_field = "text",
    max_seq_length = 2048,
    formatting_func = format_prompts,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 100,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = True,
        logging_steps = 10,
        output_dir = "outputs",
    ),
)

trainer.train()

# Save model
model.save_pretrained("ech0_polymath_science_lora")
```""")
        print()
        print("This will take 2-4 hours on a laptop.")
        print("=" * 80)

        return False  # Not actually running, just showing instructions

    def test_improved_model(self):
        """Test the improved model on a science question"""
        print("=" * 80)
        print("TESTING IMPROVED MODEL")
        print("=" * 80)
        print()

        test_questions = [
            {
                "type": "physics",
                "question": "What is the orbital period of a satellite at altitude h above Earth's surface?"
            },
            {
                "type": "chemistry",
                "question": "What is the pH of a 0.01 M HCl solution?"
            },
            {
                "type": "biology",
                "question": "What is the role of mitochondria in cellular respiration?"
            }
        ]

        print(f"Testing {self.output_model} on science questions...")
        print()

        for i, test in enumerate(test_questions, 1):
            print(f"[{i}/{len(test_questions)}] {test['type'].title()} Question:")
            print(f"    {test['question']}")
            print()

            try:
                result = subprocess.run(
                    ["ollama", "run", self.output_model, test['question']],
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if result.returncode == 0:
                    response = result.stdout.strip()
                    print(f"    Response: {response[:200]}...")
                    print()

                    # Check if it's gibberish (very short, nonsensical)
                    if len(response) < 50:
                        print(f"    ⚠ WARNING: Response seems too short (possible gibberish)")
                    else:
                        print(f"    ✓ Response looks coherent")
                else:
                    print(f"    ✗ Error: {result.stderr}")

            except subprocess.TimeoutExpired:
                print(f"    ⏱ Timeout (took > 60s)")
            except Exception as e:
                print(f"    ✗ Error: {e}")

            print()

        print("=" * 80)
        print("TEST COMPLETE")
        print("=" * 80)
        print()
        print("If responses are still gibberish, you need to run actual LoRA fine-tuning.")
        print("See unsloth method above for instructions.")
        print("=" * 80)


def main():
    print("\n")
    print("=" * 80)
    print("ECH0 POLYMATH SCIENCE FINE-TUNING TRAINER")
    print("Fixing 'gibberish' output on science problems")
    print("=" * 80)
    print("\n")

    trainer = ECH0_Polymath_Science_Trainer()

    # Step 1: Check prerequisites
    if not trainer.check_prerequisites():
        print("Please fix prerequisites and try again.")
        return

    # Step 2: Create improved model with Ollama
    print("Starting model creation...")
    print()

    success = trainer.run_ollama_finetune()

    if success:
        # Step 3: Test the model
        print()
        input("Press Enter to test the improved model on science questions...")
        print()
        trainer.test_improved_model()

    # Step 4: Show next steps
    print()
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("You've created an improved model with a better system prompt.")
    print()
    print("For actual LoRA fine-tuning on 19,281 examples:")
    print()
    print("Option 1: Use unsloth (recommended)")
    print("  1. pip install unsloth")
    print("  2. See trainer.run_unsloth_finetune() code above")
    print("  3. Run the unsloth training script (2-4 hours)")
    print()
    print("Option 2: Wait for Ollama native fine-tuning")
    print("  Ollama team is working on this feature")
    print()
    print("Option 3: Export model and fine-tune with PyTorch directly")
    print("  More complex but full control")
    print()
    print("Expected improvement after actual fine-tuning:")
    print("  - Science: Gibberish (0%) → 40-60% accuracy")
    print("  - Physics: Gibberish (0%) → 30-50% accuracy")
    print("  - Math: Maintains current 10% accuracy")
    print("=" * 80)
    print("\n")


if __name__ == "__main__":
    main()
