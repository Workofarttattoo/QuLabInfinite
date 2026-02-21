"""
ECH0 LoRA Fine-Tuning - FULL EXECUTION
Training on 19,281 math+science examples using HuggingFace base model

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import os
import json
import torch
from pathlib import Path
from datasets import Dataset

print("=" * 80)
print("ECH0 LORA FINE-TUNING - STARTING NOW")
print("Training on 19,281 examples (Math + Science + Physics)")
print("=" * 80)
print()

# Check device
if torch.backends.mps.is_available():
    device = "mps"
    print("✓ Using Apple Metal (MPS)")
elif torch.cuda.is_available():
    device = "cuda"
    print(f"✓ Using CUDA: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("⚠ Using CPU (will be slow)")
print()

# Load training data
training_data_path = Path("/Users/noone/aios/QuLabInfinite/training_data/ech0_polymath_science_training.jsonl")
print(f"Loading training data from: {training_data_path}")

training_examples = []
with open(training_data_path) as f:
    for line in f:
        training_examples.append(json.loads(line))

print(f"✓ Loaded {len(training_examples):,} examples")
print()

# Show domain breakdown
domain_counts = {}
for ex in training_examples:
    domain = ex['domain']
    domain_counts[domain] = domain_counts.get(domain, 0) + 1

print("Domain breakdown:")
for domain, count in domain_counts.items():
    print(f"  {domain.title()}: {count:,} examples")
print()

# Install dependencies if needed
print("Checking dependencies...")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    print("✓ Transformers installed")
except ImportError:
    print("Installing transformers...")
    os.system("pip install -q transformers accelerate")
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    print("✓ PEFT installed")
except ImportError:
    print("Installing peft...")
    os.system("pip install -q peft")
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

try:
    import bitsandbytes
    print("✓ bitsandbytes installed")
except ImportError:
    print("Installing bitsandbytes...")
    os.system("pip install -q bitsandbytes")

print()
print("=" * 80)
print("LOADING BASE MODEL")
print("=" * 80)
print()

# Use a smaller compatible model (7B works better on laptops)
model_name = "deepseek-ai/deepseek-math-7b-base"
print(f"Base model: {model_name}")
print("This is a 7B math model - similar to your ech0-polymath-14b")
print()

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("✓ Tokenizer loaded")
print()

print("Loading model (this may take a few minutes)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if device in ["mps", "cuda"] else torch.float32,
    device_map="auto",
    trust_remote_code=True,
    load_in_8bit=True if device != "mps" else False  # 8-bit quantization for efficiency
)
print("✓ Model loaded")
print()

# Configure LoRA
print("=" * 80)
print("CONFIGURING LORA")
print("=" * 80)
print()

lora_config = LoraConfig(
    r=8,                              # LoRA rank
    lora_alpha=16,                    # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Which layers to adapt
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

print("LoRA Configuration:")
print(f"  Rank (r): {lora_config.r}")
print(f"  Alpha: {lora_config.lora_alpha}")
print(f"  Target modules: {lora_config.target_modules}")
print(f"  Dropout: {lora_config.lora_dropout}")
print()

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print()

# Prepare dataset
print("=" * 80)
print("PREPARING DATASET")
print("=" * 80)
print()

def format_example(ex):
    """Format example for training"""
    return {
        "text": f"<|im_start|>user\n{ex['prompt']}<|im_end|>\n<|im_start|>assistant\n{ex['completion']}<|im_end|>"
    }

formatted_data = [format_example(ex) for ex in training_examples]
dataset = Dataset.from_list(formatted_data)
print(f"✓ Created dataset with {len(dataset):,} examples")
print()

# Tokenize
print("Tokenizing dataset (this may take a few minutes)...")
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,
        padding="max_length",
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizing"
)
print("✓ Dataset tokenized")
print()

# Training configuration
print("=" * 80)
print("TRAINING CONFIGURATION")
print("=" * 80)
print()

output_dir = "/Users/noone/aios/QuLabInfinite/lora_output"
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=1,      # Small batch for laptop
    gradient_accumulation_steps=4,       # Effective batch size of 4
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    bf16=True if device in ["mps", "cuda"] else False,
    fp16=False,
    logging_dir=f"{output_dir}/logs",
    report_to="none",                    # Disable wandb/tensorboard
    remove_unused_columns=False,
)

total_steps = (len(tokenized_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)) * training_args.num_train_epochs

print(f"Output directory: {output_dir}")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"Learning rate: {training_args.learning_rate}")
print(f"Total training steps: ~{total_steps}")
print(f"Estimated time: 2-4 hours")
print()

# Create trainer
print("=" * 80)
print("STARTING TRAINING")
print("=" * 80)
print()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Train!
print("Training started...")
print("This will take 2-4 hours. Progress will be shown every 10 steps.")
print()

try:
    trainer.train()

    print()
    print("=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print()

    # Save model
    output_model_dir = f"{output_dir}/final_model"
    model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)

    print(f"✓ Model saved to: {output_model_dir}")
    print()

    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("1. Test the fine-tuned model on science questions")
    print("2. Convert to GGUF format for Ollama")
    print("3. Import to Ollama: ollama create ech0-science-trained -f modelfile")
    print()
    print("Expected improvement:")
    print("  - Science: Gibberish (0%) → 40-60% accuracy")
    print("  - Physics: Gibberish (0%) → 30-50% accuracy")
    print("  - Math: Maintains 10-15% accuracy")
    print()

except KeyboardInterrupt:
    print()
    print("Training interrupted by user")
    print("Partial model saved in:", output_dir)
except Exception as e:
    print()
    print(f"Error during training: {e}")
    import traceback
    traceback.print_exc()

print("=" * 80)
