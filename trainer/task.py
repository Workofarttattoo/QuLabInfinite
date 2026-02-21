import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from agent_ft import EvalHarness
from trainer.data import Persona, format_chat_samples, load_jsonl, push_to_gcs
from trainer.lora import AdapterConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Llama with LoRA/QLoRA")
    parser.add_argument("--train_data_path", required=True)
    parser.add_argument("--eval_data_path", required=True)
    parser.add_argument("--persona_path", required=True)
    parser.add_argument("--hf_token", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", default="/gcs/your-bucket/llama-output")
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--upload_output_to", default="gs://your-bucket/llama-output")
    return parser.parse_args()


def load_model_and_tokenizer(model_name: str, adapter_cfg: AdapterConfig):
    quant_config = adapter_cfg.build_quantization_config()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto" if quant_config else None,
    )
    return model, tokenizer


def run_training(args: argparse.Namespace) -> Dict[str, Any]:
    login(token=args.hf_token)
    persona = Persona.load(args.persona_path)
    train_records = load_jsonl(args.train_data_path)
    eval_records = load_jsonl(args.eval_data_path)

    formatted_train = format_chat_samples(train_records, persona)
    formatted_eval = format_chat_samples(eval_records, persona)

    adapter_cfg = AdapterConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_qlora=args.use_qlora,
    )
    lora_config = adapter_cfg.build_lora_config()
    model, tokenizer = load_model_and_tokenizer(args.model_name, adapter_cfg)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        warmup_steps=args.warmup_steps,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=not args.use_qlora,
        bf16=args.use_qlora,
        report_to=["tensorboard"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=formatted_train,
        eval_dataset=formatted_eval,
        dataset_text_field="text",
        max_seq_length=2048,
        peft_config=lora_config,
    )

    logger.info("Starting training")
    trainer.train()
    logger.info("Training complete, running evaluation")

    eval_harness = EvalHarness(trainer.model, tokenizer)
    metrics = eval_harness.evaluate(formatted_eval)
    metrics_path = Path(args.output_dir) / "eval_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    logger.info("Saved eval metrics to %s", metrics_path)

    logger.info("Saving model artifacts to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.upload_output_to:
        for artifact in [metrics_path, Path(args.output_dir)]:
            destination = args.upload_output_to
            if artifact.is_file():
                push_to_gcs(str(artifact), f"{destination.rstrip('/')}/{artifact.name}")
            else:
                for inner in artifact.iterdir():
                    push_to_gcs(str(inner), f"{destination.rstrip('/')}/{inner.name}")

    return metrics


def main():
    args = parse_args()
    metrics = run_training(args)
    logger.info("Finished fine-tune with metrics: %s", metrics)


if __name__ == "__main__":
    main()
