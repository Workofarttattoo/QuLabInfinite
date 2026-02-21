from dataclasses import dataclass
from typing import Optional

from bitsandbytes import BitsAndBytesConfig
from peft import LoraConfig
import torch


@dataclass
class AdapterConfig:
    """LoRA/QLoRA configuration shared across training scripts."""

    r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Optional[list[str]] = None
    use_qlora: bool = False

    def build_lora_config(self) -> LoraConfig:
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules or [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

    def build_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        if not self.use_qlora:
            return None
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
