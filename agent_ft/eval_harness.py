import logging
from typing import Dict, Iterable, List

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class EvalHarness:
    """Lightweight perplexity evaluation for chat fine-tunes."""

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, max_length: int = 2048):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model.eval()

    @torch.inference_mode()
    def evaluate(self, records: Iterable[Dict[str, str]], batch_size: int = 2) -> Dict[str, float]:
        encoded_batches = self._prepare_batches(records, batch_size)
        losses: List[float] = []
        for batch in encoded_batches:
            outputs = self.model(**batch)
            loss = outputs.loss.detach().float()
            losses.append(loss.item())
        perplexity = torch.exp(torch.tensor(losses).mean()).item()
        logger.info("Eval perplexity: %.4f", perplexity)
        return {"perplexity": perplexity}

    def _prepare_batches(self, records: Iterable[Dict[str, str]], batch_size: int) -> List[Dict[str, torch.Tensor]]:
        messages = [f"{sample['text']}{sample['labels']}" for sample in records]
        tokenized = self.tokenizer(messages, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        labels = tokenized["input_ids"].clone()
        tokenized["labels"] = labels
        dataset = [{key: tensor[i] for key, tensor in tokenized.items()} for i in range(tokenized["input_ids"].size(0))]
        loader = DataLoader(dataset, batch_size=batch_size)
        return [{k: v.to(self.model.device) for k, v in batch.items()} for batch in loader]
