from dataclasses import dataclass, field
import numpy as np
from scipy.constants import k, Avogadro, g, c, h, e, pi, physical_constants

@dataclass
class NLPConfig:
    vocab_size: int = 10000
    embedding_dim: int = 50
    max_sentence_length: int = 200

@dataclass
class Tokenizer:
    config: NLPConfig = field(default_factory=NLPConfig)

    def __post_init__(self):
        self.token_to_id = {token: idx for idx, token in enumerate(range(self.config.vocab_size))}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def tokenize(self, text: str) -> np.ndarray:
        tokens = [self._get_or_create_token(token) for token in text.split()]
        return np.array(tokens[:self.config.max_sentence_length], dtype=np.float64)

    def detokenize(self, ids: np.ndarray) -> str:
        return ' '.join([self.id_to_token[id] for id in ids if id < self.config.vocab_size])

    def _get_or_create_token(self, token):
        if token not in self.token_to_id:
            next_idx = max(self.token_to_id.values()) + 1
            self.token_to_id[token] = next_idx
            self.id_to_token[next_idx] = token
        return self.token_to_id[token]

class EmbeddingLayer:
    def __init__(self, config: NLPConfig):
        self.config = config

    def forward(self, x: np.ndarray) -> np.ndarray:
        embedding_matrix = np.random.uniform(low=-1.0, high=1.0, size=(self.config.vocab_size, self.config.embedding_dim))
        return embedding_matrix[x].astype(np.float64)

class NLPModel:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.embedding_layer = EmbeddingLayer(self.tokenizer.config)

    def preprocess_text(self, text: str) -> np.ndarray:
        token_ids = self.tokenizer.tokenize(text)
        return token_ids

    def embed_text(self, text: str) -> np.ndarray:
        token_ids = self.preprocess_text(text)
        embeddings = self.embedding_layer.forward(token_ids)
        return embeddings

def run_demo():
    model = NLPModel()
    text_input = "Hello world this is a test sentence."
    tokenized_ids = model.preprocess_text(text_input)
    print(f"Tokenized IDs: {tokenized_ids}")
    sentence_embeddings = model.embed_text(text_input)
    print(f"Sentence Embeddings:\n{sentence_embeddings}")

if __name__ == '__main__':
    run_demo()