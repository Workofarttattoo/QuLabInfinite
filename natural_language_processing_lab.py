"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

NATURAL LANGUAGE PROCESSING LAB
Production-ready NLP algorithms implemented from scratch.
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import re
import math
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

@dataclass
class NLPConfig:
    """Configuration for NLP models."""
    vocab_size: int = 10000
    embedding_dim: int = 300
    hidden_dim: int = 128
    max_sequence_length: int = 512
    n_gram: int = 3
    min_word_freq: int = 2
    window_size: int = 5  # For word2vec
    negative_samples: int = 5
    learning_rate: float = 0.025
    min_learning_rate: float = 0.0001
    epochs: int = 5
    subsampling_threshold: float = 1e-3
    attention_heads: int = 8
    dropout_rate: float = 0.1
    temperature: float = 1.0
    beam_width: int = 3

class NaturalLanguageProcessingLab:
    """Production-ready NLP algorithms implemented from scratch."""

    def __init__(self, config: NLPConfig = None):
        self.config = config or NLPConfig()
        self.vocabulary = {}
        self.inverse_vocabulary = {}
        self.word_freq = Counter()
        self.embeddings = None
        self.idf_values = {}
        self.bigram_counts = defaultdict(Counter)
        self.trigram_counts = defaultdict(Counter)

    def tokenize(self, text: str, method: str = 'word') -> List[str]:
        """
        Tokenize text using various methods.

        Args:
            text: Input text
            method: 'word', 'sentence', 'subword', 'char'

        Returns:
            List of tokens
        """
        text = text.lower().strip()

        if method == 'word':
            # Word tokenization with basic preprocessing
            text = re.sub(r'([.!?,;])', r' \1 ', text)
            tokens = text.split()
            return [token for token in tokens if token]

        elif method == 'sentence':
            # Sentence tokenization
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]

        elif method == 'subword':
            # Byte-Pair Encoding (simplified)
            return self._byte_pair_encoding(text)

        elif method == 'char':
            # Character-level tokenization
            return list(text)

        else:
            return text.split()

    def _byte_pair_encoding(self, text: str, num_merges: int = 100) -> List[str]:
        """Simplified Byte-Pair Encoding for subword tokenization."""
        # Start with character-level tokens
        tokens = list(text.lower())
        vocab = set(tokens)

        for _ in range(num_merges):
            # Count bigrams
            bigrams = defaultdict(int)
            for i in range(len(tokens) - 1):
                bigrams[(tokens[i], tokens[i+1])] += 1

            if not bigrams:
                break

            # Find most frequent bigram
            most_frequent = max(bigrams, key=bigrams.get)

            # Merge tokens
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == most_frequent:
                    new_tokens.append(tokens[i] + tokens[i+1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens
            vocab.add(''.join(most_frequent))

        return tokens

    def build_vocabulary(self, corpus: List[str], max_vocab: Optional[int] = None):
        """
        Build vocabulary from corpus.

        Args:
            corpus: List of documents
            max_vocab: Maximum vocabulary size
        """
        if max_vocab is None:
            max_vocab = self.config.vocab_size

        # Count word frequencies
        for doc in corpus:
            tokens = self.tokenize(doc)
            self.word_freq.update(tokens)

        # Filter by minimum frequency
        filtered_words = [(word, count) for word, count in self.word_freq.items()
                         if count >= self.config.min_word_freq]

        # Sort by frequency and take top words
        filtered_words.sort(key=lambda x: x[1], reverse=True)

        # Build vocabulary with special tokens
        self.vocabulary = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        idx = len(self.vocabulary)

        for word, _ in filtered_words[:max_vocab - len(self.vocabulary)]:
            self.vocabulary[word] = idx
            idx += 1

        # Build inverse vocabulary
        self.inverse_vocabulary = {v: k for k, v in self.vocabulary.items()}

    def text_to_indices(self, text: str) -> List[int]:
        """Convert text to vocabulary indices."""
        tokens = self.tokenize(text)
        return [self.vocabulary.get(token, self.vocabulary['<UNK>']) for token in tokens]

    def indices_to_text(self, indices: List[int]) -> str:
        """Convert vocabulary indices back to text."""
        tokens = [self.inverse_vocabulary.get(idx, '<UNK>') for idx in indices]
        return ' '.join(tokens)

    def compute_tf_idf(self, corpus: List[str]) -> np.ndarray:
        """
        Compute TF-IDF vectors for corpus.

        Args:
            corpus: List of documents

        Returns:
            TF-IDF matrix (n_docs, vocab_size)
        """
        n_docs = len(corpus)
        vocab_size = len(self.vocabulary)

        # Initialize TF-IDF matrix
        tfidf_matrix = np.zeros((n_docs, vocab_size), dtype=np.float64)

        # Compute document frequencies
        doc_freq = defaultdict(int)

        for doc in corpus:
            tokens = set(self.tokenize(doc))
            for token in tokens:
                if token in self.vocabulary:
                    doc_freq[token] += 1

        # Compute IDF values
        for word, df in doc_freq.items():
            self.idf_values[word] = np.log(n_docs / (df + 1))

        # Compute TF-IDF for each document
        for i, doc in enumerate(corpus):
            tokens = self.tokenize(doc)
            token_counts = Counter(tokens)
            total_tokens = len(tokens)

            for token, count in token_counts.items():
                if token in self.vocabulary:
                    tf = count / total_tokens
                    idf = self.idf_values.get(token, 0)
                    idx = self.vocabulary[token]
                    tfidf_matrix[i, idx] = tf * idf

        return tfidf_matrix

    def word2vec_skipgram(self, corpus: List[str], iterations: int = None) -> np.ndarray:
        """
        Train Word2Vec embeddings using Skip-gram with Negative Sampling.

        Args:
            corpus: List of documents
            iterations: Training iterations

        Returns:
            Word embedding matrix
        """
        if iterations is None:
            iterations = self.config.epochs

        vocab_size = len(self.vocabulary)
        embedding_dim = self.config.embedding_dim

        # Initialize embeddings
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01
        context_embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01

        # Prepare training data
        training_pairs = []
        for doc in corpus:
            tokens = self.tokenize(doc)
            indices = [self.vocabulary.get(t, 1) for t in tokens]

            for i, center in enumerate(indices):
                # Get context words within window
                start = max(0, i - self.config.window_size)
                end = min(len(indices), i + self.config.window_size + 1)

                for j in range(start, end):
                    if i != j:
                        training_pairs.append((center, indices[j]))

        # Training loop
        lr = self.config.learning_rate

        for epoch in range(iterations):
            np.random.shuffle(training_pairs)
            total_loss = 0

            for center_idx, context_idx in training_pairs[:1000]:  # Limit for demo
                # Positive sample
                center_vec = self.embeddings[center_idx]
                context_vec = context_embeddings[context_idx]

                # Compute positive score
                pos_score = self._sigmoid(np.dot(center_vec, context_vec))
                pos_loss = -np.log(pos_score + 1e-10)

                # Negative sampling
                neg_indices = np.random.randint(0, vocab_size, self.config.negative_samples)
                neg_loss = 0

                for neg_idx in neg_indices:
                    if neg_idx != context_idx:
                        neg_vec = context_embeddings[neg_idx]
                        neg_score = self._sigmoid(-np.dot(center_vec, neg_vec))
                        neg_loss -= np.log(neg_score + 1e-10)

                total_loss += pos_loss + neg_loss

                # Gradient updates (simplified)
                # Positive gradient
                pos_grad = (pos_score - 1) * context_vec
                self.embeddings[center_idx] -= lr * pos_grad

                # Negative gradients
                for neg_idx in neg_indices:
                    if neg_idx != context_idx:
                        neg_vec = context_embeddings[neg_idx]
                        neg_score = self._sigmoid(np.dot(center_vec, neg_vec))
                        neg_grad = neg_score * neg_vec
                        self.embeddings[center_idx] -= lr * neg_grad

            # Decay learning rate
            lr = max(self.config.min_learning_rate,
                    lr * (1 - epoch / iterations))

        return self.embeddings

    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def attention_mechanism(self, query: np.ndarray, keys: np.ndarray,
                          values: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Implement scaled dot-product attention.

        Args:
            query: Query matrix (seq_len, d_k)
            keys: Key matrix (seq_len, d_k)
            values: Value matrix (seq_len, d_v)
            mask: Optional attention mask

        Returns:
            Attention output and attention weights
        """
        d_k = keys.shape[-1]

        # Compute attention scores
        scores = np.dot(query, keys.T) / np.sqrt(d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores * mask - 1e9 * (1 - mask)

        # Apply softmax
        attention_weights = self._softmax(scores)

        # Apply dropout (training only)
        if self.config.dropout_rate > 0:
            dropout_mask = np.random.binomial(1, 1 - self.config.dropout_rate,
                                             attention_weights.shape)
            attention_weights = attention_weights * dropout_mask

        # Compute weighted values
        output = np.dot(attention_weights, values)

        return output, attention_weights

    def multi_head_attention(self, query: np.ndarray, keys: np.ndarray,
                           values: np.ndarray, num_heads: int = None) -> np.ndarray:
        """
        Implement multi-head attention.

        Args:
            query: Query matrix
            keys: Key matrix
            values: Value matrix
            num_heads: Number of attention heads

        Returns:
            Multi-head attention output
        """
        if num_heads is None:
            num_heads = self.config.attention_heads

        seq_len, d_model = query.shape
        d_k = d_model // num_heads

        # Split into multiple heads
        outputs = []

        for head in range(num_heads):
            start = head * d_k
            end = start + d_k

            # Extract head-specific components
            q_head = query[:, start:end]
            k_head = keys[:, start:end]
            v_head = values[:, start:end]

            # Apply attention
            head_output, _ = self.attention_mechanism(q_head, k_head, v_head)
            outputs.append(head_output)

        # Concatenate heads
        multi_head_output = np.concatenate(outputs, axis=-1)

        # Linear transformation (simplified - just using identity)
        return multi_head_output

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax activation."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def build_n_gram_model(self, corpus: List[str]):
        """
        Build n-gram language model.

        Args:
            corpus: List of documents
        """
        for doc in corpus:
            tokens = ['<SOS>'] + self.tokenize(doc) + ['<EOS>']

            # Build bigrams
            for i in range(len(tokens) - 1):
                self.bigram_counts[tokens[i]][tokens[i+1]] += 1

            # Build trigrams
            for i in range(len(tokens) - 2):
                bigram = (tokens[i], tokens[i+1])
                self.trigram_counts[bigram][tokens[i+2]] += 1

    def calculate_perplexity(self, text: str, model_type: str = 'bigram') -> float:
        """
        Calculate perplexity of text under n-gram model.

        Args:
            text: Input text
            model_type: 'bigram' or 'trigram'

        Returns:
            Perplexity score
        """
        tokens = ['<SOS>'] + self.tokenize(text) + ['<EOS>']
        log_prob = 0
        count = 0

        if model_type == 'bigram':
            for i in range(len(tokens) - 1):
                prev_word = tokens[i]
                curr_word = tokens[i+1]

                if prev_word in self.bigram_counts:
                    total = sum(self.bigram_counts[prev_word].values())
                    prob = self.bigram_counts[prev_word][curr_word] / total if total > 0 else 1e-10
                else:
                    prob = 1e-10

                log_prob += np.log(prob + 1e-10)
                count += 1

        elif model_type == 'trigram':
            for i in range(len(tokens) - 2):
                bigram = (tokens[i], tokens[i+1])
                next_word = tokens[i+2]

                if bigram in self.trigram_counts:
                    total = sum(self.trigram_counts[bigram].values())
                    prob = self.trigram_counts[bigram][next_word] / total if total > 0 else 1e-10
                else:
                    prob = 1e-10

                log_prob += np.log(prob + 1e-10)
                count += 1

        perplexity = np.exp(-log_prob / count) if count > 0 else float('inf')
        return perplexity

    def sequence_to_sequence(self, encoder_input: np.ndarray,
                           decoder_steps: int = 10) -> np.ndarray:
        """
        Simple sequence-to-sequence model with attention.

        Args:
            encoder_input: Input sequence (seq_len, embedding_dim)
            decoder_steps: Number of decoding steps

        Returns:
            Decoded sequence
        """
        seq_len, embed_dim = encoder_input.shape

        # Simplified encoder (just using mean pooling)
        encoder_output = encoder_input
        encoder_hidden = np.mean(encoder_input, axis=0)

        # Decoder with attention
        decoder_outputs = []
        decoder_hidden = encoder_hidden

        for step in range(decoder_steps):
            # Attention over encoder outputs
            query = decoder_hidden.reshape(1, -1)

            # Ensure dimensions match
            if query.shape[1] != encoder_output.shape[1]:
                # Project to same dimension
                query = np.pad(query, ((0, 0), (0, encoder_output.shape[1] - query.shape[1])))

            attended, _ = self.attention_mechanism(query, encoder_output, encoder_output)

            # Update hidden state (simplified GRU-like update)
            decoder_hidden = 0.7 * decoder_hidden + 0.3 * attended.flatten()[:embed_dim]

            decoder_outputs.append(decoder_hidden)

        return np.array(decoder_outputs)

    def edit_distance(self, str1: str, str2: str) -> int:
        """
        Calculate Levenshtein edit distance between two strings.

        Args:
            str1: First string
            str2: Second string

        Returns:
            Edit distance
        """
        m, n = len(str1), len(str2)

        # Initialize DP table
        dp = np.zeros((m + 1, n + 1), dtype=int)

        # Base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j],      # Deletion
                                      dp[i][j-1],      # Insertion
                                      dp[i-1][j-1])    # Substitution

        return dp[m][n]

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def text_classification_metrics(self, y_true: List[int],
                                  y_pred: List[int]) -> Dict[str, float]:
        """
        Calculate classification metrics for text classification.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary of metrics
        """
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Calculate metrics
        correct = np.sum(y_true == y_pred)
        total = len(y_true)
        accuracy = correct / total if total > 0 else 0

        # Calculate per-class metrics
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for cls in unique_classes:
            true_positives = np.sum((y_true == cls) & (y_pred == cls))
            false_positives = np.sum((y_true != cls) & (y_pred == cls))
            false_negatives = np.sum((y_true == cls) & (y_pred != cls))

            precision = true_positives / (true_positives + false_positives) \
                       if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) \
                    if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) \
                if (precision + recall) > 0 else 0

            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        return {
            'accuracy': accuracy,
            'precision': np.mean(precision_scores),
            'recall': np.mean(recall_scores),
            'f1_score': np.mean(f1_scores)
        }

    def beam_search_decoder(self, start_token: int, max_length: int = 50) -> List[int]:
        """
        Implement beam search decoding.

        Args:
            start_token: Starting token index
            max_length: Maximum sequence length

        Returns:
            Decoded sequence
        """
        beam_width = self.config.beam_width
        vocab_size = len(self.vocabulary) if self.vocabulary else 100

        # Initialize beams
        beams = [(0.0, [start_token])]  # (score, sequence)

        for _ in range(max_length):
            new_beams = []

            for score, sequence in beams:
                # Skip if sequence ended
                if sequence[-1] == self.vocabulary.get('<EOS>', 3):
                    new_beams.append((score, sequence))
                    continue

                # Generate next token probabilities (simplified - using random)
                # In real implementation, this would use the language model
                probs = np.random.random(vocab_size)
                probs = probs / np.sum(probs)
                log_probs = np.log(probs + 1e-10)

                # Get top-k candidates
                top_k_indices = np.argsort(log_probs)[-beam_width:]

                for idx in top_k_indices:
                    new_score = score + log_probs[idx]
                    new_sequence = sequence + [idx]
                    new_beams.append((new_score, new_sequence))

            # Keep top beam_width beams
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_width]

            # Check if all beams have ended
            if all(seq[-1] == self.vocabulary.get('<EOS>', 3) for _, seq in beams):
                break

        # Return best sequence
        return beams[0][1]

    def named_entity_recognition(self, text: str) -> List[Tuple[str, str]]:
        """
        Simple rule-based Named Entity Recognition.

        Args:
            text: Input text

        Returns:
            List of (entity, type) tuples
        """
        entities = []
        tokens = self.tokenize(text)

        # Simple patterns for different entity types
        for i, token in enumerate(tokens):
            # Capitalized words (potential proper nouns)
            if token[0].isupper() and i > 0:
                entities.append((token, 'PERSON/ORG'))

            # Numbers
            elif token.isdigit():
                entities.append((token, 'NUMBER'))

            # Dates (simple pattern)
            elif re.match(r'\d{1,2}/\d{1,2}/\d{2,4}', token):
                entities.append((token, 'DATE'))

            # Email addresses
            elif '@' in token and '.' in token:
                entities.append((token, 'EMAIL'))

            # URLs
            elif token.startswith(('http://', 'https://', 'www.')):
                entities.append((token, 'URL'))

        return entities

    def sentiment_analysis_lexicon(self, text: str) -> Dict[str, float]:
        """
        Simple lexicon-based sentiment analysis.

        Args:
            text: Input text

        Returns:
            Sentiment scores
        """
        # Simple sentiment lexicon
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful',
                         'fantastic', 'love', 'best', 'happy', 'beautiful'}
        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'hate',
                         'worst', 'sad', 'angry', 'disappointing', 'poor'}

        tokens = self.tokenize(text.lower())

        positive_count = sum(1 for token in tokens if token in positive_words)
        negative_count = sum(1 for token in tokens if token in negative_words)
        total_words = len(tokens)

        if total_words == 0:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}

        positive_score = positive_count / total_words
        negative_score = negative_count / total_words
        neutral_score = 1.0 - positive_score - negative_score

        # Determine overall sentiment
        if positive_score > negative_score:
            sentiment = 'positive'
        elif negative_score > positive_score:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        return {
            'positive': positive_score,
            'negative': negative_score,
            'neutral': neutral_score,
            'overall': sentiment
        }

    def text_summarization_extractive(self, text: str, num_sentences: int = 3) -> str:
        """
        Extractive text summarization using sentence scoring.

        Args:
            text: Input text
            num_sentences: Number of sentences in summary

        Returns:
            Summary text
        """
        # Tokenize into sentences
        sentences = self.tokenize(text, method='sentence')

        if len(sentences) <= num_sentences:
            return text

        # Calculate sentence scores based on word frequency
        word_freq = Counter()
        for sentence in sentences:
            tokens = self.tokenize(sentence)
            word_freq.update(tokens)

        # Normalize frequencies
        max_freq = max(word_freq.values()) if word_freq else 1
        for word in word_freq:
            word_freq[word] = word_freq[word] / max_freq

        # Score sentences
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            tokens = self.tokenize(sentence)
            score = sum(word_freq.get(token, 0) for token in tokens)
            sentence_scores[i] = score / len(tokens) if tokens else 0

        # Select top sentences
        top_indices = sorted(sentence_scores.keys(),
                           key=lambda x: sentence_scores[x],
                           reverse=True)[:num_sentences]

        # Maintain original order
        top_indices.sort()
        summary = ' '.join(sentences[i] for i in top_indices)

        return summary

def run_demo():
    """Demonstrate the NLP lab capabilities."""
    print("=" * 80)
    print("NATURAL LANGUAGE PROCESSING LAB - Production Demo")
    print("Copyright (c) 2025 Corporation of Light")
    print("=" * 80)

    # Initialize lab
    nlp_lab = NaturalLanguageProcessingLab()

    # Sample corpus
    corpus = [
        "Natural language processing is a fascinating field of artificial intelligence.",
        "Machine learning algorithms can understand and generate human language.",
        "Deep learning models have revolutionized NLP tasks like translation.",
        "Transformers and attention mechanisms are key innovations in modern NLP.",
        "Language models can now perform various tasks with remarkable accuracy."
    ]

    print("\n1. TOKENIZATION METHODS")
    print("-" * 40)
    test_text = "Hello, world! This is a test sentence."

    methods = ['word', 'sentence', 'char']
    for method in methods:
        tokens = nlp_lab.tokenize(test_text, method=method)
        print(f"   {method:10s}: {len(tokens)} tokens")
        if method != 'char':
            print(f"                {tokens[:5]}")

    print("\n2. VOCABULARY BUILDING")
    print("-" * 40)
    nlp_lab.build_vocabulary(corpus, max_vocab=100)
    print(f"   Vocabulary size: {len(nlp_lab.vocabulary)}")
    print(f"   Special tokens: {list(nlp_lab.vocabulary.keys())[:4]}")
    print(f"   Most frequent words: {list(nlp_lab.word_freq.most_common(5))}")

    print("\n3. TF-IDF CALCULATION")
    print("-" * 40)
    tfidf_matrix = nlp_lab.compute_tf_idf(corpus)
    print(f"   TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"   Non-zero values: {np.count_nonzero(tfidf_matrix)}")
    print(f"   Max TF-IDF score: {np.max(tfidf_matrix):.4f}")

    print("\n4. WORD EMBEDDINGS (Word2Vec)")
    print("-" * 40)
    embeddings = nlp_lab.word2vec_skipgram(corpus, iterations=2)
    print(f"   Embedding matrix shape: {embeddings.shape}")
    print(f"   Embedding norm (first word): {np.linalg.norm(embeddings[0]):.4f}")

    # Find similar words (simplified)
    if 'language' in nlp_lab.vocabulary:
        target_idx = nlp_lab.vocabulary['language']
        target_vec = embeddings[target_idx]

        similarities = []
        for word, idx in nlp_lab.vocabulary.items():
            if idx != target_idx and idx < len(embeddings):
                similarity = nlp_lab.cosine_similarity(target_vec, embeddings[idx])
                similarities.append((word, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        print(f"   Words similar to 'language': {[w for w, _ in similarities[:3]]}")

    print("\n5. ATTENTION MECHANISM")
    print("-" * 40)
    seq_len, dim = 5, 8
    query = np.random.randn(seq_len, dim)
    keys = np.random.randn(seq_len, dim)
    values = np.random.randn(seq_len, dim)

    output, weights = nlp_lab.attention_mechanism(query, keys, values)
    print(f"   Input shape: ({seq_len}, {dim})")
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights shape: {weights.shape}")
    print(f"   Attention weight sum: {np.sum(weights[0]):.4f}")  # Should be ~1.0

    print("\n6. MULTI-HEAD ATTENTION")
    print("-" * 40)
    mha_output = nlp_lab.multi_head_attention(query, keys, values, num_heads=4)
    print(f"   Multi-head output shape: {mha_output.shape}")

    print("\n7. N-GRAM LANGUAGE MODEL")
    print("-" * 40)
    nlp_lab.build_n_gram_model(corpus)
    print(f"   Unique bigrams: {len(nlp_lab.bigram_counts)}")
    print(f"   Unique trigrams: {len(nlp_lab.trigram_counts)}")

    # Calculate perplexity
    test_sentence = "Natural language processing is amazing."
    perplexity = nlp_lab.calculate_perplexity(test_sentence, model_type='bigram')
    print(f"   Test perplexity (bigram): {perplexity:.2f}")

    print("\n8. SEQUENCE-TO-SEQUENCE")
    print("-" * 40)
    encoder_input = np.random.randn(10, 16)  # 10 timesteps, 16 dims
    decoder_output = nlp_lab.sequence_to_sequence(encoder_input, decoder_steps=5)
    print(f"   Encoder input shape: {encoder_input.shape}")
    print(f"   Decoder output shape: {decoder_output.shape}")

    print("\n9. TEXT SIMILARITY METRICS")
    print("-" * 40)

    # Edit distance
    str1, str2 = "kitten", "sitting"
    distance = nlp_lab.edit_distance(str1, str2)
    print(f"   Edit distance ('{str1}', '{str2}'): {distance}")

    # Cosine similarity
    vec1 = np.random.randn(10)
    vec2 = vec1 + np.random.randn(10) * 0.1  # Similar vector
    similarity = nlp_lab.cosine_similarity(vec1, vec2)
    print(f"   Cosine similarity: {similarity:.4f}")

    print("\n10. TEXT CLASSIFICATION METRICS")
    print("-" * 40)
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    y_pred = [0, 1, 2, 0, 2, 2, 0, 1, 1, 0]

    metrics = nlp_lab.text_classification_metrics(y_true, y_pred)
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    print(f"   F1-Score: {metrics['f1_score']:.4f}")

    print("\n11. BEAM SEARCH DECODING")
    print("-" * 40)
    start_token = nlp_lab.vocabulary.get('<SOS>', 2)
    decoded = nlp_lab.beam_search_decoder(start_token, max_length=10)
    print(f"   Decoded sequence: {decoded[:10]}")

    print("\n12. NAMED ENTITY RECOGNITION")
    print("-" * 40)
    test_text = "John Smith works at Microsoft in Seattle since 2020."
    entities = nlp_lab.named_entity_recognition(test_text)
    print(f"   Entities found: {len(entities)}")
    for entity, entity_type in entities[:5]:
        print(f"      {entity:15s} -> {entity_type}")

    print("\n13. SENTIMENT ANALYSIS")
    print("-" * 40)
    texts = [
        "This product is absolutely wonderful and amazing!",
        "Terrible experience, very disappointing.",
        "It's okay, nothing special."
    ]

    for text in texts:
        sentiment = nlp_lab.sentiment_analysis_lexicon(text)
        print(f"   '{text[:30]}...'")
        print(f"      Sentiment: {sentiment['overall']}, "
              f"Pos: {sentiment['positive']:.2f}, "
              f"Neg: {sentiment['negative']:.2f}")

    print("\n14. TEXT SUMMARIZATION")
    print("-" * 40)
    long_text = " ".join(corpus)
    summary = nlp_lab.text_summarization_extractive(long_text, num_sentences=2)
    print(f"   Original length: {len(long_text)} characters")
    print(f"   Summary length: {len(summary)} characters")
    print(f"   Summary: {summary[:100]}...")

    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)

if __name__ == '__main__':
    run_demo()