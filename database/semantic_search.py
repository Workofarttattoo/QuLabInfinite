"""
Semantic Results Search
Vector-based similarity search for results across labs
"""

from typing import List, Dict, Optional
import json
import logging
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class SemanticResultsSearch:
    """
    Enable semantic search across experiment results.

    Allows finding similar experiments based on:
    - Parameter values
    - Result outputs
    - Lab domain
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic search.

        Args:
            model_name: SentenceTransformer model name
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers not available")
            self.embedding_model = None
            return

        try:
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"✓ Loaded embedding model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            self.embedding_model = None

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None
        """
        if not self.embedding_model:
            return None

        try:
            embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        except Exception as e:
            logger.warning(f"Error generating embedding: {e}")
            return None

    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score (0-1)
        """
        if not embedding1 or not embedding2:
            return 0.0

        try:
            import numpy as np
            v1 = np.array(embedding1)
            v2 = np.array(embedding2)

            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0

            similarity = dot_product / (norm_v1 * norm_v2)
            return float(similarity)

        except Exception as e:
            logger.warning(f"Error computing similarity: {e}")
            return 0.0

    def find_similar_results(
        self,
        query_result: Dict,
        all_results: List[Dict],
        top_k: int = 10,
        similarity_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Find similar results to a given result.

        Args:
            query_result: Reference result
            all_results: Pool of results to search
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score

        Returns:
            List of similar results with similarity scores
        """
        if not self.embedding_model:
            logger.warning("Embedding model not available")
            return []

        # Generate embeddings
        query_text = self._result_to_text(query_result)
        query_embedding = self.generate_embedding(query_text)

        if not query_embedding:
            return []

        # Score all results
        scored_results = []

        for result in all_results:
            if result.get("task_id") == query_result.get("task_id"):
                continue  # Skip the query result itself

            result_text = self._result_to_text(result)
            result_embedding = self.generate_embedding(result_text)

            if not result_embedding:
                continue

            similarity = self.compute_similarity(query_embedding, result_embedding)

            if similarity >= similarity_threshold:
                scored_results.append({
                    **result,
                    "similarity_score": similarity
                })

        # Sort by similarity descending and return top_k
        scored_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored_results[:top_k]

    def _result_to_text(self, result: Dict) -> str:
        """Convert result to text for embedding"""
        parts = []

        # Add lab name
        if "lab_name" in result:
            parts.append(f"Lab: {result['lab_name']}")

        # Add parameters (key values)
        if "parameters" in result:
            params = result["parameters"]
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except:
                    pass

            if isinstance(params, dict):
                param_strs = [f"{k}={v}" for k, v in list(params.items())[:5]]
                parts.append(f"Parameters: {', '.join(param_strs)}")

        # Add result summary (key outputs)
        if "result" in result:
            res = result["result"]
            if isinstance(res, str):
                try:
                    res = json.loads(res)
                except:
                    pass

            if isinstance(res, dict):
                res_strs = [f"{k}={v}" for k, v in list(res.items())[:5]]
                parts.append(f"Results: {', '.join(res_strs)}")

        return " ".join(parts)

    def cluster_similar_results(
        self,
        results: List[Dict],
        num_clusters: int = 5
    ) -> Dict[int, List[Dict]]:
        """
        Cluster results based on similarity.

        Args:
            results: List of results to cluster
            num_clusters: Number of clusters

        Returns:
            Dictionary mapping cluster ID to results
        """
        if not self.embedding_model:
            logger.warning("Embedding model not available")
            return {0: results}

        try:
            import numpy as np
            from sklearn.cluster import KMeans

            # Generate embeddings for all results
            embeddings = []
            valid_results = []

            for result in results:
                text = self._result_to_text(result)
                embedding = self.generate_embedding(text)

                if embedding:
                    embeddings.append(embedding)
                    valid_results.append(result)

            if not embeddings:
                return {0: results}

            # Cluster
            embeddings_array = np.array(embeddings)
            kmeans = KMeans(
                n_clusters=min(num_clusters, len(valid_results)),
                random_state=42
            )
            labels = kmeans.fit_predict(embeddings_array)

            # Group results by cluster
            clusters = {}
            for label, result in zip(labels, valid_results):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(result)

            return clusters

        except ImportError:
            logger.warning("scikit-learn not available for clustering")
            return {0: results}
        except Exception as e:
            logger.warning(f"Error clustering results: {e}")
            return {0: results}
