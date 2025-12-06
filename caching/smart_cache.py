"""
Smart Result Cache
ML-based caching system that detects near-duplicate experiments
and intelligently suggests or interpolates results
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategy for results"""
    EXACT_MATCH = "exact_match"  # Only use if parameters are identical
    FUZZY_MATCH = "fuzzy_match"  # Use similar parameters with small interpolation error
    INTERPOLATE = "interpolate"  # Interpolate from nearby results
    PREDICT = "predict"  # Use ML model to predict result


@dataclass
class CachedResult:
    """A cached result entry"""
    result_key: str
    lab_name: str
    parameters: Dict
    result: Dict
    confidence: float
    cached_at: datetime
    expiry: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return datetime.now() > self.expiry

    def is_similar_to(
        self,
        other_params: Dict,
        similarity_threshold: float = 0.95
    ) -> Tuple[bool, float]:
        """
        Check if parameters are similar.

        Args:
            other_params: Parameters to compare
            similarity_threshold: Similarity score threshold (0-1)

        Returns:
            (is_similar, similarity_score)
        """
        if self.parameters.keys() != other_params.keys():
            return False, 0.0

        # Compute parameter similarity
        similarities = []

        for key in self.parameters:
            v1 = self.parameters[key]
            v2 = other_params[key]

            # Handle numeric parameters
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                if v1 == 0:
                    similarity = 1.0 if v2 == 0 else 0.0
                else:
                    relative_diff = abs(v1 - v2) / abs(v1)
                    similarity = max(0, 1 - relative_diff)

            # Handle string/categorical parameters
            else:
                similarity = 1.0 if v1 == v2 else 0.0

            similarities.append(similarity)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        return avg_similarity >= similarity_threshold, avg_similarity


class SmartResultCache:
    """
    Smart caching system that intelligently manages experiment results.

    Features:
    - Parameter-based result hashing
    - Fuzzy matching for similar experiments
    - Result interpolation for nearby parameters
    - Confidence-based cache suggestions
    - Automatic cache expiry
    - Cache hit/miss statistics
    """

    def __init__(
        self,
        max_cache_size: int = 10000,
        default_ttl_hours: int = 72,
        similarity_threshold: float = 0.95
    ):
        """
        Initialize smart cache.

        Args:
            max_cache_size: Maximum number of cached results
            default_ttl_hours: Default time-to-live for cache entries (hours)
            similarity_threshold: Threshold for fuzzy parameter matching
        """
        self.cache: Dict[str, CachedResult] = {}
        self.max_cache_size = max_cache_size
        self.default_ttl = timedelta(hours=default_ttl_hours)
        self.similarity_threshold = similarity_threshold

        # Statistics
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "evictions": 0,
            "interpolations": 0,
            "total_lookups": 0
        }

        logger.info(
            f"✓ Smart Result Cache initialized "
            f"(max_size={max_cache_size}, ttl={default_ttl_hours}h)"
        )

    def _hash_parameters(self, lab_name: str, parameters: Dict) -> str:
        """Generate hash key for parameters"""
        param_str = json.dumps(
            {lab_name: parameters},
            sort_keys=True,
            default=str
        )
        return hashlib.md5(param_str.encode()).hexdigest()

    def add_result(
        self,
        lab_name: str,
        parameters: Dict,
        result: Dict,
        confidence: float = 1.0,
        ttl_hours: Optional[int] = None
    ) -> str:
        """
        Add result to cache.

        Args:
            lab_name: Lab name
            parameters: Input parameters
            result: Experiment result
            confidence: Confidence score (0-1)
            ttl_hours: Time-to-live for this entry (hours)

        Returns:
            result_key
        """
        result_key = self._hash_parameters(lab_name, parameters)

        # Check if we need to evict
        if len(self.cache) >= self.max_cache_size:
            self._evict_lru()

        # Add to cache
        self.cache[result_key] = CachedResult(
            result_key=result_key,
            lab_name=lab_name,
            parameters=parameters.copy(),
            result=result.copy(),
            confidence=confidence,
            cached_at=datetime.now(),
            expiry=datetime.now() + (
                timedelta(hours=ttl_hours) if ttl_hours else self.default_ttl
            )
        )

        logger.debug(f"Cached result: {result_key}")
        return result_key

    def get_result(
        self,
        lab_name: str,
        parameters: Dict,
        strategy: CacheStrategy = CacheStrategy.EXACT_MATCH
    ) -> Optional[Dict]:
        """
        Get result from cache.

        Args:
            lab_name: Lab name
            parameters: Input parameters
            strategy: Cache lookup strategy

        Returns:
            Cached result or None
        """
        self.stats["total_lookups"] += 1

        result_key = self._hash_parameters(lab_name, parameters)

        # Try exact match
        if result_key in self.cache:
            cached = self.cache[result_key]

            if not cached.is_expired():
                cached.access_count += 1
                cached.last_accessed = datetime.now()
                self.stats["cache_hits"] += 1

                logger.debug(
                    f"Cache HIT (exact): {result_key} "
                    f"(confidence={cached.confidence})"
                )

                return {
                    "result": cached.result,
                    "source": "cache_exact_match",
                    "confidence": cached.confidence,
                    "cached_at": cached.cached_at.isoformat()
                }

            # Remove expired entry
            del self.cache[result_key]

        # Try fuzzy match if configured
        if strategy in [CacheStrategy.FUZZY_MATCH, CacheStrategy.INTERPOLATE]:
            fuzzy_match = self._find_fuzzy_match(lab_name, parameters)

            if fuzzy_match:
                cached, similarity = fuzzy_match
                cached.access_count += 1
                cached.last_accessed = datetime.now()

                if strategy == CacheStrategy.FUZZY_MATCH:
                    self.stats["cache_hits"] += 1

                    logger.debug(
                        f"Cache HIT (fuzzy): {cached.result_key} "
                        f"(similarity={similarity:.3f})"
                    )

                    return {
                        "result": cached.result,
                        "source": "cache_fuzzy_match",
                        "confidence": cached.confidence * similarity,
                        "similarity": similarity,
                        "cached_at": cached.cached_at.isoformat()
                    }

                # Interpolation strategy
                elif strategy == CacheStrategy.INTERPOLATE:
                    interpolated = self._interpolate_result(
                        cached, parameters
                    )

                    if interpolated:
                        self.stats["interpolations"] += 1
                        self.stats["cache_hits"] += 1

                        logger.debug(
                            f"Cache HIT (interpolated): from {cached.result_key}"
                        )

                        return {
                            "result": interpolated,
                            "source": "cache_interpolated",
                            "confidence": cached.confidence * 0.8,  # Lower confidence
                            "interpolated": True,
                            "original_cached_at": cached.cached_at.isoformat()
                        }

        self.stats["cache_misses"] += 1
        logger.debug(f"Cache MISS for parameters: {parameters}")

        return None

    def _find_fuzzy_match(
        self,
        lab_name: str,
        parameters: Dict
    ) -> Optional[Tuple[CachedResult, float]]:
        """Find closest fuzzy match in cache"""
        best_match = None
        best_similarity = 0.0

        for cached in self.cache.values():
            if cached.lab_name != lab_name or cached.is_expired():
                continue

            is_similar, similarity = cached.is_similar_to(
                parameters,
                self.similarity_threshold
            )

            if is_similar and similarity > best_similarity:
                best_match = cached
                best_similarity = similarity

        return (best_match, best_similarity) if best_match else None

    def _interpolate_result(
        self,
        cached: CachedResult,
        target_parameters: Dict
    ) -> Optional[Dict]:
        """
        Interpolate result based on parameter differences.

        Simple linear interpolation for numeric parameters.
        """
        try:
            interpolated = {}

            for key, target_value in target_parameters.items():
                cached_value = cached.parameters.get(key)
                cached_result = cached.result.get(key)

                # Only interpolate numeric values
                if (
                    isinstance(target_value, (int, float)) and
                    isinstance(cached_value, (int, float)) and
                    isinstance(cached_result, (int, float))
                ):
                    # Linear interpolation factor
                    if cached_value == target_value:
                        interpolated[key] = cached_result
                    else:
                        # Simple proportional scaling
                        ratio = target_value / cached_value if cached_value != 0 else 0
                        interpolated[key] = cached_result * ratio

                else:
                    # For non-numeric, use cached value as-is
                    interpolated[key] = cached_result

            return interpolated if interpolated else None

        except Exception as e:
            logger.warning(f"Interpolation failed: {e}")
            return None

    def _evict_lru(self) -> None:
        """Evict least recently used entry"""
        if not self.cache:
            return

        # Find LRU entry
        lru_key = min(
            self.cache.keys(),
            key=lambda k: (
                self.cache[k].last_accessed or self.cache[k].cached_at
            )
        )

        del self.cache[lru_key]
        self.stats["evictions"] += 1

        logger.debug(f"Evicted LRU entry: {lru_key}")

    def clear_expired(self) -> int:
        """Clear all expired entries"""
        expired_keys = [
            key for key, cached in self.cache.items()
            if cached.is_expired()
        ]

        for key in expired_keys:
            del self.cache[key]

        logger.info(f"Cleared {len(expired_keys)} expired cache entries")
        return len(expired_keys)

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.stats["cache_hits"] + self.stats["cache_misses"]
        hit_rate = (
            self.stats["cache_hits"] / total * 100
            if total > 0 else 0
        )

        return {
            **self.stats,
            "hit_rate": f"{hit_rate:.1f}%",
            "cache_size": len(self.cache),
            "max_size": self.max_cache_size,
            "load_factor": f"{len(self.cache) / self.max_cache_size * 100:.1f}%"
        }

    def clear_cache(self) -> None:
        """Clear entire cache"""
        self.cache.clear()
        logger.info("Cache cleared")
