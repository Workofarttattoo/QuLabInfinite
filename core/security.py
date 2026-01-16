import os
from typing import List, Sequence

# Blocked keys that should never be used in production
DEFAULT_BLOCKLIST = {
    "demo_key_12345",
    "pro_key_67890",
    "enterprise_key_abcde",
    "qulab_master_key_2025",
    "qulab_demo_key",
}

# Words that typically indicate a placeholder or sample secret
PLACEHOLDER_KEYWORDS = ("changeme", "example", "sample", "demo", "test", "placeholder")


def _contains_placeholder(key: str, placeholder_keywords: Sequence[str]) -> bool:
    """Check if a key contains placeholder keywords."""
    lower_key = key.lower()
    return any(word in lower_key for word in placeholder_keywords)


def load_api_keys_from_env(
    env_var: str = "QU_LAB_MASTER_KEYS",
    *,
    blocklist: Sequence[str] = DEFAULT_BLOCKLIST,
    placeholder_keywords: Sequence[str] = PLACEHOLDER_KEYWORDS,
) -> List[str]:
    """
    Load API keys from an environment variable and validate them.

    The environment variable should contain a comma-separated list of API keys.
    Raises a RuntimeError if the variable is unset, empty, or includes default/placeholder keys.
    """
    raw_keys = os.getenv(env_var, "")
    if not raw_keys:
        raise RuntimeError(
            f"{env_var} must be set to a comma-separated list of secure API keys; "
            "startup aborted because no keys were provided."
        )

    keys = [key.strip() for key in raw_keys.split(",") if key.strip()]
    if not keys:
        raise RuntimeError(
            f"{env_var} was provided but did not contain any usable keys; "
            "please supply at least one non-empty key."
        )

    invalid_keys = [
        key
        for key in keys
        if key in blocklist or _contains_placeholder(key, placeholder_keywords)
    ]
    if invalid_keys:
        raise RuntimeError(
            f"{env_var} includes default or placeholder keys ({', '.join(invalid_keys)}); "
            "replace them with unique, secret values before starting the service."
        )

    return keys
