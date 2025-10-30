
"""
Simple uncertainty quantification helpers: conformal regression & MC-dropout-like shim.
These are lightweight stubs; plug in your model APIs where noted.
"""
import math
from typing import List, Tuple

def conformal_interval(residuals: List[float], alpha: float = 0.1) -> float:
    """
    Return the quantile radius for absolute residuals.
    """
    if not residuals:
        return float("nan")
    abs_res = sorted(abs(r) for r in residuals)
    k = int(math.ceil((1 - alpha) * (len(abs_res) + 1))) - 1
    k = max(0, min(k, len(abs_res) - 1))
    return abs_res[k]

def mc_dropout_like(pred_fn, x, T: int = 16) -> Tuple[float, float]:
    """
    Call `pred_fn(x, train_mode=True)` T times and return (mean, std).
    Your `pred_fn` should respect train_mode and introduce stochasticity (e.g., dropout).
    """
    preds = []
    for _ in range(T):
        preds.append(float(pred_fn(x, train_mode=True)))
    mu = sum(preds) / len(preds)
    var = sum((p - mu)**2 for p in preds) / max(1, len(preds) - 1)
    return mu, math.sqrt(var)
