
"""
Sprint 3: minimal spectra/XRD encoder + contrastive alignment scaffolding.
Replace the dummy encoders with your preferred framework.
"""
from typing import List
import math

def encode_curve(x:List[float], y:List[float]) -> List[float]:
    # Very small handcrafted features: peaks count, centroid, spread, roughness.
    if not x or not y or len(x) != len(y):
        return [0.0, 0.0, 0.0, 0.0]
    n = len(x)
    s = sum(y)
    cx = sum(xi*yi for xi, yi in zip(x, y)) / (s if s else 1.0)
    var = sum(((xi - cx)**2)*yi for xi, yi in zip(x, y)) / (s if s else 1.0)
    rough = sum(abs(y[i]-y[i-1]) for i in range(1, n)) / n
    # Dummy peak count by gradient sign changes
    peaks = sum(1 for i in range(1, n-1) if y[i] > y[i-1] and y[i] > y[i+1])
    return [float(peaks), float(cx), float(var), float(rough)]

def encode_text(s:str) -> List[float]:
    # Silly bag-of-characters norm — replace with a real text encoder when training.
    return [float(len(s)), float(sum(map(ord, s)) % 997) / 997.0]

def contrastive_score(a:List[float], b:List[float]) -> float:
    # Cosine similarity
    da = math.sqrt(sum(x*x for x in a)) or 1.0
    db = math.sqrt(sum(x*x for x in b)) or 1.0
    dot = sum(x*y for x, y in zip(a, b))
    return dot/(da*db)

def demo_align(x:List[float], y:List[float], caption:str) -> float:
    v_curve = encode_curve(x, y)
    v_text  = encode_text(caption)
    return contrastive_score(v_curve, v_text)
