import time
from enum import Enum

class RiskLevel(Enum):
    SAFE = "safe"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

_RISK_LEVEL_INDICES = {r: i for i, r in enumerate(RiskLevel)}

class P:
    def __init__(self, r):
        self.risk_level = r

pairwise = [P(RiskLevel.LOW), P(RiskLevel.HIGH), P(RiskLevel.SAFE), P(RiskLevel.CRITICAL), P(RiskLevel.MODERATE)] * 100

start = time.perf_counter()
for _ in range(1000):
    max_risk = max([p.risk_level for p in pairwise], key=lambda r: list(RiskLevel).index(r))
t1 = time.perf_counter() - start

start = time.perf_counter()
for _ in range(1000):
    max_risk = max([p.risk_level for p in pairwise], key=lambda r: _RISK_LEVEL_INDICES[r])
t2 = time.perf_counter() - start

print(f"Original: {t1:.4f}s")
print(f"Optimized: {t2:.4f}s")
print(f"Speedup: {t1/t2:.2f}x")
