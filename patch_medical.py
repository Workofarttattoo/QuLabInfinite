with open("qulab/labs/medical/drug_interaction.py", "r") as f:
    content = f.read()

import re

# Insert _RISK_LEVEL_INDICES
idx = content.find("class RiskLevel(Enum):")
if idx == -1: print("ERR"); sys.exit(1)

# Find end of class
end_idx = content.find("class CYP450Enzyme(Enum):")

new_class = """class RiskLevel(Enum):
    \"\"\"Risk classification\"\"\"
    SAFE = "safe"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

_RISK_LEVEL_INDICES = {r: i for i, r in enumerate(RiskLevel)}

"""

content = content[:idx] + new_class + content[end_idx:]

content = content.replace(
    "max_risk = max([p.risk_level for p in pairwise], key=lambda r: list(RiskLevel).index(r))",
    "max_risk = max([p.risk_level for p in pairwise], key=lambda r: _RISK_LEVEL_INDICES[r])"
)

with open("qulab/labs/medical/drug_interaction.py", "w") as f:
    f.write(content)
