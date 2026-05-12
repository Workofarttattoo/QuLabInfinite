import sys
import os
sys.path.append(os.getcwd())
import time
from qulab.labs.medical.drug_interaction import DrugInteractionAnalyzer, InteractionType, RiskLevel, DRUG_DATABASE

def test():
    analyzer = DrugInteractionAnalyzer()
    drugs = list(DRUG_DATABASE.keys())

    # Pre-warm
    result = analyzer.analyze_network(drugs)

    start_time = time.time()
    for _ in range(100):
        analyzer.analyze_network(drugs)
    end_time = time.time()

    print(f"Time taken: {end_time - start_time:.4f}s")

test()
