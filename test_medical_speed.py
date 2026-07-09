import sys
import os
import time
sys.path.append(os.getcwd())
from qulab.labs.medical.drug_interaction import DrugInteractionAnalyzer, DRUG_DATABASE

def test():
    analyzer = DrugInteractionAnalyzer()
    drugs = list(DRUG_DATABASE.keys())

    start_time = time.time()
    for _ in range(200):
        analyzer.analyze_network(drugs)
    end_time = time.time()

    print(f"Time taken: {end_time - start_time:.4f}s")

test()
