import numpy as np
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("ecology_lab", "ecology_lab.py")
ecology_lab = importlib.util.module_from_spec(spec)
sys.modules["ecology_lab"] = ecology_lab
spec.loader.exec_module(ecology_lab)

lab = ecology_lab.EcologyLab()
np.random.seed(42)
landscape = np.random.choice([0, 1], size=(50, 50), p=[0.7, 0.3])
res = lab.habitat_fragmentation_analysis(landscape)
print(res)
