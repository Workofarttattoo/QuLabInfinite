import sys
import os
import types
import time
sys.path.append(os.getcwd())

mock_np = types.ModuleType('numpy')
mock_np.float64 = float
mock_np.array = lambda x, **kwargs: x
mock_np.ndarray = list
mock_np.zeros = lambda shape: [[0]*shape[1] for _ in range(shape[0])] if isinstance(shape, tuple) else [0]*shape
mock_np.log = lambda x: x
mock_np.exp = lambda x: x
sys.modules['numpy'] = mock_np

# Create mock scipy that linprog uses
mock_scipy = types.ModuleType('scipy')
mock_scipy.optimize = types.ModuleType('optimize')
class LinprogResult:
    def __init__(self):
        self.success = True
        self.x = [0]*10
        self.fun = 10.0
mock_scipy.optimize.linprog = lambda *args, **kwargs: LinprogResult()
sys.modules['scipy'] = mock_scipy
sys.modules['scipy.optimize'] = mock_scipy.optimize

from qulab.labs.biology.metabolomics_lab.metabolomics_engine import MetabolomicsEngine

engine = MetabolomicsEngine()

# Monkey patch np zeros since our naive mock is bad
def dummy_zeros(shape):
    if isinstance(shape, tuple):
        class MatrixMock:
            def __init__(self):
                self.data = {}
            def __setitem__(self, key, val):
                self.data[key] = val
            def __getitem__(self, key):
                return self.data.get(key, 0)
            def tolist(self):
                return []
        return MatrixMock()
    return [0]*shape

mock_np.zeros = dummy_zeros

try:
    # Pre-warm
    engine.flux_balance_analysis('PK')

    start_time = time.time()
    for _ in range(10000):
        engine.flux_balance_analysis('PK')
    end_time = time.time()

    print(f"Time taken: {end_time - start_time:.4f}s")
except Exception as e:
    print(e)
