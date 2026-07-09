import sys
import os
import types

mock_np = types.ModuleType('numpy')
mock_np.float64 = float
mock_np.array = lambda x, **kwargs: x
mock_np.ndarray = list
sys.modules['numpy'] = mock_np

mock_scipy = types.ModuleType('scipy')
mock_scipy.constants = types.ModuleType('constants')
mock_scipy.constants.k = 1.380649e-23
mock_scipy.constants.Avogadro = 6.02214076e23
mock_scipy.constants.g = 9.80665
mock_scipy.constants.c = 299792458
mock_scipy.constants.h = 6.62607015e-34
mock_scipy.constants.e = 1.602176634e-19
mock_scipy.constants.pi = 3.141592653589793
mock_scipy.constants.physical_constants = {
    "alanine mass": (1.0, "u", 0.0),
    "arginine mass": (1.0, "u", 0.0),
    "asparagine mass": (1.0, "u", 0.0),
    "aspartic acid mass": (1.0, "u", 0.0),
    "cysteine mass": (1.0, "u", 0.0),
    "glutamic acid mass": (1.0, "u", 0.0),
    "glutamine mass": (1.0, "u", 0.0),
    "glycine mass": (1.0, "u", 0.0),
    "histidine mass": (1.0, "u", 0.0),
    "isoleucine mass": (1.0, "u", 0.0),
    "leucine mass": (1.0, "u", 0.0),
    "lysine mass": (1.0, "u", 0.0),
    "methionine mass": (1.0, "u", 0.0),
    "phenylalanine mass": (1.0, "u", 0.0),
    "proline mass": (1.0, "u", 0.0),
    "serine mass": (1.0, "u", 0.0),
    "threonine mass": (1.0, "u", 0.0),
    "tryptophan mass": (1.0, "u", 0.0),
    "tyrosine mass": (1.0, "u", 0.0),
    "valine mass": (1.0, "u", 0.0)
}
sys.modules['scipy'] = mock_scipy
sys.modules['scipy.constants'] = mock_scipy.constants

sys.path.append(os.getcwd())
import time

try:
    from qulab.labs.biology.bioinformatics import BioinformaticsLab

    # Pre-warm
    lab = BioinformaticsLab("ATCGTAGC")
    lab.analyze_sequence("ATCGTACGAAAAGGGGGTTTTTTCCCCCCC")

    start_time = time.time()
    for _ in range(1000):
        lab.analyze_sequence("ATCGTACGAAAAGGGGGTTTTTTCCCCCCC" * 100)
    end_time = time.time()

    print(f"Time taken: {end_time - start_time:.4f}s")

except Exception as e:
    import traceback
    traceback.print_exc()
    print(type(e), e)
