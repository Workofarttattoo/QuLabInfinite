import sys
import unittest

# Mock missing modules
class MockPandas:
    def __init__(self):
        pass
    def to_numeric(self, *args, **kwargs):
        pass
    class DataFrame:
        def __init__(self, *args, **kwargs):
            pass

class MockNumpy:
    pass

sys.modules['pandas'] = MockPandas()
sys.modules['numpy'] = MockNumpy()

import pandas as pd
pd.DataFrame = type('MockDataFrame', (), {})
pd.Series = type('MockSeries', (), {})
