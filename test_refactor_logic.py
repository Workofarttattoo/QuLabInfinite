import unittest
from unittest.mock import MagicMock
import sys

# Mocks
class MockBaseModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def MockField(default=..., **kwargs):
    return default

pydantic = MagicMock()
pydantic.BaseModel = MockBaseModel
pydantic.Field = MockField
sys.modules["pydantic"] = pydantic

fastapi = MagicMock()
def pass_through_decorator(func):
    return func
class MockFastAPI:
    def __init__(self, **kwargs): pass
    def get(self, *args, **kwargs): return pass_through_decorator
    def post(self, *args, **kwargs): return pass_through_decorator

fastapi.FastAPI = MockFastAPI
fastapi.HTTPException = Exception
sys.modules["fastapi"] = fastapi
sys.modules["fastapi.responses"] = MagicMock()
sys.modules["numpy"] = MagicMock()

import genetic_variant_analyzer_api
from genetic_variant_analyzer_api import process_batch_variants, VariantRequest, VariantType

class TestGeneticVariantAnalyzer(unittest.TestCase):
    def test_process_batch_variants(self):
        variants = [
            VariantRequest(
                gene="BRCA1",
                chromosome="chr17",
                position=43044295,
                ref_allele="AG",
                alt_allele="A",
                variant_type=VariantType.DELETION,
                rsid="rs80357906",
                genotype="0/1"
            )
        ]

        results = process_batch_variants(variants)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["gene"], "BRCA1")
        self.assertEqual(results[0]["rsid"], "rs80357906")
        self.assertIn("clinical_significance", results[0])

if __name__ == '__main__':
    unittest.main()
