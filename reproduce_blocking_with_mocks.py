import sys
import asyncio
import time
from unittest.mock import MagicMock

# --- MOCKS START ---
# We need to mock these BEFORE importing the target module

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

# --- MOCKS END ---

# Now we can import the module
# We need to add current dir to path
sys.path.append('.')
import genetic_variant_analyzer_api
from genetic_variant_analyzer_api import analyze_batch_endpoint, BatchVariantRequest, VariantRequest, VariantType

# Create a large batch
BATCH_SIZE = 50000
# Increased batch size to ensure significant blocking.
# 5000 might be too fast if the operations are just lookups.
# The hash operation is fast.

variants = [
    VariantRequest(
        gene=f"GENE{i}",
        chromosome="chr1",
        position=i,
        ref_allele="A",
        alt_allele="T",
        variant_type=VariantType.SNP,
        rsid=f"rs{i}",
        genotype="0/1"
    )
    for i in range(BATCH_SIZE)
]
request = BatchVariantRequest(variants=variants)

class Monitor:
    def __init__(self):
        self.max_delay = 0
        self.running = True

async def heartbeat(monitor):
    last_time = time.time()
    while monitor.running:
        await asyncio.sleep(0.01)
        current_time = time.time()
        # Expected time diff is 0.01
        delay = current_time - last_time - 0.01
        if delay > monitor.max_delay:
            monitor.max_delay = delay
        last_time = current_time

async def run_benchmark():
    print(f"Preparing to process {BATCH_SIZE} variants...")
    monitor = Monitor()

    # Start heartbeat
    heartbeat_task = asyncio.create_task(heartbeat(monitor))

    # Allow heartbeat to start
    await asyncio.sleep(0.05)

    start_time = time.time()
    print("Calling endpoint...")
    await analyze_batch_endpoint(request)
    end_time = time.time()

    monitor.running = False
    await heartbeat_task

    print(f"Total processing time: {end_time - start_time:.4f}s")
    print(f"Max event loop delay: {monitor.max_delay:.4f}s")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
