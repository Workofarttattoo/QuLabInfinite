import asyncio
import time
import sys
import os

# Make sure we can import the api file
sys.path.append('.')

from genetic_variant_analyzer_api import analyze_batch_endpoint, BatchVariantRequest, VariantRequest, VariantType

# Create a large batch
BATCH_SIZE = 20000
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
