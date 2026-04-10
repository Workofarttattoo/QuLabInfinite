import pytest
import time
import asyncio
from fastapi.testclient import TestClient
import httpx
from httpx import ASGITransport
from materials_api import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

@pytest.mark.asyncio
async def test_concurrent_performance():
    # We use httpx AsyncClient connected directly to the ASGI app
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        start_time = time.time()

        tasks = []
        for _ in range(50):
            # Using formula search to trigger full table scan
            tasks.append(ac.get("/search?formula=Fe&limit=1000"))

        responses = await asyncio.gather(*tasks)

        end_time = time.time()
        duration = end_time - start_time

        # Verify all responses succeeded
        for r in responses:
            assert r.status_code == 200

        print(f"\nTime for 50 concurrent requests: {duration:.4f} seconds")
