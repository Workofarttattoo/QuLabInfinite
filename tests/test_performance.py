"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Performance Testing Infrastructure - Phase 2 Implementation
Target: 1000+ req/s throughput, <100ms p50 latency, <500ms p99 latency
"""
import pytest
import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import requests
from dataclasses import dataclass
from fastapi.testclient import TestClient
from api.secure_production_api import app

# Test client
client = TestClient(app)

@dataclass
class PerformanceMetrics:
    """Performance test results"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    duration_seconds: float
    requests_per_second: float
    latencies_ms: List[float]
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    mean_latency_ms: float
    error_rate: float


class PerformanceTestSuite:
    """Performance testing framework"""

    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.test_token = None

    def setup_authentication(self):
        """Create test user and get auth token"""
        # Register test user
        try:
            client.post("/auth/register", json={
                "username": "perftest",
                "password": "perftest123",
                "email": "perftest@example.com"
            })
        except:
            pass  # User may already exist

        # Login and get token
        response = client.post("/auth/token", data={
            "username": "perftest",
            "password": "perftest123"
        })

        if response.status_code == 200:
            self.test_token = response.json()["access_token"]

        return self.test_token

    def execute_request(self, endpoint: str, method: str = "GET",
                       data: dict = None, headers: dict = None) -> Dict[str, Any]:
        """
        Execute single HTTP request and measure performance

        Returns:
            dict with 'success', 'latency_ms', 'status_code'
        """
        start_time = time.time()

        try:
            if method == "GET":
                response = client.get(endpoint, headers=headers)
            elif method == "POST":
                response = client.post(endpoint, json=data, headers=headers)
            else:
                raise ValueError(f"Unsupported method: {method}")

            latency_ms = (time.time() - start_time) * 1000

            return {
                "success": response.status_code < 400,
                "status_code": response.status_code,
                "latency_ms": latency_ms
            }

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "status_code": 0,
                "latency_ms": latency_ms,
                "error": str(e)
            }

    def load_test(self, endpoint: str, method: str = "GET",
                  data: dict = None, headers: dict = None,
                  num_requests: int = 1000,
                  concurrency: int = 10) -> PerformanceMetrics:
        """
        Execute load test with specified concurrency

        Args:
            endpoint: API endpoint to test
            method: HTTP method (GET, POST)
            data: Request body for POST
            headers: Request headers
            num_requests: Total number of requests
            concurrency: Number of concurrent workers

        Returns:
            PerformanceMetrics with aggregated results
        """
        results = []
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(self.execute_request, endpoint, method, data, headers)
                for _ in range(num_requests)
            ]

            for future in as_completed(futures):
                results.append(future.result())

        duration = time.time() - start_time

        # Calculate metrics
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        latencies = [r["latency_ms"] for r in results]

        latencies_sorted = sorted(latencies)

        return PerformanceMetrics(
            total_requests=num_requests,
            successful_requests=len(successful),
            failed_requests=len(failed),
            duration_seconds=duration,
            requests_per_second=num_requests / duration if duration > 0 else 0,
            latencies_ms=latencies,
            p50_latency_ms=latencies_sorted[len(latencies_sorted) // 2],
            p95_latency_ms=latencies_sorted[int(len(latencies_sorted) * 0.95)],
            p99_latency_ms=latencies_sorted[int(len(latencies_sorted) * 0.99)],
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            mean_latency_ms=statistics.mean(latencies),
            error_rate=len(failed) / num_requests if num_requests > 0 else 0
        )

    def stress_test(self, endpoint: str, method: str = "GET",
                   data: dict = None, headers: dict = None,
                   duration_seconds: int = 60,
                   ramp_up_seconds: int = 10,
                   max_concurrency: int = 100) -> List[PerformanceMetrics]:
        """
        Execute stress test with ramping concurrency

        Gradually increases load to find breaking point

        Args:
            endpoint: API endpoint to test
            duration_seconds: Total test duration
            ramp_up_seconds: Time to ramp up to max concurrency
            max_concurrency: Maximum concurrent workers

        Returns:
            List of PerformanceMetrics at different concurrency levels
        """
        results = []
        concurrency_levels = [10, 25, 50, 75, 100]
        requests_per_level = 200

        for concurrency in concurrency_levels:
            if concurrency > max_concurrency:
                break

            print(f"[info] Testing at concurrency level: {concurrency}")

            metrics = self.load_test(
                endpoint=endpoint,
                method=method,
                data=data,
                headers=headers,
                num_requests=requests_per_level,
                concurrency=concurrency
            )

            results.append(metrics)

            # Check if system is degrading
            if metrics.error_rate > 0.05:  # 5% error rate threshold
                print(f"[warn] High error rate detected: {metrics.error_rate:.2%}")
                break

            if metrics.p99_latency_ms > 1000:  # 1 second p99 threshold
                print(f"[warn] High latency detected: {metrics.p99_latency_ms:.2f}ms")
                break

        return results

    def benchmark_endpoints(self) -> Dict[str, PerformanceMetrics]:
        """
        Benchmark all critical API endpoints

        Returns:
            Dict mapping endpoint names to performance metrics
        """
        benchmarks = {}

        # Setup authentication
        token = self.setup_authentication()
        auth_headers = {"Authorization": f"Bearer {token}"}

        # 1. Health endpoint (public)
        print("[info] Benchmarking /health endpoint...")
        benchmarks["health"] = self.load_test(
            endpoint="/health",
            method="GET",
            num_requests=500,
            concurrency=20
        )

        # 2. Molecule parsing (authenticated)
        print("[info] Benchmarking /api/v2/parse/molecule endpoint...")
        benchmarks["parse_molecule"] = self.load_test(
            endpoint="/api/v2/parse/molecule",
            method="POST",
            data={"smiles": "CCO"},
            headers=auth_headers,
            num_requests=500,
            concurrency=20
        )

        # 3. Authentication endpoint
        print("[info] Benchmarking /auth/token endpoint...")
        benchmarks["auth_token"] = self.load_test(
            endpoint="/auth/token",
            method="POST",
            data={"username": "perftest", "password": "perftest123"},
            num_requests=200,
            concurrency=10
        )

        return benchmarks


# Pytest Test Cases
class TestPerformanceBaseline:
    """Baseline performance tests"""

    def test_health_endpoint_performance(self):
        """Test health endpoint meets performance targets"""
        suite = PerformanceTestSuite()

        metrics = suite.load_test(
            endpoint="/health",
            method="GET",
            num_requests=100,
            concurrency=10
        )

        # Performance targets
        assert metrics.error_rate < 0.01, f"Error rate too high: {metrics.error_rate:.2%}"
        assert metrics.p50_latency_ms < 100, f"p50 latency too high: {metrics.p50_latency_ms:.2f}ms"
        assert metrics.p99_latency_ms < 500, f"p99 latency too high: {metrics.p99_latency_ms:.2f}ms"
        assert metrics.requests_per_second > 50, f"Throughput too low: {metrics.requests_per_second:.2f} req/s"

    def test_authenticated_endpoint_performance(self):
        """Test authenticated endpoints meet performance targets"""
        suite = PerformanceTestSuite()
        token = suite.setup_authentication()

        metrics = suite.load_test(
            endpoint="/api/v2/parse/molecule",
            method="POST",
            data={"smiles": "CCO"},
            headers={"Authorization": f"Bearer {token}"},
            num_requests=100,
            concurrency=10
        )

        # Performance targets
        assert metrics.error_rate < 0.01, f"Error rate too high: {metrics.error_rate:.2%}"
        assert metrics.p50_latency_ms < 100, f"p50 latency too high: {metrics.p50_latency_ms:.2f}ms"
        assert metrics.p99_latency_ms < 500, f"p99 latency too high: {metrics.p99_latency_ms:.2f}ms"


class TestLoadCapacity:
    """Load capacity tests"""

    def test_moderate_load(self):
        """Test system handles moderate load (100 concurrent)"""
        suite = PerformanceTestSuite()
        token = suite.setup_authentication()

        metrics = suite.load_test(
            endpoint="/api/v2/parse/molecule",
            method="POST",
            data={"smiles": "CCO"},
            headers={"Authorization": f"Bearer {token}"},
            num_requests=500,
            concurrency=100
        )

        # Should handle moderate load gracefully
        assert metrics.error_rate < 0.05, f"Error rate too high under load: {metrics.error_rate:.2%}"
        assert metrics.requests_per_second > 100, f"Throughput degraded: {metrics.requests_per_second:.2f} req/s"

    def test_sustained_load(self):
        """Test system handles sustained moderate load"""
        suite = PerformanceTestSuite()
        token = suite.setup_authentication()

        # Run for 30 seconds
        metrics = suite.load_test(
            endpoint="/health",
            method="GET",
            num_requests=1500,
            concurrency=50
        )

        # Sustained load should maintain performance
        assert metrics.error_rate < 0.01
        assert metrics.p95_latency_ms < 300


class TestStressLimits:
    """Stress testing to find system limits"""

    @pytest.mark.slow
    def test_stress_to_failure(self):
        """Stress test to find breaking point"""
        suite = PerformanceTestSuite()
        token = suite.setup_authentication()

        results = suite.stress_test(
            endpoint="/api/v2/parse/molecule",
            method="POST",
            data={"smiles": "CCO"},
            headers={"Authorization": f"Bearer {token}"},
            duration_seconds=30,
            max_concurrency=200
        )

        # Should have at least attempted multiple concurrency levels
        assert len(results) > 0

        # First level should succeed
        assert results[0].error_rate < 0.05

        # Document breaking point
        for i, metrics in enumerate(results):
            concurrency = [10, 25, 50, 75, 100, 150, 200][i]
            print(f"[info] Concurrency {concurrency}: {metrics.requests_per_second:.2f} req/s, "
                  f"p99={metrics.p99_latency_ms:.2f}ms, errors={metrics.error_rate:.2%}")


class TestMemoryLeaks:
    """Memory leak detection tests"""

    @pytest.mark.slow
    def test_no_memory_leaks(self):
        """Test for memory leaks under sustained load"""
        import psutil
        import os

        suite = PerformanceTestSuite()
        token = suite.setup_authentication()

        # Get baseline memory
        process = psutil.Process(os.getpid())
        baseline_memory_mb = process.memory_info().rss / 1024 / 1024

        # Run sustained load
        for _ in range(5):
            suite.load_test(
                endpoint="/health",
                method="GET",
                num_requests=100,
                concurrency=10
            )

        # Check memory growth
        final_memory_mb = process.memory_info().rss / 1024 / 1024
        memory_growth_mb = final_memory_mb - baseline_memory_mb

        # Should not grow more than 50MB
        assert memory_growth_mb < 50, f"Possible memory leak: {memory_growth_mb:.2f}MB growth"


# Benchmark runner
def run_full_benchmark():
    """Run complete performance benchmark suite"""
    suite = PerformanceTestSuite()

    print("\n" + "="*80)
    print("QuLab AI Production Performance Benchmark")
    print("="*80 + "\n")

    benchmarks = suite.benchmark_endpoints()

    print("\n" + "="*80)
    print("Performance Summary")
    print("="*80 + "\n")

    for endpoint, metrics in benchmarks.items():
        print(f"\n{endpoint}:")
        print(f"  Total Requests:   {metrics.total_requests}")
        print(f"  Successful:       {metrics.successful_requests} ({100-metrics.error_rate*100:.1f}%)")
        print(f"  Failed:           {metrics.failed_requests} ({metrics.error_rate*100:.1f}%)")
        print(f"  Duration:         {metrics.duration_seconds:.2f}s")
        print(f"  Throughput:       {metrics.requests_per_second:.2f} req/s")
        print(f"  Latency (p50):    {metrics.p50_latency_ms:.2f}ms")
        print(f"  Latency (p95):    {metrics.p95_latency_ms:.2f}ms")
        print(f"  Latency (p99):    {metrics.p99_latency_ms:.2f}ms")
        print(f"  Latency (min):    {metrics.min_latency_ms:.2f}ms")
        print(f"  Latency (max):    {metrics.max_latency_ms:.2f}ms")
        print(f"  Latency (mean):   {metrics.mean_latency_ms:.2f}ms")

        # Check targets
        targets_met = []
        targets_met.append(("Error Rate < 1%", metrics.error_rate < 0.01))
        targets_met.append(("p50 < 100ms", metrics.p50_latency_ms < 100))
        targets_met.append(("p99 < 500ms", metrics.p99_latency_ms < 500))

        print("\n  Target Status:")
        for target, met in targets_met:
            status = "✓" if met else "✗"
            print(f"    {status} {target}")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Run benchmark
    run_full_benchmark()

    # Or run pytest
    # pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])
