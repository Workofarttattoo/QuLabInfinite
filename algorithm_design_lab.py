"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

ALGORITHM DESIGN LAB - Production Ready Implementation
Advanced sorting, graph algorithms, dynamic programming, and complexity analysis
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
import time
import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable, Any
from collections import defaultdict, deque
import random

# Constants and configuration
@dataclass
class AlgorithmConfig:
    max_array_size: int = 10000
    max_graph_nodes: int = 1000
    random_seed: Optional[int] = None
    verbose: bool = False
    benchmark_iterations: int = 100

@dataclass
class ComplexityResult:
    """Store algorithm complexity analysis results"""
    algorithm_name: str
    input_size: int
    execution_time: float
    space_complexity: str
    time_complexity: str
    comparisons: int = 0
    swaps: int = 0

class AlgorithmDesignLab:
    """Comprehensive algorithm design laboratory with sorting, graph, and DP algorithms"""

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        self.config = config or AlgorithmConfig()
        if self.config.random_seed:
            np.random.seed(self.config.random_seed)
            random.seed(self.config.random_seed)
        self.performance_history: List[ComplexityResult] = []

    # ============= SORTING ALGORITHMS =============

    def quicksort(self, arr: np.ndarray, track_stats: bool = False) -> Tuple[np.ndarray, int, int]:
        """
        Implements quicksort algorithm with Lomuto partition scheme.
        Average: O(n log n), Worst: O(n²), Space: O(log n)
        """
        arr = arr.copy()
        comparisons = [0]
        swaps = [0]

        def partition(low: int, high: int) -> int:
            pivot = arr[high]
            i = low - 1
            for j in range(low, high):
                comparisons[0] += 1
                if arr[j] <= pivot:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]
                    swaps[0] += 1
            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            swaps[0] += 1
            return i + 1

        def quicksort_recursive(low: int, high: int):
            if low < high:
                pi = partition(low, high)
                quicksort_recursive(low, pi - 1)
                quicksort_recursive(pi + 1, high)

        quicksort_recursive(0, len(arr) - 1)
        return arr, comparisons[0], swaps[0]

    def mergesort(self, arr: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Implements merge sort algorithm.
        Time: O(n log n) always, Space: O(n)
        """
        arr = arr.copy()
        comparisons = [0]

        def merge(left: np.ndarray, right: np.ndarray) -> np.ndarray:
            result = []
            i = j = 0
            while i < len(left) and j < len(right):
                comparisons[0] += 1
                if left[i] <= right[j]:
                    result.append(left[i])
                    i += 1
                else:
                    result.append(right[j])
                    j += 1
            result.extend(left[i:])
            result.extend(right[j:])
            return np.array(result)

        def mergesort_recursive(arr: np.ndarray) -> np.ndarray:
            if len(arr) <= 1:
                return arr
            mid = len(arr) // 2
            left = mergesort_recursive(arr[:mid])
            right = mergesort_recursive(arr[mid:])
            return merge(left, right)

        sorted_arr = mergesort_recursive(arr)
        return sorted_arr, comparisons[0]

    def heapsort(self, arr: np.ndarray) -> np.ndarray:
        """
        Implements heap sort algorithm using max heap.
        Time: O(n log n), Space: O(1)
        """
        arr = arr.copy()
        n = len(arr)

        def heapify(i: int, heap_size: int):
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2

            if left < heap_size and arr[left] > arr[largest]:
                largest = left
            if right < heap_size and arr[right] > arr[largest]:
                largest = right

            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                heapify(largest, heap_size)

        # Build max heap
        for i in range(n // 2 - 1, -1, -1):
            heapify(i, n)

        # Extract elements from heap
        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            heapify(0, i)

        return arr

    def radix_sort(self, arr: np.ndarray) -> np.ndarray:
        """
        Implements radix sort for non-negative integers.
        Time: O(d * n) where d is number of digits, Space: O(n)
        """
        if len(arr) == 0:
            return arr

        arr = arr.astype(int).copy()
        max_val = np.max(arr)
        exp = 1

        while max_val // exp > 0:
            counting_sort_by_digit(arr, exp)
            exp *= 10

        return arr

    # ============= GRAPH ALGORITHMS =============

    def dijkstra(self, graph: Dict[int, List[Tuple[int, float]]],
                 start: int, end: Optional[int] = None) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
        """
        Dijkstra's shortest path algorithm using min-heap.
        Time: O((V + E) log V), Space: O(V)
        """
        distances = {node: float('inf') for node in graph}
        distances[start] = 0
        previous = {node: None for node in graph}
        pq = [(0, start)]
        visited = set()

        while pq:
            current_dist, current = heapq.heappop(pq)

            if current in visited:
                continue

            visited.add(current)

            if end and current == end:
                break

            for neighbor, weight in graph.get(current, []):
                distance = current_dist + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))

        return distances, previous

    def bellman_ford(self, edges: List[Tuple[int, int, float]],
                     num_vertices: int, start: int) -> Tuple[Dict[int, float], bool]:
        """
        Bellman-Ford algorithm for shortest paths with negative weights.
        Time: O(VE), Space: O(V)
        Returns distances and whether negative cycle exists
        """
        distances = {i: float('inf') for i in range(num_vertices)}
        distances[start] = 0

        # Relax edges V-1 times
        for _ in range(num_vertices - 1):
            for u, v, weight in edges:
                if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight

        # Check for negative cycles
        negative_cycle = False
        for u, v, weight in edges:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                negative_cycle = True
                break

        return distances, negative_cycle

    def floyd_warshall(self, adj_matrix: np.ndarray) -> np.ndarray:
        """
        Floyd-Warshall all-pairs shortest path algorithm.
        Time: O(V³), Space: O(V²)
        """
        n = len(adj_matrix)
        dist = adj_matrix.copy()

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

        return dist

    def kruskal_mst(self, edges: List[Tuple[int, int, float]],
                    num_vertices: int) -> Tuple[List[Tuple[int, int, float]], float]:
        """
        Kruskal's minimum spanning tree algorithm using Union-Find.
        Time: O(E log E), Space: O(V)
        """
        class UnionFind:
            def __init__(self, n):
                self.parent = list(range(n))
                self.rank = [0] * n

            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]

            def union(self, x, y):
                px, py = self.find(x), self.find(y)
                if px == py:
                    return False
                if self.rank[px] < self.rank[py]:
                    px, py = py, px
                self.parent[py] = px
                if self.rank[px] == self.rank[py]:
                    self.rank[px] += 1
                return True

        edges = sorted(edges, key=lambda x: x[2])
        uf = UnionFind(num_vertices)
        mst = []
        total_weight = 0

        for u, v, weight in edges:
            if uf.union(u, v):
                mst.append((u, v, weight))
                total_weight += weight
                if len(mst) == num_vertices - 1:
                    break

        return mst, total_weight

    def topological_sort(self, graph: Dict[int, List[int]]) -> Optional[List[int]]:
        """
        Topological sort using Kahn's algorithm (BFS-based).
        Time: O(V + E), Space: O(V)
        Returns None if cycle exists
        """
        in_degree = defaultdict(int)
        for node in graph:
            for neighbor in graph[node]:
                in_degree[neighbor] += 1

        queue = deque([node for node in graph if in_degree[node] == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)

            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result if len(result) == len(graph) else None

    def strongly_connected_components(self, graph: Dict[int, List[int]]) -> List[List[int]]:
        """
        Kosaraju's algorithm for finding strongly connected components.
        Time: O(V + E), Space: O(V)
        """
        def dfs1(node, visited, stack):
            visited.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    dfs1(neighbor, visited, stack)
            stack.append(node)

        def dfs2(node, visited, component):
            visited.add(node)
            component.append(node)
            for neighbor in reversed_graph.get(node, []):
                if neighbor not in visited:
                    dfs2(neighbor, visited, component)

        # First DFS to fill stack
        visited = set()
        stack = []
        for node in graph:
            if node not in visited:
                dfs1(node, visited, stack)

        # Create reversed graph
        reversed_graph = defaultdict(list)
        for node in graph:
            for neighbor in graph[node]:
                reversed_graph[neighbor].append(node)

        # Second DFS on reversed graph
        visited = set()
        sccs = []
        while stack:
            node = stack.pop()
            if node not in visited:
                component = []
                dfs2(node, visited, component)
                sccs.append(component)

        return sccs

    # ============= DYNAMIC PROGRAMMING =============

    def longest_common_subsequence(self, seq1: str, seq2: str) -> Tuple[int, str]:
        """
        Finds longest common subsequence using dynamic programming.
        Time: O(mn), Space: O(mn)
        """
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        # Reconstruct LCS
        lcs = []
        i, j = m, n
        while i > 0 and j > 0:
            if seq1[i-1] == seq2[j-1]:
                lcs.append(seq1[i-1])
                i -= 1
                j -= 1
            elif dp[i-1][j] > dp[i][j-1]:
                i -= 1
            else:
                j -= 1

        return dp[m][n], ''.join(reversed(lcs))

    def knapsack_01(self, weights: List[int], values: List[int],
                    capacity: int) -> Tuple[int, List[int]]:
        """
        Solves 0/1 knapsack problem using dynamic programming.
        Time: O(nW), Space: O(nW) where W is capacity
        """
        n = len(weights)
        dp = [[0] * (capacity + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            for w in range(capacity + 1):
                if weights[i-1] <= w:
                    dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1])
                else:
                    dp[i][w] = dp[i-1][w]

        # Reconstruct solution
        selected = []
        w = capacity
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                selected.append(i-1)
                w -= weights[i-1]

        return dp[n][capacity], list(reversed(selected))

    def edit_distance(self, str1: str, str2: str) -> Tuple[int, List[str]]:
        """
        Computes Levenshtein distance and transformation sequence.
        Time: O(mn), Space: O(mn)
        """
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        # Reconstruct operations
        operations = []
        i, j = m, n
        while i > 0 or j > 0:
            if i == 0:
                operations.append(f"Insert '{str2[j-1]}'")
                j -= 1
            elif j == 0:
                operations.append(f"Delete '{str1[i-1]}'")
                i -= 1
            elif str1[i-1] == str2[j-1]:
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i-1][j-1] + 1:
                operations.append(f"Replace '{str1[i-1]}' with '{str2[j-1]}'")
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i-1][j] + 1:
                operations.append(f"Delete '{str1[i-1]}'")
                i -= 1
            else:
                operations.append(f"Insert '{str2[j-1]}'")
                j -= 1

        return dp[m][n], list(reversed(operations))

    def matrix_chain_multiplication(self, dimensions: List[int]) -> Tuple[int, str]:
        """
        Finds optimal parenthesization for matrix chain multiplication.
        Time: O(n³), Space: O(n²)
        """
        n = len(dimensions) - 1
        dp = [[0] * n for _ in range(n)]
        split = [[-1] * n for _ in range(n)]

        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                dp[i][j] = float('inf')
                for k in range(i, j):
                    cost = (dp[i][k] + dp[k+1][j] +
                           dimensions[i] * dimensions[k+1] * dimensions[j+1])
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        split[i][j] = k

        def build_parenthesization(i, j):
            if i == j:
                return f"M{i}"
            k = split[i][j]
            return f"({build_parenthesization(i, k)} × {build_parenthesization(k+1, j)})"

        return dp[0][n-1], build_parenthesization(0, n-1)

    # ============= COMPLEXITY ANALYSIS =============

    def analyze_sorting_complexity(self, size: int) -> Dict[str, ComplexityResult]:
        """Analyzes complexity of different sorting algorithms"""
        arr = np.random.random(size)
        results = {}

        algorithms = [
            ("QuickSort", self.quicksort, "O(n log n)", "O(log n)"),
            ("MergeSort", self.mergesort, "O(n log n)", "O(n)"),
            ("HeapSort", self.heapsort, "O(n log n)", "O(1)")
        ]

        for name, func, time_comp, space_comp in algorithms:
            start_time = time.time()
            if name == "QuickSort":
                sorted_arr, comps, swaps = func(arr, track_stats=True)
                result = ComplexityResult(
                    algorithm_name=name,
                    input_size=size,
                    execution_time=time.time() - start_time,
                    time_complexity=time_comp,
                    space_complexity=space_comp,
                    comparisons=comps,
                    swaps=swaps
                )
            else:
                func(arr)
                result = ComplexityResult(
                    algorithm_name=name,
                    input_size=size,
                    execution_time=time.time() - start_time,
                    time_complexity=time_comp,
                    space_complexity=space_comp
                )

            results[name] = result
            self.performance_history.append(result)

        return results

    def benchmark_algorithms(self, input_sizes: List[int]) -> Dict[str, List[float]]:
        """Benchmarks algorithms across different input sizes"""
        results = defaultdict(list)

        for size in input_sizes:
            complexity_results = self.analyze_sorting_complexity(size)
            for algo_name, result in complexity_results.items():
                results[algo_name].append(result.execution_time)

        return dict(results)

def counting_sort_by_digit(arr: np.ndarray, exp: int):
    """Helper function for radix sort"""
    n = len(arr)
    output = np.zeros(n, dtype=int)
    count = np.zeros(10, dtype=int)

    for i in range(n):
        index = (arr[i] // exp) % 10
        count[index] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    for i in range(n - 1, -1, -1):
        index = (arr[i] // exp) % 10
        output[count[index] - 1] = arr[i]
        count[index] -= 1

    arr[:] = output

def run_demo():
    """Comprehensive demonstration of algorithm design lab"""
    print("="*60)
    print("ALGORITHM DESIGN LAB - Comprehensive Demo")
    print("="*60)

    lab = AlgorithmDesignLab(AlgorithmConfig(verbose=True))

    # Sorting demonstrations
    print("\n1. SORTING ALGORITHMS")
    print("-" * 40)
    test_array = np.random.randint(0, 100, 20)
    print(f"Original array: {test_array}")

    sorted_arr, comps, swaps = lab.quicksort(test_array)
    print(f"QuickSort: {sorted_arr}")
    print(f"  Comparisons: {comps}, Swaps: {swaps}")

    sorted_arr, comps = lab.mergesort(test_array)
    print(f"MergeSort: {sorted_arr}")
    print(f"  Comparisons: {comps}")

    # Graph algorithms
    print("\n2. GRAPH ALGORITHMS")
    print("-" * 40)

    # Create sample graph
    graph = {
        0: [(1, 4), (2, 2)],
        1: [(2, 1), (3, 5)],
        2: [(3, 8), (4, 10)],
        3: [(4, 2), (5, 6)],
        4: [(5, 3)],
        5: []
    }

    distances, prev = lab.dijkstra(graph, 0)
    print(f"Dijkstra shortest paths from 0: {distances}")

    # Dynamic programming
    print("\n3. DYNAMIC PROGRAMMING")
    print("-" * 40)

    # LCS example
    seq1, seq2 = "AGGTAB", "GXTXAYB"
    length, lcs = lab.longest_common_subsequence(seq1, seq2)
    print(f"LCS of '{seq1}' and '{seq2}': '{lcs}' (length: {length})")

    # Knapsack example
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = 50
    max_value, items = lab.knapsack_01(weights, values, capacity)
    print(f"Knapsack: Max value = {max_value}, Items = {items}")

    # Edit distance
    str1, str2 = "saturday", "sunday"
    distance, ops = lab.edit_distance(str1, str2)
    print(f"Edit distance from '{str1}' to '{str2}': {distance}")
    print(f"Operations: {ops[:3]}...")  # Show first 3 operations

    # Complexity analysis
    print("\n4. COMPLEXITY ANALYSIS")
    print("-" * 40)
    sizes = [100, 500, 1000]
    benchmarks = lab.benchmark_algorithms(sizes)

    for algo, times in benchmarks.items():
        print(f"{algo}: {[f'{t:.4f}s' for t in times]}")

if __name__ == '__main__':
    run_demo()