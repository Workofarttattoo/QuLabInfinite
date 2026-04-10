"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

GRAPH THEORY LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
from scipy.constants import pi

@dataclass
class Graph:
    nodes: int = 0
    edges: int = 0
    adjacency_matrix: np.ndarray = field(init=False)
    
    def __post_init__(self):
        self.adjacency_matrix = np.zeros((self.nodes, self.nodes), dtype=np.float64)
        
    def add_edge(self, node_a: int, node_b: int) -> None:
        if 0 <= node_a < self.nodes and 0 <= node_b < self.nodes:
            self.adjacency_matrix[node_a][node_b] = 1.0
            self.adjacency_matrix[node_b][node_a] = 1.0
        else:
            raise ValueError("Node indices must be within the range of [0, nodes).")
    
    def remove_edge(self, node_a: int, node_b: int) -> None:
        if 0 <= node_a < self.nodes and 0 <= node_b < self.nodes:
            self.adjacency_matrix[node_a][node_b] = 0.0
            self.adjacency_matrix[node_b][node_a] = 0.0
        else:
            raise ValueError("Node indices must be within the range of [0, nodes).")
    
    def has_edge(self, node_a: int, node_b: int) -> bool:
        if 0 <= node_a < self.nodes and 0 <= node_b < self.nodes:
            return self.adjacency_matrix[node_a][node_b] == 1.0
        else:
            raise ValueError("Node indices must be within the range of [0, nodes).")
    
    def get_degree(self, node: int) -> int:
        if 0 <= node < self.nodes:
            return int(np.sum(self.adjacency_matrix[node]))
        else:
            raise ValueError("Node index is out of bounds.")
    
    def clone(self):
        new_graph = Graph(nodes=self.nodes)
        new_graph.adjacency_matrix = np.copy(self.adjacency_matrix)
        return new_graph
    
    def get_neighbors(self, node: int) -> List[int]:
        if 0 <= node < self.nodes:
            neighbors_indices = np.nonzero(self.adjacency_matrix[node])[0]
            return [index for index in neighbors_indices]
        else:
            raise ValueError("Node index is out of bounds.")
    
    def bfs_traversal(self, start_node: int) -> List[int]:
        if 0 <= start_node < self.nodes:
            visited = np.zeros((self.nodes,), dtype=bool)
            queue = [start_node]
            
            traversal_order = []
            while queue:
                current_node = queue.pop(0)
                if not visited[current_node]:
                    visited[current_node] = True
                    traversal_order.append(current_node)
                    
                    for neighbor in self.get_neighbors(current_node):
                        if not visited[neighbor]:
                            queue.append(neighbor)
            
            return traversal_order
        else:
            raise ValueError("Node index is out of bounds.")
    
    def dfs_traversal(self, start_node: int) -> List[int]:
        if 0 <= start_node < self.nodes:
            visited = np.zeros((self.nodes,), dtype=bool)
            stack = [start_node]
            
            traversal_order = []
            while stack:
                current_node = stack.pop()
                if not visited[current_node]:
                    visited[current_node] = True
                    traversal_order.append(current_node)
                    
                    for neighbor in reversed(self.get_neighbors(current_node)):
                        if not visited[neighbor]:
                            stack.append(neighbor)
            
            return traversal_order
        else:
            raise ValueError("Node index is out of bounds.")
    
    def find_shortest_path(self, start_node: int, end_node: int) -> Tuple[List[int], float]:
        if 0 <= start_node < self.nodes and 0 <= end_node < self.nodes:
            visited = np.zeros((self.nodes,), dtype=bool)
            distances = np.full((self.nodes,), fill_value=np.inf, dtype=float)
            predecessors = [-1] * self.nodes
            queue = [start_node]
            
            distances[start_node] = 0.0
            
            while queue:
                current_node = queue.pop(0)
                
                if not visited[current_node]:
                    visited[current_node] = True
                    
                    for neighbor in self.get_neighbors(current_node):
                        distance_to_neighbor = distances[current_node] + 1.0
                        if distance_to_neighbor < distances[neighbor]:
                            distances[neighbor] = distance_to_neighbor
                            predecessors[neighbor] = current_node
                            queue.append(neighbor)
            
            path = []
            current_node = end_node
            while current_node != -1:
                path.insert(0, current_node)
                current_node = predecessors[current_node]
            
            if len(path) == 1 and start_node != end_node:
                return [], np.inf
            
            shortest_path_length = distances[end_node] if distances[end_node] < np.inf else None
            return path, shortest_path_length
        else:
            raise ValueError("Node index is out of bounds.")
    
    def find_all_shortest_paths(self) -> dict:
        all_shortest_paths = {}
        
        for start_node in range(self.nodes):
            node_distances = {start_node: 0.0}
            
            visited = np.zeros((self.nodes,), dtype=bool)
            queue = [start_node]
            
            while queue:
                current_node = queue.pop(0)
                
                if not visited[current_node]:
                    visited[current_node] = True
                    
                    for neighbor in self.get_neighbors(current_node):
                        distance_to_neighbor = node_distances[current_node] + 1.0
                        node_distances[neighbor] = distance_to_neighbor
                        queue.append(neighbor)
            
            all_shortest_paths[start_node] = node_distances
        
        return all_shortest_paths
    
    def find_connected_components(self) -> List[List[int]]:
        visited = np.zeros((self.nodes,), dtype=bool)
        components = []
        
        for start_node in range(self.nodes):
            if not visited[start_node]:
                component = self.bfs_traversal(start_node)
                components.append(component)
                
                for node in component:
                    visited[node] = True
        
        return components
    
    def is_bipartite(self) -> bool:
        color_map = {}
        
        queue = []
        start_nodes = list(range(self.nodes))
        np.random.shuffle(start_nodes)

        for start_node in start_nodes:
            if start_node not in color_map and self.adjacency_matrix[start_node].any():
                queue.append((start_node, 0)) # (node, color)
                
                while queue:
                    current_node, current_color = queue.pop(0)
                    
                    if current_node not in color_map:
                        color_map[current_node] = current_color
                    
                    for neighbor in self.get_neighbors(current_node):
                        if neighbor not in color_map or color_map[neighbor] == color_map[current_node]:
                            color_map[neighbor] = 1 - color_map[current_node]
                            queue.append((neighbor, 1 - color_map[current_node]))
                        elif color_map[neighbor] == current_color:
                            return False
        
        return True
    
    def find_max_flow(self, source: int, sink: int) -> float:
        if self.has_edge(source, sink):
            raise ValueError("Source and sink should not have a direct edge between them.")
        
        residual_graph = np.copy(self.adjacency_matrix)
        max_flow_value = 0.0
        
        while True:
            path_exists, bottleneck_capacity = self.find_augmenting_path(residual_graph, source, sink)
            
            if not path_exists:
                break
            
            max_flow_value += bottleneck_capacity
            
            augmented_residual_graph = np.copy(residual_graph)
            
            for node_index in range(len(path_exists) - 1):
                current_node = path_exists[node_index]
                next_node = path_exists[node_index + 1]
                
                if augmented_residual_graph[current_node][next_node] > 0.5:
                    augmented_residual_graph[current_node][next_node] -= bottleneck_capacity
                    augmented_residual_graph[next_node][current_node] += bottleneck_capacity
            
            residual_graph = augmented_residual_graph
        
        return max_flow_value
    
    def find_augmenting_path(self, residual_graph: np.ndarray, source: int, sink: int) -> Tuple[List[int], float]:
        visited_nodes = set()
        
        path_exists, bottleneck_capacity, stack = self.bfs_search(source, sink, visited_nodes, residual_graph)
        return path_exists, bottleneck_capacity
    
    def bfs_search(self, current_node: int, target_node: int, visited: set, residual_graph: np.ndarray) -> Tuple[List[int], float]:
        if current_node == target_node:
            return ([current_node], 1.0)
        
        neighbors = self.get_neighbors(current_node)
        path_exists, bottleneck_capacity = None, None
        
        for neighbor in neighbors:
            if neighbor not in visited and residual_graph[current_node][neighbor] > 0.5:
                visited.add(neighbor)
                
                next_path_exists, next_bottleneck_capacity = self.bfs_search(neighbor, target_node, visited, residual_graph)
                
                if next_bottleneck_capacity is None:
                    continue
                
                current_residual_capacity = residual_graph[current_node][neighbor]
                
                new_bottleneck_capacity = min(next_bottleneck_capacity, current_residual_capacity)
                
                if path_exists is not None and bottleneck_capacity < new_bottleneck_capacity:
                    path_exists = [current_node] + next_path_exists
                    bottleneck_capacity = new_bottleneck_capacity
        
        return (path_exists, bottleneck_capacity) if path_exists is not None else (None, None)

def run_demo():
    graph = Graph(nodes=6)
    
    edges = [(0, 1), (0, 3), (2, 3), (2, 4), (3, 5)]
    
    for edge in edges:
        graph.add_edge(edge[0], edge[1])
    
    print("Adjacency Matrix:")
    print(graph.adjacency_matrix)
    
    print("\nDegree of each node:")
    for i in range(graph.nodes):
        print(f"Node {i}: Degree = {graph.get_degree(i)}")
    
    print("\nBFS Traversal from Node 0: ")
    bfs_traversal_result = graph.bfs_traversal(0)
    print(bfs_traversal_result)

    print("\nDFS Traversal from Node 0:")
    dfs_traversal_result = graph.dfs_traversal(0)
    print(dfs_traversal_result)
    
    print("\nShortest Path from Node 0 to Node 5: ")
    shortest_path, path_length = graph.find_shortest_path(0, 5)
    if not shortest_path:
        print("No path exists.")
    else:
        print(f"Path: {shortest_path}")
        print(f"Length: {path_length}")

    print("\nConnected Components:")
    connected_components = graph.find_connected_components()
    for component in connected_components:
        print(component)

    print("\nIs Bipartite?")
    is_bipartite_result = graph.is_bipartite()
    print(is_bipartite_result)
    
if __name__ == '__main__':
    run_demo()
