"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Semantic Lattice - Knowledge Graph for Experimental Data
Multi-dimensional property spaces with analogical reasoning and inference
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


class RelationType(Enum):
    """Types of relationships between concepts"""
    CAUSALITY = "causality"  # A causes B
    SIMILARITY = "similarity"  # A is similar to B
    COMPOSITION = "composition"  # A is composed of B
    INTERACTION = "interaction"  # A interacts with B
    CORRELATION = "correlation"  # A correlates with B
    SUBSTITUTION = "substitution"  # A can substitute for B
    HIERARCHY = "hierarchy"  # A is a subclass of B
    TEMPORAL = "temporal"  # A precedes B


@dataclass
class ConceptNode:
    """Represents a concept in the knowledge graph"""
    node_id: str
    concept_type: str  # "material", "reaction", "condition", "result", "property"
    name: str
    properties: Dict[str, Any]
    embedding: Optional[np.ndarray] = None  # Vector representation
    confidence: float = 1.0  # 0.0 to 1.0
    source: str = "experiment"  # "experiment", "prediction", "literature"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def similarity(self, other: 'ConceptNode') -> float:
        """Compute semantic similarity to another concept"""
        if self.embedding is None or other.embedding is None:
            # Fallback to property-based similarity
            return self._property_similarity(other)

        # Cosine similarity of embeddings
        norm_self = np.linalg.norm(self.embedding)
        norm_other = np.linalg.norm(other.embedding)
        if norm_self == 0 or norm_other == 0:
            return 0.0
        return float(np.dot(self.embedding, other.embedding) / (norm_self * norm_other))

    def _property_similarity(self, other: 'ConceptNode') -> float:
        """Compute similarity based on shared properties"""
        shared_keys = set(self.properties.keys()) & set(other.properties.keys())
        if not shared_keys:
            return 0.0

        matches = 0
        for key in shared_keys:
            if self.properties[key] == other.properties[key]:
                matches += 1
        return matches / len(shared_keys)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "node_id": self.node_id,
            "concept_type": self.concept_type,
            "name": self.name,
            "properties": self.properties,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "confidence": self.confidence,
            "source": self.source,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


@dataclass
class RelationshipEdge:
    """Represents a relationship between concepts"""
    edge_id: str
    source_node: str
    target_node: str
    relation_type: RelationType
    strength: float = 1.0  # 0.0 to 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    bidirectional: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "edge_id": self.edge_id,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "relation_type": self.relation_type.value,
            "strength": self.strength,
            "properties": self.properties,
            "confidence": self.confidence,
            "bidirectional": self.bidirectional,
            "metadata": self.metadata,
            "created_at": self.created_at
        }


class PropertyTensor:
    """Multi-dimensional property space with interpolation"""

    def __init__(self, property_names: List[str]):
        self.property_names = property_names
        self.dimension = len(property_names)
        self.data_points: List[Tuple[np.ndarray, Any]] = []  # (coordinates, value)

    def add_point(self, coordinates: Dict[str, float], value: Any) -> None:
        """Add data point to property space"""
        coord_array = np.array([coordinates.get(name, 0.0) for name in self.property_names])
        self.data_points.append((coord_array, value))

    def interpolate(self, coordinates: Dict[str, float], k: int = 5) -> Tuple[Any, float]:
        """Interpolate value at given coordinates using k-nearest neighbors"""
        if not self.data_points:
            return None, 0.0

        query_point = np.array([coordinates.get(name, 0.0) for name in self.property_names])

        # Compute distances to all points
        distances = []
        for coord, value in self.data_points:
            dist = np.linalg.norm(query_point - coord)
            distances.append((dist, value))

        # Sort by distance and take k nearest
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:min(k, len(distances))]

        # Weighted average by inverse distance
        if k_nearest[0][0] == 0:  # Exact match
            return k_nearest[0][1], 1.0

        weights = [1.0 / (d + 1e-10) for d, _ in k_nearest]
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # If values are numeric, compute weighted average
        values = [v for _, v in k_nearest]
        if all(isinstance(v, (int, float)) for v in values):
            interpolated = sum(w * v for w, (_, v) in zip(normalized_weights, k_nearest))
            confidence = 1.0 / (1.0 + k_nearest[0][0])  # Confidence based on nearest distance
            return interpolated, confidence
        else:
            # Non-numeric: return most common value
            from collections import Counter
            value_counts = Counter(values)
            most_common = value_counts.most_common(1)[0][0]
            confidence = value_counts[most_common] / len(values)
            return most_common, confidence

    def get_gradient(self, coordinates: Dict[str, float], property_name: str, epsilon: float = 0.01) -> Dict[str, float]:
        """Compute gradient of property with respect to coordinates"""
        gradients = {}
        base_value, _ = self.interpolate(coordinates)

        if not isinstance(base_value, (int, float)):
            return {}

        for name in self.property_names:
            perturbed = coordinates.copy()
            perturbed[name] = perturbed.get(name, 0.0) + epsilon
            perturbed_value, _ = self.interpolate(perturbed)

            if isinstance(perturbed_value, (int, float)):
                gradients[name] = (perturbed_value - base_value) / epsilon

        return gradients


class InferenceEngine:
    """Predict properties of untested combinations using graph structure"""

    def __init__(self, knowledge_graph: 'KnowledgeGraph'):
        self.kg = knowledge_graph

    def predict_property(self, concept_id: str, property_name: str) -> Tuple[Any, float]:
        """Predict property value for concept using graph relationships"""
        concept = self.kg.get_node(concept_id)
        if not concept:
            return None, 0.0

        # If property exists, return it
        if property_name in concept.properties:
            return concept.properties[property_name], concept.confidence

        # Find similar concepts with this property
        similar = self.kg.find_similar_nodes(concept_id, top_k=10)
        predictions = []

        for sim_id, similarity in similar:
            sim_concept = self.kg.get_node(sim_id)
            if sim_concept and property_name in sim_concept.properties:
                value = sim_concept.properties[property_name]
                confidence = similarity * sim_concept.confidence
                predictions.append((value, confidence))

        if not predictions:
            return None, 0.0

        # Weighted average by confidence
        if all(isinstance(v, (int, float)) for v, _ in predictions):
            total_conf = sum(c for _, c in predictions)
            weighted_avg = sum(v * c for v, c in predictions) / total_conf
            return weighted_avg, total_conf / len(predictions)
        else:
            # Return most confident prediction
            predictions.sort(key=lambda x: x[1], reverse=True)
            return predictions[0]

    def analogical_reasoning(self, source_pair: Tuple[str, str],
                           target_start: str) -> Optional[str]:
        """
        Analogical reasoning: if A:B :: C:?, predict ?
        Example: if steel:strong :: aluminum:?, predict "lightweight_strong"
        """
        node_a = self.kg.get_node(source_pair[0])
        node_b = self.kg.get_node(source_pair[1])
        node_c = self.kg.get_node(target_start)

        if not (node_a and node_b and node_c):
            return None

        # Find relationship between A and B
        edges_ab = self.kg.get_edges_between(source_pair[0], source_pair[1])
        if not edges_ab:
            return None

        relation_type = edges_ab[0].relation_type

        # Find candidates related to C with same relation type
        candidates = []
        for edge in self.kg.edges.values():
            if edge.source_node == target_start and edge.relation_type == relation_type:
                candidate = self.kg.get_node(edge.target_node)
                if candidate:
                    # Score by similarity to B
                    similarity = node_b.similarity(candidate)
                    candidates.append((edge.target_node, similarity))

        if not candidates:
            return None

        # Return best candidate
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def predict_untested_combination(self, components: List[str],
                                    property_name: str) -> Tuple[Any, float]:
        """Predict property of untested combination of components"""
        # Get properties of individual components
        component_values = []
        for comp_id in components:
            value, conf = self.predict_property(comp_id, property_name)
            if value is not None:
                component_values.append((value, conf))

        if not component_values:
            return None, 0.0

        # Check if property is additive (use simple average)
        if all(isinstance(v, (int, float)) for v, _ in component_values):
            # Weighted average by confidence
            total_conf = sum(c for _, c in component_values)
            avg_value = sum(v * c for v, c in component_values) / total_conf
            avg_conf = total_conf / len(component_values)

            # Reduce confidence for untested combinations
            return avg_value, avg_conf * 0.7
        else:
            # Non-numeric: return most confident value
            component_values.sort(key=lambda x: x[1], reverse=True)
            return component_values[0][0], component_values[0][1] * 0.7


class KnowledgeGraph:
    """Persistent knowledge graph with fast queries"""

    def __init__(self):
        self.nodes: Dict[str, ConceptNode] = {}
        self.edges: Dict[str, RelationshipEdge] = {}
        self.node_index: Dict[str, Set[str]] = defaultdict(set)  # concept_type -> node_ids
        self.edge_index: Dict[str, Set[str]] = defaultdict(set)  # relation_type -> edge_ids
        self.adjacency_out: Dict[str, List[str]] = defaultdict(list)  # node_id -> edge_ids (outgoing)
        self.adjacency_in: Dict[str, List[str]] = defaultdict(list)  # node_id -> edge_ids (incoming)

    def add_node(self, node: ConceptNode) -> None:
        """Add concept node to graph"""
        self.nodes[node.node_id] = node
        self.node_index[node.concept_type].add(node.node_id)
        LOG.info(f"[info] Added node {node.node_id} ({node.concept_type}: {node.name})")

    def add_edge(self, edge: RelationshipEdge) -> None:
        """Add relationship edge to graph"""
        self.edges[edge.edge_id] = edge
        self.edge_index[edge.relation_type.value].add(edge.edge_id)
        self.adjacency_out[edge.source_node].append(edge.edge_id)
        self.adjacency_in[edge.target_node].append(edge.edge_id)
        if edge.bidirectional:
            self.adjacency_out[edge.target_node].append(edge.edge_id)
            self.adjacency_in[edge.source_node].append(edge.edge_id)
        LOG.info(f"[info] Added edge {edge.edge_id} ({edge.source_node} --{edge.relation_type.value}--> {edge.target_node})")

    def get_node(self, node_id: str) -> Optional[ConceptNode]:
        """Get node by ID"""
        return self.nodes.get(node_id)

    def get_edge(self, edge_id: str) -> Optional[RelationshipEdge]:
        """Get edge by ID"""
        return self.edges.get(edge_id)

    def get_nodes_by_type(self, concept_type: str) -> List[ConceptNode]:
        """Get all nodes of specific type"""
        node_ids = self.node_index.get(concept_type, set())
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]

    def get_edges_by_type(self, relation_type: RelationType) -> List[RelationshipEdge]:
        """Get all edges of specific type"""
        edge_ids = self.edge_index.get(relation_type.value, set())
        return [self.edges[eid] for eid in edge_ids if eid in self.edges]

    def get_outgoing_edges(self, node_id: str) -> List[RelationshipEdge]:
        """Get all edges originating from node"""
        edge_ids = self.adjacency_out.get(node_id, [])
        return [self.edges[eid] for eid in edge_ids if eid in self.edges]

    def get_incoming_edges(self, node_id: str) -> List[RelationshipEdge]:
        """Get all edges pointing to node"""
        edge_ids = self.adjacency_in.get(node_id, [])
        return [self.edges[eid] for eid in edge_ids if eid in self.edges]

    def get_edges_between(self, source_id: str, target_id: str) -> List[RelationshipEdge]:
        """Get all edges between two nodes"""
        outgoing = self.get_outgoing_edges(source_id)
        return [edge for edge in outgoing if edge.target_node == target_id]

    def find_similar_nodes(self, node_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find most similar nodes (fast query <10ms for 10k nodes)"""
        query_node = self.get_node(node_id)
        if not query_node:
            return []

        similarities = []
        # Only compare nodes of same type for speed
        candidates = self.get_nodes_by_type(query_node.concept_type)

        for candidate in candidates:
            if candidate.node_id == node_id:
                continue
            sim = query_node.similarity(candidate)
            similarities.append((candidate.node_id, sim))

        # Sort and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def query_by_properties(self, concept_type: str, filters: Dict[str, Any]) -> List[ConceptNode]:
        """Query nodes by property filters"""
        candidates = self.get_nodes_by_type(concept_type)
        results = []

        for node in candidates:
            match = True
            for key, value in filters.items():
                if key not in node.properties or node.properties[key] != value:
                    match = False
                    break
            if match:
                results.append(node)

        return results

    def detect_contradictions(self) -> List[Dict[str, Any]]:
        """Detect contradictory information in knowledge graph"""
        contradictions = []

        # Check for conflicting property values between similar concepts
        for node_id, node in self.nodes.items():
            similar = self.find_similar_nodes(node_id, top_k=5)
            for sim_id, similarity in similar:
                if similarity > 0.9:  # Very similar concepts
                    sim_node = self.get_node(sim_id)
                    if sim_node:
                        # Check for conflicting properties
                        shared_props = set(node.properties.keys()) & set(sim_node.properties.keys())
                        for prop in shared_props:
                            if isinstance(node.properties[prop], (int, float)) and \
                               isinstance(sim_node.properties[prop], (int, float)):
                                # Check if values differ significantly
                                val1, val2 = node.properties[prop], sim_node.properties[prop]
                                if val1 != 0 and abs(val1 - val2) / abs(val1) > 0.5:  # >50% difference
                                    contradictions.append({
                                        "type": "property_mismatch",
                                        "node1": node_id,
                                        "node2": sim_id,
                                        "property": prop,
                                        "value1": val1,
                                        "value2": val2,
                                        "similarity": similarity
                                    })

        return contradictions

    def identify_knowledge_gaps(self) -> List[Dict[str, Any]]:
        """Identify missing information and unexplored areas"""
        gaps = []

        # Find nodes with incomplete property sets
        property_sets = defaultdict(set)
        for node in self.nodes.values():
            property_sets[node.concept_type].update(node.properties.keys())

        for node in self.nodes.values():
            expected_props = property_sets[node.concept_type]
            missing_props = expected_props - set(node.properties.keys())
            if len(missing_props) > len(expected_props) * 0.3:  # Missing >30% of properties
                gaps.append({
                    "type": "incomplete_properties",
                    "node_id": node.node_id,
                    "missing_properties": list(missing_props)
                })

        # Find isolated nodes (no edges)
        for node_id in self.nodes.keys():
            if not self.adjacency_out[node_id] and not self.adjacency_in[node_id]:
                gaps.append({
                    "type": "isolated_node",
                    "node_id": node_id
                })

        return gaps

    def save(self, filepath: str) -> None:
        """Save knowledge graph to JSON file"""
        data = {
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "edges": {eid: edge.to_dict() for eid, edge in self.edges.items()},
            "metadata": {
                "num_nodes": len(self.nodes),
                "num_edges": len(self.edges),
                "saved_at": time.time()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        LOG.info(f"[info] Knowledge graph saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load knowledge graph from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Load nodes
        for node_data in data["nodes"].values():
            if node_data["embedding"] is not None:
                node_data["embedding"] = np.array(node_data["embedding"])
            node = ConceptNode(**node_data)
            self.add_node(node)

        # Load edges
        for edge_data in data["edges"].values():
            edge_data["relation_type"] = RelationType(edge_data["relation_type"])
            edge = RelationshipEdge(**edge_data)
            self.add_edge(edge)

        LOG.info(f"[info] Knowledge graph loaded from {filepath}: {len(self.nodes)} nodes, {len(self.edges)} edges")

    def visualize_ascii(self, center_node: str, depth: int = 2) -> str:
        """Generate ASCII visualization of graph neighborhood"""
        visited = set()
        lines = []

        def traverse(node_id: str, current_depth: int, prefix: str):
            if current_depth > depth or node_id in visited:
                return
            visited.add(node_id)

            node = self.get_node(node_id)
            if not node:
                return

            lines.append(f"{prefix}{node.name} ({node.concept_type})")

            if current_depth < depth:
                edges = self.get_outgoing_edges(node_id)
                for i, edge in enumerate(edges):
                    is_last = i == len(edges) - 1
                    connector = "└── " if is_last else "├── "
                    next_prefix = prefix + ("    " if is_last else "│   ")
                    lines.append(f"{prefix}{connector}[{edge.relation_type.value}]")
                    traverse(edge.target_node, current_depth + 1, next_prefix)

        traverse(center_node, 0, "")
        return "\n".join(lines)


if __name__ == "__main__":
    # Demo
    kg = KnowledgeGraph()

    # Add materials
    steel = ConceptNode(
        node_id="mat_steel_304",
        concept_type="material",
        name="AISI 304 Stainless Steel",
        properties={
            "tensile_strength": 515,  # MPa
            "yield_strength": 205,
            "density": 8.0,  # g/cm³
            "corrosion_resistance": "excellent"
        },
        embedding=np.random.randn(128)
    )

    aluminum = ConceptNode(
        node_id="mat_aluminum_6061",
        concept_type="material",
        name="Aluminum 6061",
        properties={
            "tensile_strength": 310,
            "yield_strength": 276,
            "density": 2.7,
            "corrosion_resistance": "good"
        },
        embedding=np.random.randn(128)
    )

    kg.add_node(steel)
    kg.add_node(aluminum)

    # Add relationship
    edge = RelationshipEdge(
        edge_id="edge_001",
        source_node="mat_steel_304",
        target_node="mat_aluminum_6061",
        relation_type=RelationType.SUBSTITUTION,
        strength=0.7,
        properties={"use_case": "lightweight_applications"}
    )
    kg.add_edge(edge)

    # Test similarity
    similar = kg.find_similar_nodes("mat_steel_304", top_k=5)
    print(f"Similar to steel: {similar}")

    # Test inference
    inference = InferenceEngine(kg)
    predicted, conf = inference.predict_property("mat_aluminum_6061", "tensile_strength")
    print(f"Predicted tensile strength: {predicted} MPa (confidence: {conf:.2f})")

    # Visualize
    print("\nGraph visualization:")
    print(kg.visualize_ascii("mat_steel_304", depth=2))

    # Save
    kg.save("/tmp/knowledge_graph_demo.json")
