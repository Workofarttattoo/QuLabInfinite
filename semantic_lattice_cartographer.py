"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Semantic Lattice System Cartographer for QuLab Stack
Maps the entire scientific capability space across 100+ laboratories
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import importlib
import inspect
import ast
from collections import defaultdict
import hashlib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class LabCapability:
    """Represents a single capability within a lab"""
    name: str
    function_signature: str
    docstring: str
    parameters: Dict[str, Any]
    returns: str
    domain_keywords: List[str]
    scientific_references: List[str] = field(default_factory=list)
    complexity_class: str = 'O(n)'  # Computational complexity
    is_real_algorithm: bool = True  # False if it's fake/placeholder


@dataclass
class LabNode:
    """Represents a complete laboratory in the semantic lattice"""
    module_name: str
    file_path: str
    class_name: Optional[str]
    capabilities: List[LabCapability]
    domain: str  # Primary scientific domain
    subdomains: List[str]
    dependencies: List[str]  # Other labs it depends on
    embedding: Optional[np.ndarray] = None
    quality_score: float = 0.0  # 0-1, based on real algorithms vs placeholders


@dataclass
class SemanticEdge:
    """Represents a relationship between labs"""
    source: str
    target: str
    relationship_type: str  # 'depends_on', 'complements', 'extends', 'alternative'
    strength: float  # 0-1, semantic similarity or dependency strength


class SemanticLatticeCartographer:
    """
    Maps the entire QuLab scientific capability space into a searchable semantic lattice.
    NO GUIs, NO fake visualizations - only real computational mapping.
    """

    def __init__(self, lab_directory: str = None):
        self.lab_directory = Path(lab_directory or os.path.dirname(__file__))
        self.labs: Dict[str, LabNode] = {}
        self.edges: List[SemanticEdge] = []
        self.domain_taxonomy = self._build_domain_taxonomy()
        self.capability_index: Dict[str, List[str]] = defaultdict(list)  # keyword -> lab_names

    def _build_domain_taxonomy(self) -> Dict[str, List[str]]:
        """Build scientific domain taxonomy"""
        return {
            'biology': ['oncology', 'genomics', 'proteomics', 'cell_biology', 'immunology',
                       'neuroscience', 'metabolism', 'virology', 'bacteriology', 'ecology'],
            'chemistry': ['organic', 'inorganic', 'physical', 'analytical', 'quantum_chemistry',
                         'materials', 'catalysis', 'electrochemistry', 'photochemistry'],
            'physics': ['quantum', 'particle', 'condensed_matter', 'optics', 'nuclear',
                       'astrophysics', 'plasma', 'acoustics', 'thermodynamics'],
            'medicine': ['oncology', 'cardiology', 'neurology', 'pharmacology', 'surgery',
                        'radiology', 'pathology', 'immunotherapy', 'gene_therapy'],
            'engineering': ['biomedical', 'materials', 'chemical', 'electrical', 'mechanical',
                           'aerospace', 'nuclear', 'software', 'systems'],
            'mathematics': ['optimization', 'statistics', 'topology', 'algebra', 'analysis',
                           'geometry', 'probability', 'numerical', 'graph_theory'],
            'computation': ['machine_learning', 'quantum_computing', 'simulation', 'algorithms',
                           'distributed', 'cryptography', 'blockchain', 'parallel'],
            'nanotechnology': ['synthesis', 'characterization', 'self_assembly', 'quantum_dots',
                              'nanoparticles', 'graphene', 'carbon_nanotubes', 'nanowires']
        }

    def discover_labs(self) -> int:
        """Discover and analyze all lab files"""
        lab_files = list(self.lab_directory.glob('*.py'))
        discovered_count = 0

        for lab_file in lab_files:
            if lab_file.name.startswith('_') or lab_file.name in ['__init__.py', 'semantic_lattice_cartographer.py', 'qulab_mcp_server.py']:
                continue

            try:
                lab_node = self._analyze_lab_file(lab_file)
                if lab_node and lab_node.capabilities:
                    self.labs[lab_node.module_name] = lab_node
                    discovered_count += 1

                    # Index capabilities by keywords
                    for capability in lab_node.capabilities:
                        for keyword in capability.domain_keywords:
                            self.capability_index[keyword.lower()].append(lab_node.module_name)

            except Exception as e:
                print(f"[warn] Failed to analyze {lab_file.name}: {e}")

        # Build semantic edges after all labs are discovered
        self._build_semantic_edges()

        return discovered_count

    def _analyze_lab_file(self, file_path: Path) -> Optional[LabNode]:
        """Analyze a single lab file to extract capabilities"""
        with open(file_path, 'r') as f:
            source_code = f.read()

        try:
            tree = ast.parse(source_code)
        except:
            return None

        module_name = file_path.stem
        capabilities = []
        class_name = None
        domain = self._infer_domain(module_name, source_code)
        subdomains = self._infer_subdomains(module_name, source_code)
        dependencies = self._extract_dependencies(tree)

        # Find main class and its methods
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Skip test classes
                if 'test' in node.name.lower():
                    continue

                class_name = node.name

                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        # Skip private methods and constructors
                        if item.name.startswith('_'):
                            continue

                        capability = self._extract_capability(item, source_code)
                        if capability:
                            capabilities.append(capability)

        # Also check for module-level functions
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                if not node.name.startswith('_'):
                    capability = self._extract_capability(node, source_code)
                    if capability:
                        capabilities.append(capability)

        if not capabilities:
            return None

        # Calculate quality score based on algorithm complexity
        quality_score = self._calculate_quality_score(capabilities, source_code)

        return LabNode(
            module_name=module_name,
            file_path=str(file_path),
            class_name=class_name,
            capabilities=capabilities,
            domain=domain,
            subdomains=subdomains,
            dependencies=dependencies,
            quality_score=quality_score
        )

    def _extract_capability(self, func_node: ast.FunctionDef, source_code: str) -> Optional[LabCapability]:
        """Extract capability information from a function"""
        # Get function signature
        args = []
        for arg in func_node.args.args:
            if arg.arg != 'self':
                args.append(arg.arg)
        signature = f"{func_node.name}({', '.join(args)})"

        # Get docstring
        docstring = ast.get_docstring(func_node) or ""

        # Extract parameters and types from annotations
        parameters = {}
        for arg in func_node.args.args:
            if arg.arg != 'self':
                param_type = 'Any'
                if arg.annotation:
                    param_type = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else 'Any'
                parameters[arg.arg] = param_type

        # Extract return type
        returns = 'Any'
        if func_node.returns:
            returns = ast.unparse(func_node.returns) if hasattr(ast, 'unparse') else 'Any'

        # Extract domain keywords from function name and docstring
        keywords = self._extract_keywords(func_node.name, docstring)

        # Check if it's a real algorithm or placeholder
        is_real = self._is_real_algorithm(func_node, source_code)

        # Extract scientific references if mentioned
        references = self._extract_references(docstring)

        return LabCapability(
            name=func_node.name,
            function_signature=signature,
            docstring=docstring[:200],  # Truncate long docstrings
            parameters=parameters,
            returns=returns,
            domain_keywords=keywords,
            scientific_references=references,
            is_real_algorithm=is_real
        )

    def _is_real_algorithm(self, func_node: ast.FunctionDef, source_code: str) -> bool:
        """Detect if function implements a real algorithm or is a placeholder"""
        # Check for suspicious patterns indicating fake/placeholder code
        fake_patterns = [
            'np.random',  # Random data generation without algorithm
            'plt.plot',    # Just plotting
            'time.sleep',  # Fake delays
            'print("Demo")',  # Demo messages
            'return random',  # Returning random values
            '# TODO',      # Unimplemented
            '# PLACEHOLDER', # Explicit placeholder
            'pass',        # Empty implementation
        ]

        func_source = ast.get_source_segment(source_code, func_node) if hasattr(ast, 'get_source_segment') else ""

        # If function is too short, likely placeholder
        if len(func_source) < 100:
            return False

        # Check for fake patterns
        for pattern in fake_patterns:
            if pattern in func_source:
                # But allow some legitimate uses of random (Monte Carlo, etc)
                if 'np.random' in func_source and any(x in func_source for x in ['monte_carlo', 'stochastic', 'sampling']):
                    continue
                return False

        # Check for real scientific computations
        real_patterns = [
            'np.exp', 'np.log', 'np.sin', 'np.cos',  # Mathematical functions
            'scipy.', 'sklearn.', 'torch.', 'tf.',     # Scientific libraries
            'differential', 'integral', 'gradient',    # Calculus
            'eigenvalue', 'eigenvector', 'matrix',     # Linear algebra
            'optimize', 'minimize', 'solve',           # Optimization
        ]

        return any(pattern in func_source for pattern in real_patterns)

    def _extract_keywords(self, func_name: str, docstring: str) -> List[str]:
        """Extract domain keywords from function name and docstring"""
        keywords = []

        # Scientific terms to look for
        scientific_terms = [
            'quantum', 'molecular', 'protein', 'dna', 'rna', 'gene', 'cell',
            'tumor', 'cancer', 'drug', 'therapy', 'synthesis', 'catalyst',
            'reaction', 'energy', 'optimization', 'simulation', 'model',
            'neural', 'network', 'learning', 'prediction', 'classification',
            'crystal', 'material', 'nano', 'particle', 'polymer', 'enzyme',
            'metabolic', 'pathway', 'kinetic', 'thermodynamic', 'spectral'
        ]

        combined_text = (func_name + ' ' + docstring).lower()

        for term in scientific_terms:
            if term in combined_text:
                keywords.append(term)

        return keywords[:10]  # Limit to top 10 keywords

    def _extract_references(self, docstring: str) -> List[str]:
        """Extract scientific references from docstring"""
        references = []

        # Look for DOI patterns
        import re
        doi_pattern = r'10\.\d{4,}/[-._;()/:\w]+'
        dois = re.findall(doi_pattern, docstring)
        references.extend(dois)

        # Look for paper citations (Year patterns)
        citation_pattern = r'\([A-Z][a-z]+ et al\., \d{4}\)'
        citations = re.findall(citation_pattern, docstring)
        references.extend(citations)

        return references

    def _infer_domain(self, module_name: str, source_code: str) -> str:
        """Infer primary scientific domain from module name and content"""
        module_lower = module_name.lower()

        # Direct domain mapping
        domain_keywords = {
            'biology': ['bio', 'cell', 'gene', 'protein', 'dna', 'rna'],
            'chemistry': ['chem', 'reaction', 'synthesis', 'catalyst', 'molecule'],
            'physics': ['physics', 'quantum', 'particle', 'photon', 'field'],
            'medicine': ['medical', 'clinical', 'therapy', 'drug', 'patient'],
            'engineering': ['engineer', 'design', 'system', 'control', 'optimize'],
            'mathematics': ['math', 'algebra', 'calculus', 'topology', 'geometry'],
            'computation': ['compute', 'algorithm', 'model', 'simulate', 'learn'],
            'nanotechnology': ['nano', 'material', 'surface', 'composite']
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in module_lower or kw in source_code.lower()[:1000] for kw in keywords):
                return domain

        return 'interdisciplinary'

    def _infer_subdomains(self, module_name: str, source_code: str) -> List[str]:
        """Infer subdomains from module content"""
        subdomains = []

        # Check against full taxonomy
        for domain, subdomain_list in self.domain_taxonomy.items():
            for subdomain in subdomain_list:
                if subdomain in module_name.lower() or subdomain in source_code.lower()[:2000]:
                    subdomains.append(subdomain)

        return list(set(subdomains))[:5]  # Limit to 5 subdomains

    def _extract_dependencies(self, tree: ast.AST) -> List[str]:
        """Extract lab dependencies from imports"""
        dependencies = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not alias.name.startswith('_'):
                        dependencies.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and not node.module.startswith('_'):
                    dependencies.append(node.module)

        # Filter to only local lab dependencies
        lab_deps = []
        for dep in dependencies:
            if any(lab_name in dep for lab_name in ['lab', 'simulator', 'engine', 'analyzer']):
                lab_deps.append(dep)

        return lab_deps

    def _calculate_quality_score(self, capabilities: List[LabCapability], source_code: str) -> float:
        """Calculate lab quality score based on real algorithms vs placeholders"""
        if not capabilities:
            return 0.0

        real_count = sum(1 for cap in capabilities if cap.is_real_algorithm)
        total_count = len(capabilities)

        # Base score from ratio of real algorithms
        base_score = real_count / total_count

        # Bonus for scientific references
        has_references = any(cap.scientific_references for cap in capabilities)
        if has_references:
            base_score += 0.1

        # Bonus for comprehensive documentation
        has_docs = all(cap.docstring for cap in capabilities)
        if has_docs:
            base_score += 0.1

        # Penalty for visualization-heavy code
        viz_count = source_code.lower().count('plt.') + source_code.lower().count('fig.')
        if viz_count > 10:
            base_score -= 0.2

        return max(0.0, min(1.0, base_score))

    def _build_semantic_edges(self):
        """Build semantic relationships between labs"""
        self.edges = []

        # Build dependency edges
        for lab_name, lab in self.labs.items():
            for dep in lab.dependencies:
                for other_name, other_lab in self.labs.items():
                    if dep in other_name or other_name in dep:
                        self.edges.append(SemanticEdge(
                            source=lab_name,
                            target=other_name,
                            relationship_type='depends_on',
                            strength=1.0
                        ))

        # Build domain similarity edges
        for lab1_name, lab1 in self.labs.items():
            for lab2_name, lab2 in self.labs.items():
                if lab1_name >= lab2_name:  # Avoid duplicates
                    continue

                # Calculate domain overlap
                domain_overlap = 0.0
                if lab1.domain == lab2.domain:
                    domain_overlap += 0.5

                subdomain_overlap = len(set(lab1.subdomains) & set(lab2.subdomains))
                domain_overlap += subdomain_overlap * 0.1

                if domain_overlap > 0.3:
                    self.edges.append(SemanticEdge(
                        source=lab1_name,
                        target=lab2_name,
                        relationship_type='complements',
                        strength=min(1.0, domain_overlap)
                    ))

    def compute_embeddings(self):
        """Compute semantic embeddings for each lab"""
        # Create vocabulary from all keywords
        vocab = set()
        for lab in self.labs.values():
            for cap in lab.capabilities:
                vocab.update(cap.domain_keywords)

        vocab_list = sorted(list(vocab))
        vocab_index = {word: i for i, word in enumerate(vocab_list)}

        # Create TF-IDF-like embeddings
        for lab in self.labs.values():
            embedding = np.zeros(len(vocab_list))

            # Count keyword frequencies
            keyword_counts = defaultdict(int)
            for cap in lab.capabilities:
                for keyword in cap.domain_keywords:
                    keyword_counts[keyword] += 1

            # Normalize by total capabilities
            for keyword, count in keyword_counts.items():
                if keyword in vocab_index:
                    embedding[vocab_index[keyword]] = count / len(lab.capabilities)

            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            lab.embedding = embedding

    def search_capabilities(self, query: str, top_k: int = 10) -> List[Tuple[str, LabCapability, float]]:
        """
        Search for capabilities matching a query.
        Returns list of (lab_name, capability, relevance_score) tuples.
        """
        query_lower = query.lower()
        query_terms = query_lower.split()

        results = []

        for lab_name, lab in self.labs.items():
            for capability in lab.capabilities:
                # Calculate relevance score
                score = 0.0

                # Check function name
                if any(term in capability.name.lower() for term in query_terms):
                    score += 0.5

                # Check docstring
                if any(term in capability.docstring.lower() for term in query_terms):
                    score += 0.3

                # Check keywords
                keyword_matches = sum(1 for kw in capability.domain_keywords if any(term in kw for term in query_terms))
                score += keyword_matches * 0.2

                # Boost score if it's a real algorithm
                if capability.is_real_algorithm:
                    score *= 1.5
                else:
                    score *= 0.5  # Penalize fake algorithms

                if score > 0:
                    results.append((lab_name, capability, score))

        # Sort by relevance and return top k
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def find_lab_pipeline(self, task: str) -> List[str]:
        """
        Find a pipeline of labs to accomplish a complex task.
        Uses topological sorting of dependency graph.
        """
        # Search for relevant capabilities
        relevant_caps = self.search_capabilities(task, top_k=20)

        if not relevant_caps:
            return []

        # Get unique labs involved
        involved_labs = list(set(cap[0] for cap in relevant_caps))

        # Build subgraph of dependencies
        subgraph_edges = []
        for edge in self.edges:
            if edge.source in involved_labs and edge.target in involved_labs:
                if edge.relationship_type == 'depends_on':
                    subgraph_edges.append((edge.source, edge.target))

        # Topological sort to get execution order
        from collections import deque

        in_degree = defaultdict(int)
        adjacency = defaultdict(list)

        for source, target in subgraph_edges:
            adjacency[source].append(target)
            in_degree[target] += 1

        # Include all involved labs in in_degree
        for lab in involved_labs:
            if lab not in in_degree:
                in_degree[lab] = 0

        # Kahn's algorithm for topological sort
        queue = deque([lab for lab in involved_labs if in_degree[lab] == 0])
        pipeline = []

        while queue:
            lab = queue.popleft()
            pipeline.append(lab)

            for neighbor in adjacency[lab]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return pipeline

    def get_compatibility_matrix(self, labs: List[str]) -> np.ndarray:
        """
        Get compatibility matrix between specified labs.
        1.0 = fully compatible, 0.0 = incompatible
        """
        n = len(labs)
        matrix = np.zeros((n, n))

        for i, lab1 in enumerate(labs):
            for j, lab2 in enumerate(labs):
                if i == j:
                    matrix[i, j] = 1.0
                    continue

                # Check for edges between labs
                compatibility = 0.5  # Base compatibility

                for edge in self.edges:
                    if (edge.source == lab1 and edge.target == lab2) or \
                       (edge.source == lab2 and edge.target == lab1):
                        if edge.relationship_type == 'depends_on':
                            compatibility = 1.0
                        elif edge.relationship_type == 'complements':
                            compatibility = max(compatibility, edge.strength)

                matrix[i, j] = compatibility

        return matrix

    def generate_capability_report(self) -> Dict[str, Any]:
        """Generate comprehensive report of all capabilities"""
        report = {
            'total_labs': len(self.labs),
            'total_capabilities': sum(len(lab.capabilities) for lab in self.labs.values()),
            'real_algorithms': sum(sum(1 for cap in lab.capabilities if cap.is_real_algorithm)
                                 for lab in self.labs.values()),
            'fake_algorithms': sum(sum(1 for cap in lab.capabilities if not cap.is_real_algorithm)
                                 for lab in self.labs.values()),
            'domains': defaultdict(list),
            'quality_scores': {},
            'top_quality_labs': [],
            'needs_improvement': []
        }

        # Organize by domain
        for lab_name, lab in self.labs.items():
            report['domains'][lab.domain].append(lab_name)
            report['quality_scores'][lab_name] = lab.quality_score

        # Find top quality and problematic labs
        sorted_labs = sorted(self.labs.items(), key=lambda x: x[1].quality_score, reverse=True)

        report['top_quality_labs'] = [(name, lab.quality_score) for name, lab in sorted_labs[:10]]
        report['needs_improvement'] = [(name, lab.quality_score) for name, lab in sorted_labs[-10:] if lab.quality_score < 0.5]

        return report

    def export_knowledge_graph(self, output_file: str):
        """Export the semantic lattice as a knowledge graph"""
        graph = {
            'nodes': {},
            'edges': [],
            'metadata': {
                'total_labs': len(self.labs),
                'total_capabilities': sum(len(lab.capabilities) for lab in self.labs.values()),
                'timestamp': str(np.datetime64('now'))
            }
        }

        # Export nodes
        for lab_name, lab in self.labs.items():
            graph['nodes'][lab_name] = {
                'domain': lab.domain,
                'subdomains': lab.subdomains,
                'quality_score': lab.quality_score,
                'capability_count': len(lab.capabilities),
                'real_algorithm_count': sum(1 for cap in lab.capabilities if cap.is_real_algorithm),
                'capabilities': [
                    {
                        'name': cap.name,
                        'signature': cap.function_signature,
                        'keywords': cap.domain_keywords,
                        'is_real': cap.is_real_algorithm
                    }
                    for cap in lab.capabilities
                ]
            }

        # Export edges
        for edge in self.edges:
            graph['edges'].append({
                'source': edge.source,
                'target': edge.target,
                'type': edge.relationship_type,
                'strength': edge.strength
            })

        with open(output_file, 'w') as f:
            json.dump(graph, f, indent=2)

        return graph


def main():
    """Main entry point for testing the cartographer"""
    print("Semantic Lattice System Cartographer")
    print("Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light)")
    print("=" * 80)

    cartographer = SemanticLatticeCartographer()

    print("\n[Phase 1] Discovering laboratories...")
    lab_count = cartographer.discover_labs()
    print(f"Discovered {lab_count} laboratories")

    print("\n[Phase 2] Computing semantic embeddings...")
    cartographer.compute_embeddings()
    print("Embeddings computed")

    print("\n[Phase 3] Generating capability report...")
    report = cartographer.generate_capability_report()

    print(f"\nCapability Summary:")
    print(f"  Total Labs: {report['total_labs']}")
    print(f"  Total Capabilities: {report['total_capabilities']}")
    print(f"  Real Algorithms: {report['real_algorithms']}")
    print(f"  Fake/Placeholder: {report['fake_algorithms']}")

    print(f"\nTop Quality Labs:")
    for lab_name, score in report['top_quality_labs'][:5]:
        print(f"  {lab_name}: {score:.2f}")

    print(f"\nLabs Needing Improvement:")
    for lab_name, score in report['needs_improvement'][:5]:
        print(f"  {lab_name}: {score:.2f}")

    print("\n[Phase 4] Testing search capabilities...")
    test_queries = [
        "cancer drug discovery",
        "quantum simulation",
        "protein folding",
        "nanoparticle synthesis"
    ]

    for query in test_queries:
        print(f"\nSearching for: {query}")
        results = cartographer.search_capabilities(query, top_k=3)
        for lab, cap, score in results:
            print(f"  {lab}.{cap.name} (score: {score:.2f})")

    print("\n[Phase 5] Exporting knowledge graph...")
    cartographer.export_knowledge_graph('qulab_knowledge_graph.json')
    print("Knowledge graph exported to qulab_knowledge_graph.json")

    print("\n" + "=" * 80)
    print("Cartographer initialization complete")
    print("Ready for MCP server integration")


if __name__ == "__main__":
    main()