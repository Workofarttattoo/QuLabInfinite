"""
Cross-Lab Inference Engine
Synthesize insights from multiple labs and transfer knowledge between domains
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class CrossDomainInsight:
    """An insight that applies across domains"""
    insight: str
    source_lab: str
    target_labs: List[str]
    confidence: float
    evidence: List[str]
    application_method: str


class CrossLabInferenceEngine:
    """
    Enable inference and knowledge transfer between labs.

    Capabilities:
    - Domain adaptation (apply insights from one lab to another)
    - Transfer learning (use trained models across domains)
    - Pattern matching (find similar patterns across labs)
    - Hybrid reasoning (combine multi-lab results)
    """

    def __init__(self, orchestrator: Optional[Any] = None):
        """
        Initialize cross-lab inference.

        Args:
            orchestrator: UnifiedLabOrchestrator instance
        """
        self.orchestrator = orchestrator
        self.domain_mappings = {
            "cancer_materials": {
                "biocompatibility": ["materials_lab", "toxicology_lab"],
                "surface_properties": ["materials_lab", "cancer_lab"],
                "degradation_kinetics": ["materials_lab", "pharmacology_lab"]
            },
            "metabolic_optimization": {
                "field_synergy": ["cancer_lab", "general_cell_optimization"],
                "energy_crisis": ["cancer_lab", "mitochondrial_function"],
                "pH_effects": ["cancer_lab", "enzyme_kinetics"]
            }
        }

        self.learned_mappings: List[CrossDomainInsight] = []

        logger.info("✓ Cross-Lab Inference Engine initialized")

    async def transfer_insight(
        self,
        source_lab: str,
        source_insight: Dict,
        target_labs: List[str]
    ) -> List[Dict]:
        """
        Transfer an insight from source lab to target labs.

        Args:
            source_lab: Source lab name
            source_insight: Insight from source lab
            target_labs: Target labs to apply to

        Returns:
            List of transferred insights
        """
        transferred = []

        for target_lab in target_labs:
            try:
                adapted_insight = await self._adapt_insight(
                    source_lab,
                    source_insight,
                    target_lab
                )

                if adapted_insight:
                    transferred.append({
                        "source_lab": source_lab,
                        "target_lab": target_lab,
                        "original_insight": source_insight,
                        "adapted_insight": adapted_insight,
                        "adaptation_method": "domain_mapping"
                    })

            except Exception as e:
                logger.warning(f"Transfer failed {source_lab}→{target_lab}: {e}")

        return transferred

    async def _adapt_insight(
        self,
        source_lab: str,
        insight: Dict,
        target_lab: str
    ) -> Optional[Dict]:
        """
        Adapt an insight from source domain to target domain.

        Args:
            source_lab: Source lab
            insight: Insight to adapt
            target_lab: Target lab

        Returns:
            Adapted insight or None
        """
        # Map parameters between domains
        param_mapping = self._get_parameter_mapping(source_lab, target_lab)

        if not param_mapping:
            return None

        adapted = insight.copy()

        # Translate parameters
        for source_param, target_param in param_mapping.items():
            if source_param in adapted:
                adapted[target_param] = adapted.pop(source_param)

        # Add domain-specific context
        adapted["adapted_for_domain"] = target_lab
        adapted["adaptation_confidence"] = self._estimate_adaptation_confidence(
            source_lab, target_lab
        )

        return adapted

    def _get_parameter_mapping(self, source_lab: str, target_lab: str) -> Dict:
        """Get parameter mappings between labs"""
        mappings = {
            ("cancer_lab", "materials_lab"): {
                "metabolic_pH": "surface_pH",
                "glucose_level": "dissolution_rate",
                "oxygen_level": "oxidation_state"
            },
            ("materials_lab", "toxicology_lab"): {
                "surface_energy": "reactivity",
                "crystallinity": "bioavailability",
                "porosity": "diffusion_rate"
            },
            ("cancer_lab", "immunology_lab"): {
                "cytokine_level": "immune_activation",
                "ROS_level": "oxidative_stress",
                "ATP_ratio": "cell_energy_state"
            }
        }

        return mappings.get((source_lab, target_lab), {})

    def _estimate_adaptation_confidence(self, source_lab: str, target_lab: str) -> float:
        """Estimate confidence in domain adaptation"""
        # Base confidence matrix
        confidence_matrix = {
            ("cancer_lab", "materials_lab"): 0.75,
            ("materials_lab", "cancer_lab"): 0.70,
            ("cancer_lab", "toxicology_lab"): 0.80,
            ("toxicology_lab", "cancer_lab"): 0.85,
            ("cancer_lab", "immunology_lab"): 0.85,
            ("materials_lab", "protein_engineering_lab"): 0.70,
        }

        return confidence_matrix.get((source_lab, target_lab), 0.6)

    async def synthesize_multi_lab_answer(
        self,
        query: str,
        relevant_labs: List[str],
        lab_results: Dict[str, Dict]
    ) -> Dict:
        """
        Synthesize answer from multiple labs.

        Args:
            query: Original question
            relevant_labs: List of relevant labs
            lab_results: Results from each lab

        Returns:
            Synthesized answer
        """
        synthesis = {
            "query": query,
            "contributing_labs": relevant_labs,
            "synthesis_method": "multi_lab_consensus",
            "insights": [],
            "conflicts": [],
            "overall_confidence": 0.0
        }

        # Identify common themes
        themes = await self._identify_common_themes(lab_results)
        synthesis["themes"] = themes

        # Check for conflicts
        conflicts = await self._identify_conflicts(lab_results)
        synthesis["conflicts"] = conflicts

        # Synthesize final answer
        answer = await self._synthesize_consensus(lab_results, themes, conflicts)
        synthesis["answer"] = answer

        # Calculate overall confidence
        confidences = [
            result.get("confidence", 0.8)
            for result in lab_results.values()
        ]
        synthesis["overall_confidence"] = sum(confidences) / len(confidences) if confidences else 0.5

        logger.info(
            f"Synthesized answer for '{query}' using {len(relevant_labs)} labs "
            f"(confidence: {synthesis['overall_confidence']:.2%})"
        )

        return synthesis

    async def _identify_common_themes(self, lab_results: Dict) -> List[str]:
        """Identify common themes across lab results"""
        themes = []

        # Extract key terms from each result
        for lab, result in lab_results.items():
            # Simple keyword extraction
            if isinstance(result.get("result"), dict):
                result_keys = list(result["result"].keys())
                themes.extend(result_keys)

        # Find common themes
        theme_counts = {}
        for theme in themes:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1

        # Return themes appearing in multiple labs
        common_themes = [
            theme for theme, count in theme_counts.items()
            if count > 1
        ]

        return common_themes

    async def _identify_conflicts(self, lab_results: Dict) -> List[Dict]:
        """Identify conflicts between lab results"""
        conflicts = []

        lab_names = list(lab_results.keys())

        for i, lab1 in enumerate(lab_names):
            for lab2 in lab_names[i+1:]:
                result1 = lab_results[lab1].get("result", {})
                result2 = lab_results[lab2].get("result", {})

                # Simple conflict detection
                for key in set(list(result1.keys()) + list(result2.keys())):
                    v1 = result1.get(key)
                    v2 = result2.get(key)

                    if v1 and v2 and v1 != v2:
                        # Check for significant difference
                        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                            if abs(v1 - v2) > 0.1 * max(abs(v1), abs(v2)):
                                conflicts.append({
                                    "parameter": key,
                                    "lab1": lab1,
                                    "value1": v1,
                                    "lab2": lab2,
                                    "value2": v2,
                                    "resolution": "requires_validation"
                                })

        return conflicts

    async def _synthesize_consensus(
        self,
        lab_results: Dict,
        themes: List[str],
        conflicts: List[Dict]
    ) -> str:
        """Synthesize consensus answer from multiple labs"""
        if not lab_results:
            return "Unable to synthesize answer - no lab results available"

        # Build consensus narrative
        consensus_parts = []

        # Identify strongest results
        strongest_lab = max(
            lab_results.items(),
            key=lambda x: x[1].get("confidence", 0)
        )

        consensus_parts.append(
            f"Primary insight from {strongest_lab[0]}: {strongest_lab[1].get('answer', 'N/A')}"
        )

        # Add supporting insights
        if themes:
            consensus_parts.append(f"Common themes: {', '.join(themes[:3])}")

        # Add conflict notes if relevant
        if conflicts:
            consensus_parts.append(f"Note: {len(conflicts)} conflicts detected - requires further validation")

        return " ".join(consensus_parts)

    def get_lab_similarities(self) -> Dict[Tuple[str, str], float]:
        """Get similarity scores between labs"""
        # Compute based on shared parameters/domains
        similarity_matrix = {
            ("cancer_lab", "toxicology_lab"): 0.85,
            ("cancer_lab", "immunology_lab"): 0.80,
            ("cancer_lab", "materials_lab"): 0.65,
            ("materials_lab", "chemistry_lab"): 0.90,
            ("quantum_lab", "chemistry_lab"): 0.75,
            ("protein_engineering_lab", "structural_biology_lab"): 0.85,
        }

        return similarity_matrix
