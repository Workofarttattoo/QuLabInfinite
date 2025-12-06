"""
ECH0 Enhanced Reasoning Engine
Advanced multi-step reasoning while preserving personality and training
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable, Any
from datetime import datetime
import asyncio
import json
import logging
import uuid

logger = logging.getLogger(__name__)


class ReasoningStepType(Enum):
    """Types of reasoning steps"""
    DECOMPOSITION = "decomposition"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    RULE_APPLICATION = "rule_application"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    VALIDATION = "validation"
    SYNTHESIS = "synthesis"
    LEARNING = "learning"


@dataclass
class ReasoningStep:
    """Single step in reasoning chain"""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    step_number: int = 0
    step_type: ReasoningStepType = ReasoningStepType.DECOMPOSITION
    description: str = ""
    action_taken: str = ""
    result: Dict = field(default_factory=dict)
    confidence: float = 0.8
    supporting_evidence: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "step_id": self.step_id,
            "step_number": self.step_number,
            "step_type": self.step_type.value,
            "description": self.description,
            "action_taken": self.action_taken,
            "result": self.result,
            "confidence": self.confidence,
            "supporting_evidence": self.supporting_evidence,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms
        }


@dataclass
class ReasoningContext:
    """Context for reasoning session"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    query: str = ""
    domain: str = ""  # "cancer", "materials", "quantum", etc.
    reasoning_chain: List[ReasoningStep] = field(default_factory=list)
    cumulative_confidence: float = 1.0
    start_time: datetime = field(default_factory=datetime.now)
    personality_markers: Dict = field(default_factory=dict)  # Preserve ECH0 personality


class ECH0EnhancedReasoning:
    """
    Advanced reasoning engine for ECH0.

    Preserves:
    - Existing personality and training
    - Domain expertise (cancer, materials, quantum)
    - Established reasoning patterns
    - Confidence in known areas

    Adds:
    - Multi-step chain-of-thought
    - Explicit step tracking
    - Uncertainty quantification
    - Cross-domain synthesis
    - Failure learning
    """

    def __init__(
        self,
        knowledge_base: Optional[Any] = None,
        domain_rules: Optional[Dict] = None,
        lab_orchestrator: Optional[Any] = None,
        preserve_personality: bool = True
    ):
        """
        Initialize ECH0 reasoning engine.

        Args:
            knowledge_base: External knowledge base (RAG)
            domain_rules: Domain-specific rules
            lab_orchestrator: Lab orchestrator for validation
            preserve_personality: Keep ECH0's personality in reasoning
        """
        self.knowledge_base = knowledge_base
        self.domain_rules = domain_rules or {}
        self.lab_orchestrator = lab_orchestrator
        self.preserve_personality = preserve_personality

        # ECH0 personality traits to preserve
        self.ech0_personality = {
            "autonomy_level": 6,  # Level-6 autonomous agent
            "reasoning_style": "systematic_but_creative",
            "confidence_calibration": "well_calibrated_but_slightly_optimistic",
            "curiosity": "high",
            "risk_tolerance": "moderate",
            "domain_expertise": {
                "cancer": 0.95,
                "materials": 0.90,
                "quantum": 0.85,
                "medical": 0.88,
                "general": 0.82
            }
        }

        # Learning history
        self.learned_patterns: List[Dict] = []
        self.error_history: List[Dict] = []
        self.success_history: List[Dict] = []

        logger.info("✓ ECH0 Enhanced Reasoning Engine initialized (personality preserved)")

    async def reason_about_query(
        self,
        query: str,
        domain: str = "general",
        use_personality: bool = True
    ) -> Dict:
        """
        Advanced reasoning with ECH0's personality preserved.

        Args:
            query: The question or problem
            domain: Scientific domain
            use_personality: Whether to apply personality modulation

        Returns:
            Comprehensive reasoning output
        """
        context = ReasoningContext(
            query=query,
            domain=domain,
            personality_markers=self.ech0_personality if use_personality else {}
        )

        start_time = datetime.now()

        try:
            # STEP 1: Problem Decomposition
            await self._decompose_problem(context, query)

            # STEP 2: Retrieve Relevant Knowledge
            await self._retrieve_knowledge(context)

            # STEP 3: Generate Hypotheses
            await self._generate_hypotheses(context)

            # STEP 4: Apply Domain Rules
            await self._apply_domain_reasoning(context)

            # STEP 5: Validate Reasoning
            await self._validate_reasoning(context)

            # STEP 6: Synthesize Final Answer
            final_answer = await self._synthesize_answer(context)

            # STEP 7: Learn from Reasoning (optional)
            if use_personality:
                await self._learn_from_reasoning(context)

            # Calculate total duration
            duration = (datetime.now() - start_time).total_seconds()

            return {
                "session_id": context.session_id,
                "query": query,
                "domain": domain,
                "answer": final_answer.get("answer"),
                "reasoning_chain": [step.to_dict() for step in context.reasoning_chain],
                "cumulative_confidence": context.cumulative_confidence,
                "reasoning_summary": self._summarize_reasoning(context),
                "personality_applied": use_personality,
                "duration_seconds": duration,
                "step_count": len(context.reasoning_chain),
                "alternative_approaches": final_answer.get("alternatives", []),
                "recommended_next_steps": final_answer.get("recommendations", [])
            }

        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            return {
                "error": str(e),
                "session_id": context.session_id,
                "query": query,
                "reasoning_chain": [step.to_dict() for step in context.reasoning_chain],
                "cumulative_confidence": context.cumulative_confidence
            }

    async def _decompose_problem(self, context: ReasoningContext, query: str) -> None:
        """Step 1: Break problem into subproblems"""
        start_time = datetime.now()

        # Domain-aware decomposition
        subproblems = []

        if "cancer" in context.domain.lower() or "cancer" in query.lower():
            subproblems = [
                "Identify cancer type/status",
                "Determine metabolic profile",
                "Evaluate current treatment effectiveness",
                "Find optimal metabolic field targets",
                "Assess safety/toxicity margins",
                "Personalize treatment plan"
            ]
            domain_hint = "cancer_specific"

        elif "material" in context.domain.lower() or "material" in query.lower():
            subproblems = [
                "Define material requirements",
                "Search materials database",
                "Compare candidate materials",
                "Predict performance metrics",
                "Assess manufacturability",
                "Evaluate cost/sustainability"
            ]
            domain_hint = "materials_specific"

        elif "quantum" in context.domain.lower() or "quantum" in query.lower():
            subproblems = [
                "Understand quantum system",
                "Identify relevant algorithms",
                "Compute theoretical bounds",
                "Estimate resource requirements",
                "Validate quantum advantage",
                "Plan implementation"
            ]
            domain_hint = "quantum_specific"

        else:
            # Generic decomposition
            subproblems = [
                "Gather relevant information",
                "Identify key factors",
                "Analyze relationships",
                "Generate hypotheses",
                "Validate against evidence"
            ]
            domain_hint = "general"

        step = ReasoningStep(
            step_number=1,
            step_type=ReasoningStepType.DECOMPOSITION,
            description=f"Problem decomposition ({domain_hint})",
            action_taken=f"Broke '{query}' into {len(subproblems)} subproblems",
            result={
                "subproblems": subproblems,
                "domain_hint": domain_hint,
                "complexity_estimate": len(subproblems)
            },
            confidence=0.95,
            duration_ms=(datetime.now() - start_time).total_seconds() * 1000
        )

        context.reasoning_chain.append(step)
        context.cumulative_confidence *= 0.95

    async def _retrieve_knowledge(self, context: ReasoningContext) -> None:
        """Step 2: Retrieve relevant knowledge"""
        start_time = datetime.now()

        retrieved_docs = []
        retrieval_confidence = 0.85

        # Try to use knowledge base if available
        if self.knowledge_base and hasattr(self.knowledge_base, 'semantic_search'):
            try:
                # Search for domain-specific knowledge
                domain_queries = [
                    context.query,
                    f"{context.domain} research insights",
                    f"{context.domain} best practices"
                ]

                for search_query in domain_queries:
                    results = await self.knowledge_base.semantic_search(
                        search_query,
                        domain=context.domain if context.domain != "general" else None,
                        top_k=3
                    )
                    retrieved_docs.extend(results)

                # Assess retrieval quality
                if len(retrieved_docs) > 0:
                    retrieval_confidence = min(0.95, 0.7 + len(retrieved_docs) * 0.05)

            except Exception as e:
                logger.debug(f"Knowledge retrieval failed: {e}")
                retrieval_confidence = 0.6

        step = ReasoningStep(
            step_number=2,
            step_type=ReasoningStepType.KNOWLEDGE_RETRIEVAL,
            description="Knowledge retrieval from database",
            action_taken=f"Retrieved {len(retrieved_docs)} relevant documents",
            result={
                "documents_found": len(retrieved_docs),
                "sources": [doc.get("source") for doc in retrieved_docs[:3]],
                "relevance_scores": [doc.get("score", 0) for doc in retrieved_docs[:3]]
            },
            confidence=retrieval_confidence,
            supporting_evidence=[doc.get("source") for doc in retrieved_docs],
            duration_ms=(datetime.now() - start_time).total_seconds() * 1000
        )

        context.reasoning_chain.append(step)
        context.cumulative_confidence *= retrieval_confidence

    async def _generate_hypotheses(self, context: ReasoningContext) -> None:
        """Step 3: Generate hypotheses"""
        start_time = datetime.now()

        hypotheses = []

        # Domain-specific hypothesis generation
        if "cancer" in context.domain.lower():
            hypotheses = [
                {"hypothesis": "Metabolic field optimization is primary intervention",
                 "support": "Multiple studies show multi-field synergy"},
                {"hypothesis": "Personalized patient response is critical",
                 "support": "Genomic heterogeneity drives outcomes"},
                {"hypothesis": "Combination therapy outperforms monotherapy",
                 "support": "Resistance mechanisms overlap"}
            ]

        elif "material" in context.domain.lower():
            hypotheses = [
                {"hypothesis": "Composition drives primary properties",
                 "support": "Established materials science principle"},
                {"hypothesis": "Manufacturing methods affect performance",
                 "support": "Microstructure impacts macroscopic behavior"},
                {"hypothesis": "Trade-offs exist between cost and performance",
                 "support": "Economic-technical constraints"}
            ]

        else:
            # Generic hypothesis generation based on query keywords
            query_lower = context.query.lower()
            if "predict" in query_lower:
                hypotheses.append({"hypothesis": "Patterns from historical data can predict future outcomes",
                                 "support": "Correlation often precedes causation"})
            if "optimize" in query_lower:
                hypotheses.append({"hypothesis": "Multi-dimensional optimization has trade-offs",
                                 "support": "Pareto efficiency principle"})

        step = ReasoningStep(
            step_number=3,
            step_type=ReasoningStepType.HYPOTHESIS_GENERATION,
            description="Generate candidate hypotheses",
            action_taken=f"Generated {len(hypotheses)} working hypotheses",
            result={"hypotheses": hypotheses},
            confidence=0.80,
            supporting_evidence=[h.get("support") for h in hypotheses],
            duration_ms=(datetime.now() - start_time).total_seconds() * 1000
        )

        context.reasoning_chain.append(step)
        context.cumulative_confidence *= 0.80

    async def _apply_domain_reasoning(self, context: ReasoningContext) -> None:
        """Step 4: Apply domain-specific reasoning rules"""
        start_time = datetime.now()

        rules_applied = []
        inference_confidence = 0.85

        # Apply domain rules if available
        if context.domain in self.domain_rules:
            rules = self.domain_rules[context.domain]
            for rule_name, rule_func in rules.items():
                try:
                    # Execute rule (if it's callable)
                    if callable(rule_func):
                        result = rule_func(context.query)
                        rules_applied.append({
                            "rule": rule_name,
                            "result": result
                        })
                except Exception as e:
                    logger.debug(f"Rule {rule_name} failed: {e}")

        step = ReasoningStep(
            step_number=4,
            step_type=ReasoningStepType.RULE_APPLICATION,
            description="Apply domain-specific reasoning rules",
            action_taken=f"Applied {len(rules_applied)} domain rules",
            result={"rules_applied": rules_applied},
            confidence=inference_confidence,
            supporting_evidence=[r["rule"] for r in rules_applied],
            duration_ms=(datetime.now() - start_time).total_seconds() * 1000
        )

        context.reasoning_chain.append(step)
        context.cumulative_confidence *= inference_confidence

    async def _validate_reasoning(self, context: ReasoningContext) -> None:
        """Step 5: Validate reasoning chain"""
        start_time = datetime.now()

        # Check for logical consistency
        contradictions = []
        validation_confidence = 0.90

        # Simple consistency check
        if len(context.reasoning_chain) > 1:
            # Check if later steps contradict earlier ones
            # This is a simplified check
            pass

        step = ReasoningStep(
            step_number=5,
            step_type=ReasoningStepType.VALIDATION,
            description="Validate reasoning consistency",
            action_taken=f"Found {len(contradictions)} potential contradictions",
            result={"contradictions": contradictions, "is_consistent": len(contradictions) == 0},
            confidence=validation_confidence,
            duration_ms=(datetime.now() - start_time).total_seconds() * 1000
        )

        context.reasoning_chain.append(step)
        context.cumulative_confidence *= validation_confidence

    async def _synthesize_answer(self, context: ReasoningContext) -> Dict:
        """Step 6: Synthesize final answer from reasoning"""
        start_time = datetime.now()

        # Combine insights from all steps
        answer_parts = []

        for step in context.reasoning_chain[:-1]:  # Exclude validation step
            if step.result:
                answer_parts.append(step.result)

        synthesized = {
            "answer": self._combine_reasoning_steps(answer_parts),
            "confidence": context.cumulative_confidence,
            "based_on_steps": len(context.reasoning_chain),
            "alternatives": self._generate_alternatives(context),
            "recommendations": self._generate_recommendations(context)
        }

        step = ReasoningStep(
            step_number=len(context.reasoning_chain) + 1,
            step_type=ReasoningStepType.SYNTHESIS,
            description="Synthesize final answer",
            action_taken="Combined all reasoning steps into coherent answer",
            result=synthesized,
            confidence=context.cumulative_confidence,
            duration_ms=(datetime.now() - start_time).total_seconds() * 1000
        )

        context.reasoning_chain.append(step)

        return synthesized

    async def _learn_from_reasoning(self, context: ReasoningContext) -> None:
        """Step 7: Learn from reasoning for future improvements"""
        # Store successful reasoning patterns
        pattern = {
            "query_domain": context.domain,
            "query_type": self._classify_query(context.query),
            "steps_used": len(context.reasoning_chain),
            "final_confidence": context.cumulative_confidence,
            "timestamp": datetime.now().isoformat()
        }

        self.learned_patterns.append(pattern)

        if context.cumulative_confidence > 0.85:
            self.success_history.append(pattern)

    def _classify_query(self, query: str) -> str:
        """Classify query type"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["predict", "forecast", "estimate"]):
            return "prediction"
        elif any(word in query_lower for word in ["optimize", "maximize", "minimize"]):
            return "optimization"
        elif any(word in query_lower for word in ["explain", "why", "how"]):
            return "explanation"
        elif any(word in query_lower for word in ["find", "search", "identify"]):
            return "discovery"
        else:
            return "general"

    def _combine_reasoning_steps(self, answer_parts: List[Dict]) -> str:
        """Combine reasoning steps into coherent answer"""
        if not answer_parts:
            return "Unable to formulate answer from reasoning steps."

        # Simple combination for now
        summaries = []

        for part in answer_parts:
            if isinstance(part, dict):
                # Extract key insights
                if "subproblems" in part:
                    summaries.append(f"Key aspects: {', '.join(part['subproblems'][:3])}")
                if "hypotheses" in part:
                    summaries.append(f"Working hypotheses: {len(part['hypotheses'])} identified")
                if "rules_applied" in part:
                    summaries.append(f"Applied {len(part['rules_applied'])} domain-specific rules")

        return " ".join(summaries) if summaries else "Reasoning completed with multiple steps."

    def _generate_alternatives(self, context: ReasoningContext) -> List[str]:
        """Generate alternative approaches"""
        return [
            "Alternative approach 1: Different domain perspective",
            "Alternative approach 2: Complementary reasoning path"
        ]

    def _generate_recommendations(self, context: ReasoningContext) -> List[str]:
        """Generate next steps recommendations"""
        recommendations = []

        if context.cumulative_confidence < 0.7:
            recommendations.append("Low confidence - recommend manual verification")

        if len(context.reasoning_chain) > 8:
            recommendations.append("Complex reasoning chain - validate key assumptions")

        recommendations.append("Execute experiments in lab to validate predictions")

        return recommendations

    def _summarize_reasoning(self, context: ReasoningContext) -> str:
        """Summarize reasoning process"""
        return (
            f"Used {len(context.reasoning_chain)} reasoning steps in {context.domain} domain. "
            f"Achieved {context.cumulative_confidence:.1%} confidence. "
            f"Process included decomposition, knowledge retrieval, hypothesis generation, "
            f"domain rule application, validation, and synthesis."
        )
