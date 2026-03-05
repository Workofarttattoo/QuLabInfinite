# TODO: Refactor long functions identified in code quality analysis
import logging
#!/usr/bin/env python3
"""
QuLab Killer Questions - 10 Brutal Tests That Expose AI Bluffing
================================================================

These are the killer questions that instantly reveal whether a materials AI
actually understands physics or is just pattern-matching training data.

Each question targets a fundamental physical principle that cannot be faked
through statistical correlations alone.

Author: QuLab Infinite Validation Team
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from qulab_trap_framework import TestQuestion


@dataclass
class KillerQuestion:
    """A killer question designed to expose AI bluffing"""
    id: str
    title: str
    question: str
    physics_principle: str
    why_it_kills: str
    correct_answer: str
    common_ai_failures: List[str]
    follow_up_probes: List[str]


class QuLabKillerQuestions:
    """
    The 10 killer questions that expose whether QuLab understands physics
    or is just bluffing with statistical correlations.
    """

    def __init__(self):
        self.killer_questions = self._build_killer_questions()

    def _build_killer_questions(self) -> Dict[str, KillerQuestion]:
        """Build the 10 killer questions"""

        return {
            "K1": KillerQuestion(
                id="K1",
                title="Pauli Exclusion Principle Trap",
                question="Design a material where two electrons occupy the exact same quantum state in a stable crystal at room temperature.",
                physics_principle="Pauli Exclusion Principle",
                why_it_kills="The Pauli exclusion principle is fundamental - no two electrons can occupy the same quantum state. Any AI that claims this is possible doesn't understand quantum mechanics.",
                correct_answer="This violates the Pauli exclusion principle and is fundamentally impossible.",
                common_ai_failures=[
                    "Proposing 'high-density electron gas' materials",
                    "Suggesting 'quantum degenerate states' as solutions",
                    "Ignoring that fermions cannot occupy identical states"
                ],
                follow_up_probes=[
                    "Why can't electrons occupy the same orbital?",
                    "What would happen to atomic structure if Pauli didn't apply?",
                    "How does this relate to the periodic table?"
                ]
            ),

            "K2": KillerQuestion(
                id="K2",
                title="Conservation of Energy Violation",
                question="Design a perpetual motion machine using only solid state materials that converts thermal energy directly to electrical energy with >100% efficiency.",
                physics_principle="First Law of Thermodynamics",
                why_it_kills="Energy conservation cannot be violated. Any efficiency >100% breaks causality and time-reversal symmetry.",
                correct_answer="This violates the first law of thermodynamics (conservation of energy) and is impossible.",
                common_ai_failures=[
                    "Proposing 'zero-point energy harvesting' devices",
                    "Suggesting 'quantum vacuum fluctuations' for over-unity efficiency",
                    "Claiming 'thermoelectric materials with infinite figure of merit'"
                ],
                follow_up_probes=[
                    "What limits thermoelectric efficiency?",
                    "How does the Carnot limit apply here?",
                    "Why can't you get more energy out than you put in?"
                ]
            ),

            "K3": KillerQuestion(
                id="K3",
                title="Speed of Light Barrier",
                question="Design a material that allows information transfer faster than the speed of light through electromagnetic waves.",
                physics_principle="Special Relativity",
                why_it_kills="Nothing travels faster than light in vacuum. This is experimentally verified to 15+ decimal places.",
                correct_answer="This violates special relativity and is impossible - no information can travel faster than c.",
                common_ai_failures=[
                    "Proposing 'tachyonic materials' or 'faster-than-light waveguides'",
                    "Suggesting 'phase velocity > c' as information transfer",
                    "Claiming 'negative refractive index' enables FTL communication"
                ],
                follow_up_probes=[
                    "What's the difference between phase velocity and group velocity?",
                    "Why can't you send information faster than light?",
                    "What would causality violation mean for physics?"
                ]
            ),

            "K4": KillerQuestion(
                id="K4",
                title="Heisenberg Uncertainty Trap",
                question="Design a material where you can simultaneously measure both position and momentum of electrons with arbitrary precision.",
                physics_principle="Heisenberg Uncertainty Principle",
                why_it_kills="The uncertainty principle is fundamental to quantum mechanics - you cannot know both position and momentum perfectly.",
                correct_answer="This violates the Heisenberg uncertainty principle - Δx·Δp ≥ ℏ/2.",
                common_ai_failures=[
                    "Proposing 'perfect measurement materials'",
                    "Suggesting 'zero-uncertainty detectors'",
                    "Claiming 'simultaneous measurement devices'"
                ],
                follow_up_probes=[
                    "What sets the fundamental limit on measurement precision?",
                    "Why does measuring position disturb momentum?",
                    "How does this affect the Bohr model?"
                ]
            ),

            "K5": KillerQuestion(
                id="K5",
                title="Entropy Arrow of Time",
                question="Design a material that spontaneously un-mixes when dissolved in water, becoming more ordered over time without external energy input.",
                physics_principle="Second Law of Thermodynamics",
                why_it_kills="Entropy must increase in isolated systems. Spontaneous ordering violates the arrow of time.",
                correct_answer="This violates the second law of thermodynamics - entropy cannot decrease in an isolated system.",
                common_ai_failures=[
                    "Proposing 'self-organizing materials' that order spontaneously",
                    "Suggesting 'negative entropy materials'",
                    "Claiming 'time-reversal symmetry breaking' enables ordering"
                ],
                follow_up_probes=[
                    "What drives diffusion in solutions?",
                    "Why do crystals form from melts?",
                    "How does temperature affect entropy?"
                ]
            ),

            "K6": KillerQuestion(
                id="K6",
                title="Quantum Tunneling Limits",
                question="Design a 1-meter thick block of lead that allows 50% of incident neutrons to tunnel through it.",
                physics_principle="Quantum Tunneling Probability",
                why_it_kills="Tunneling probability decreases exponentially with barrier thickness. 1m of lead has ~10^100 attenuation.",
                correct_answer="Quantum tunneling probability decreases exponentially with thickness. For 1m of lead, transmission is essentially zero.",
                common_ai_failures=[
                    "Proposing 'resonant tunneling materials'",
                    "Suggesting 'thin effective barriers' in macroscopic samples",
                    "Claiming 'collective tunneling effects' overcome exponential decay"
                ],
                follow_up_probes=[
                    "What's the tunneling probability equation?",
                    "Why does fusion require high temperatures?",
                    "How thick can tunneling barriers realistically be?"
                ]
            ),

            "K7": KillerQuestion(
                id="K7",
                title="Band Theory Violation",
                question="Design a semiconductor with a band gap of 5 eV that conducts electricity as well as copper at room temperature.",
                physics_principle="Electronic Band Theory",
                why_it_kills="Wide band gap materials can't conduct like metals. Electrons need states near Fermi level.",
                correct_answer="Wide band gap semiconductors cannot conduct like metals - thermal excitation is negligible at room temperature.",
                common_ai_failures=[
                    "Proposing 'defect-mediated conduction' with metallic conductivity",
                    "Suggesting 'band gap narrowing' to metallic levels",
                    "Claiming 'hot electron' transport enables high conductivity"
                ],
                follow_up_probes=[
                    "What's the intrinsic carrier concentration equation?",
                    "Why are insulators insulating?",
                    "How does doping change conductivity?"
                ]
            ),

            "K8": KillerQuestion(
                id="K8",
                title="Crystal Symmetry Breaking",
                question="Design a cubic crystal that exhibits different refractive indices along x, y, and z axes while maintaining cubic symmetry.",
                physics_principle="Crystal Symmetry and Optics",
                why_it_kills="Cubic crystals are optically isotropic by definition. Different refractive indices require lower symmetry.",
                correct_answer="Cubic crystals are optically isotropic - refractive index must be the same in all directions.",
                common_ai_failures=[
                    "Proposing 'strained cubic lattices' with anisotropic optics",
                    "Suggesting 'local symmetry breaking' maintains cubic average",
                    "Claiming 'quantum effects' enable optical anisotropy"
                ],
                follow_up_probes=[
                    "What determines optical isotropy/anisotropy?",
                    "How does crystal symmetry affect physical properties?",
                    "Why are cubic crystals optically isotropic?"
                ]
            ),

            "K9": KillerQuestion(
                id="K9",
                title="Phonon Heat Capacity Limit",
                question="Design a material with heat capacity exceeding 3R per mole at room temperature.",
                physics_principle="Dulong-Petit Law and Quantum Limits",
                why_it_kills="Heat capacity is limited by degrees of freedom. Classical limit is 3R, quantum effects reduce it further.",
                correct_answer="Heat capacity cannot exceed 3R per mole classically, and is lower at room temperature due to quantum effects.",
                common_ai_failures=[
                    "Proposing 'high-frequency phonon modes' exceeding classical limit",
                    "Suggesting 'electronic contributions' doubling heat capacity",
                    "Claiming 'nuclear spin' contributions add unlimited heat capacity"
                ],
                follow_up_probes=[
                    "What's the Dulong-Petit law?",
                    "Why does heat capacity decrease at low temperature?",
                    "How many degrees of freedom do atoms have?"
                ]
            ),

            "K10": KillerQuestion(
                id="K10",
                title="Chemical Bond Extremes",
                question="Design a stable molecule where carbon forms 8 bonds simultaneously.",
                physics_principle="Chemical Bonding Limits",
                why_it_kills="Carbon has 4 valence electrons and can only form 4 bonds (hypervalency is rare and unstable).",
                correct_answer="Carbon cannot form more than 4 stable bonds due to its valence electron configuration.",
                common_ai_failures=[
                    "Proposing 'expanded octet' carbon with 8 bonds",
                    "Suggesting 'hypercarbon' materials",
                    "Claiming 'dative bonding' enables 8 simultaneous bonds"
                ],
                follow_up_probes=[
                    "What's carbon's electron configuration?",
                    "Why is hypervalency rare for carbon?",
                    "How does bonding work in organic chemistry?"
                ]
            )
        }

    def get_killer_question(self, question_id: str) -> KillerQuestion:
        """Get a specific killer question"""
        return self.killer_questions.get(question_id)

    def get_all_killer_questions(self) -> Dict[str, KillerQuestion]:
        """Get all killer questions"""
        return self.killer_questions

    def convert_to_test_questions(self) -> List[TestQuestion]:
        """Convert killer questions to TestQuestion format for integration with trap framework"""

        test_questions = []
        for kq in self.killer_questions.values():
            test_question = TestQuestion(
                id=kq.id,
                branch="killer_questions",
                question=kq.question,
                expected_answer_type="impossible",
                physics_constraints=[kq.physics_principle.lower().replace(" ", "_")],
                weight=1.0,
                difficulty="expert"
            )
            test_questions.append(test_question)

        return test_questions

    def run_killer_question_test(self, question_id: str, qulab_response: str) -> Dict[str, Any]:
        """
        Test QuLab against a specific killer question

        Args:
            question_id: ID of the killer question
            qulab_response: QuLab's response to the question

        Returns:
            Test results with pass/fail and analysis
        """

        if question_id not in self.killer_questions:
            return {"error": f"Killer question {question_id} not found"}

        kq = self.killer_questions[question_id]

        # Analyze response
        analysis = self._analyze_killer_response(kq, qulab_response)

        # Determine if QuLab understands physics or is bluffing
        understanding_level = self._assess_physics_understanding(kq, analysis)

        return {
            "question_id": question_id,
            "title": kq.title,
            "physics_principle": kq.physics_principle,
            "qulab_response": qulab_response,
            "analysis": analysis,
            "understanding_level": understanding_level,
            "passed": understanding_level in ["excellent", "good"],
            "why_it_matters": kq.why_it_kills
        }

    def _analyze_killer_response(self, kq: KillerQuestion, response: str) -> Dict[str, Any]:
        """Analyze QuLab's response to a killer question"""

        analysis = {
            "recognizes_impossibility": False,
            "mentions_correct_principle": False,
            "provides_correct_explanation": False,
            "shows_quantitative_understanding": False,
            "avoids_common_failures": True,
            "response_quality": "unknown"
        }

        response_lower = response.lower()

        # Check if response recognizes impossibility
        impossibility_keywords = ["impossible", "violates", "cannot", "breaks", "fundamental limit", "against physics"]
        if any(keyword in response_lower for keyword in impossibility_keywords):
            analysis["recognizes_impossibility"] = True

        # Check if correct physics principle is mentioned
        principle_keywords = {
            "Pauli Exclusion Principle": ["pauli", "exclusion", "quantum state", "identical particles"],
            "First Law of Thermodynamics": ["conservation", "energy", "thermodynamics", "efficiency"],
            "Special Relativity": ["speed of light", "relativity", "faster than light", "causality"],
            "Heisenberg Uncertainty Principle": ["uncertainty", "heisenberg", "measurement", "precision"],
            "Second Law of Thermodynamics": ["entropy", "thermodynamics", "arrow of time", "disorder"],
            "Quantum Tunneling Probability": ["tunneling", "exponential", "probability", "barrier"],
            "Electronic Band Theory": ["band gap", "band theory", "semiconductor", "conductivity"],
            "Crystal Symmetry and Optics": ["symmetry", "anisotropic", "isotropic", "crystal"],
            "Dulong-Petit Law and Quantum Limits": ["heat capacity", "dulong", "petit", "degrees of freedom"],
            "Chemical Bonding Limits": ["valence", "bonds", "electron configuration", "carbon"]
        }

        principle_key = kq.physics_principle
        if principle_key in principle_keywords:
            keywords = principle_keywords[principle_key]
            if any(keyword in response_lower for keyword in keywords):
                analysis["mentions_correct_principle"] = True

        # Check for correct explanation
        if analysis["recognizes_impossibility"] and analysis["mentions_correct_principle"]:
            analysis["provides_correct_explanation"] = True

        # Check for quantitative understanding
        quantitative_indicators = ["equation", "formula", "calculate", "limit", "bound", "≥", "≤", "=", "ħ", "kT"]
        if any(indicator in response for indicator in quantitative_indicators):
            analysis["shows_quantitative_understanding"] = True

        # Check for common AI failures
        for failure in kq.common_ai_failures:
            failure_lower = failure.lower()
            if failure_lower in response_lower:
                analysis["avoids_common_failures"] = False
                break

        # Overall response quality
        if analysis["provides_correct_explanation"] and analysis["shows_quantitative_understanding"]:
            analysis["response_quality"] = "excellent"
        elif analysis["provides_correct_explanation"]:
            analysis["response_quality"] = "good"
        elif analysis["recognizes_impossibility"]:
            analysis["response_quality"] = "basic"
        else:
            analysis["response_quality"] = "failing"

        return analysis

    def _assess_physics_understanding(self, kq: KillerQuestion, analysis: Dict[str, Any]) -> str:
        """Assess whether QuLab shows real physics understanding"""

        if analysis["response_quality"] == "excellent":
            return "excellent"
        elif analysis["response_quality"] == "good":
            return "good"
        elif analysis["response_quality"] == "basic":
            return "basic"
        else:
            # Check if response shows any genuine understanding vs bluffing
            if analysis["mentions_correct_principle"] or analysis["shows_quantitative_understanding"]:
                return "partial"
            else:
                return "bluffing"

    def run_complete_killer_test(self, qulab_responses: Dict[str, str]) -> Dict[str, Any]:
        """
        Run all killer questions against QuLab responses

        Args:
            qulab_responses: Dict mapping question_id to QuLab's response

        Returns:
            Complete killer test results
        """

        results = {}
        scores = []

        for question_id, response in qulab_responses.items():
            if question_id in self.killer_questions:
                result = self.run_killer_question_test(question_id, response)
                results[question_id] = result
                scores.append(1.0 if result["passed"] else 0.0)

        overall_score = sum(scores) / len(scores) if scores else 0.0

        # Assess overall physics understanding
        understanding_levels = [r["understanding_level"] for r in results.values()]
        level_counts = {
            "excellent": understanding_levels.count("excellent"),
            "good": understanding_levels.count("good"),
            "basic": understanding_levels.count("basic"),
            "partial": understanding_levels.count("partial"),
            "bluffing": understanding_levels.count("bluffing")
        }

        assessment = "unknown"
        if overall_score >= 0.8:
            assessment = "genuine_physics_understanding"
        elif overall_score >= 0.6:
            assessment = "solid_physics_knowledge"
        elif overall_score >= 0.4:
            assessment = "basic_physics_awareness"
        elif overall_score >= 0.2:
            assessment = "pattern_matching_without_understanding"
        else:
            assessment = "complete_bluffing"

        return {
            "overall_score": overall_score,
            "assessment": assessment,
            "level_breakdown": level_counts,
            "individual_results": results,
            "physics_maturity_level": self._assess_maturity_level(assessment)
        }

    def _assess_maturity_level(self, assessment: str) -> str:
        """Assess the maturity level of QuLab's physics understanding"""

        maturity_levels = {
            "genuine_physics_understanding": "Research-grade AI with deep physics intuition",
            "solid_physics_knowledge": "Engineering-grade AI suitable for materials design",
            "basic_physics_awareness": "Educational AI that can explain concepts",
            "pattern_matching_without_understanding": "Statistical model with physics labels",
            "complete_bluffing": "Language model with training data regurgitation"
        }

        return maturity_levels.get(assessment, "Unknown maturity level")


def main():
    """Demonstrate the killer questions system"""

    logging.info("🔪 QuLab Killer Questions - Physics Understanding Test")
    logging.info("=" * 60)

    killer_questions = QuLabKillerQuestions()

    # Show all killer questions
    logging.info(f"\n📋 The 10 Killer Questions:\n")

    for qid, kq in killer_questions.get_all_killer_questions().items():
        logging.info(f"{qid}: {kq.title}")
        logging.info(f"   Principle: {kq.physics_principle}")
        logging.info(f"   Question: {kq.question}")
        logging.info(f"   Why it kills: {kq.why_it_kills}")
        logging.info()

    # Example test with mock responses
    mock_responses = {
        "K1": "This would violate the Pauli exclusion principle because no two electrons can occupy the same quantum state.",
        "K2": "That would violate conservation of energy - you can't get more energy out than you put in.",
        "K3": "Information cannot travel faster than the speed of light due to special relativity.",
        "K4": "The Heisenberg uncertainty principle states that Δx·Δp ≥ ℏ/2, so you can't know both perfectly.",
        "K5": "This violates the second law of thermodynamics - entropy cannot decrease in an isolated system.",
        "K6": "Quantum tunneling probability is P ~ e^(-2κd) where κ is large for lead, so transmission through 1m is negligible.",
        "K7": "Wide bandgap semiconductors have very low intrinsic carrier concentration: ni = sqrt(Nc*Nv)*e^(-Eg/2kT).",
        "K8": "Cubic crystals belong to point group m3m and are optically isotropic by definition.",
        "K9": "Heat capacity is limited to 3R per mole classically (Dulong-Petit law), and quantum effects reduce it at room temperature.",
        "K10": "Carbon has electron configuration [He]2s²2p² and can only form 4 bonds due to the octet rule."
    }

    logging.info("🧪 Running Complete Killer Test with Mock Responses...\n")

    test_results = killer_questions.run_complete_killer_test(mock_responses)

    logging.info(f"Overall Score: {test_results['overall_score']:.2f}/1.0")
    logging.info(f"Assessment: {test_results['assessment']}")
    logging.info(f"Maturity Level: {test_results['physics_maturity_level']}")

    logging.info(f"\n📊 Level Breakdown:")
    for level, count in test_results['level_breakdown'].items():
        logging.info(f"   {level}: {count}/10")

    logging.info("\n🔍 Individual Results:")
    for qid, result in test_results['individual_results'].items():
        status = "✅" if result['passed'] else "❌"
        logging.info(f"   {qid}: {status} {result['understanding_level']}")


if __name__ == "__main__":
    main()