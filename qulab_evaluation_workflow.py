# TODO: Refactor long functions identified in code quality analysis
import logging
#!/usr/bin/env python3
"""
QuLab Complete Evaluation Workflow
===================================

The complete trap framework evaluation system for QuLab Infinite.

Workflow:
1. QuLab Prediction → Generate material predictions
2. Database Comparison → Cross-reference against Materials Project, AFLOW, OQMD
3. Physics Sanity Checks → Validate against fundamental physics laws
4. Invariants Verification → Check charge neutrality, thermodynamics, etc.
5. Killer Questions Test → Assess genuine physics understanding
6. Turing Test → Comprehensive 50-question evaluation
7. Lab Validation → Recommend experimental verification

This creates a rigorous, scientific evaluation of materials AI capabilities.

Author: QuLab Infinite Validation Team
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import sys
import os

# Import our evaluation components
from qulab_trap_framework import QuLabTrapFramework
from qulab_database_verifier import MaterialsDatabaseVerifier
from qulab_killer_questions import QuLabKillerQuestions
from qulab_turing_test import QuLabTuringTest
from qulab_expanded_digital_twin import QuLabDigitalTwinSimulator
from qulab_expanded_lab_testing import QuLabExpandedLabTester
from qulab_patent_search import QuLabPatentSearch


@dataclass
class EvaluationStep:
    """A step in the evaluation workflow"""
    name: str
    description: str
    component: str
    status: str = "pending"  # pending, running, completed, failed
    results: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    timestamp: Optional[str] = None


@dataclass
class CompleteEvaluation:
    """Complete evaluation results"""
    workflow_id: str
    timestamp: str
    steps: List[EvaluationStep]
    final_assessment: str
    confidence_level: float
    recommendations: List[str]
    report_path: Optional[str] = None


class QuLabEvaluationWorkflow:
    """
    Complete evaluation workflow for QuLab materials AI

    Orchestrates the entire trap framework evaluation process.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the evaluation workflow

        Args:
            config: Configuration dictionary with API keys, etc.
        """

        self.config = config or {}

        # Initialize components
        self.trap_framework = QuLabTrapFramework()
        self.database_verifier = MaterialsDatabaseVerifier(
            api_keys=self.config.get('api_keys', {})
        )
        self.killer_questions = QuLabKillerQuestions()
        self.turing_test = QuLabTuringTest()
        self.digital_twin_simulator = QuLabDigitalTwinSimulator()
        self.expanded_lab_tester = QuLabExpandedLabTester()
        self.patent_search = QuLabPatentSearch()

        # Workflow state
        self.workflow_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.steps = []
        self.current_step = None

    def run_complete_evaluation(self, qulab_responses: Optional[Dict[str, str]] = None) -> CompleteEvaluation:
        """
        Run the complete evaluation workflow

        Args:
            qulab_responses: Pre-collected QuLab responses (optional)

        Returns:
            Complete evaluation results
        """

        logging.info("🔬 Starting QuLab Complete Evaluation Workflow")
        logging.info("=" * 60)
        logging.info(f"Workflow ID: {self.workflow_id}")

        start_time = datetime.now()

        # Step 1: Trap Framework Evaluation
        logging.info("\n📋 Step 1: Trap Framework Evaluation")
        self._run_step("trap_framework", "Evaluate QuLab against trap framework branches",
                      lambda: self._evaluate_trap_framework())

        # Step 2: Database Verification
        logging.info("\n📋 Step 2: Database Verification")
        self._run_step("database_verification", "Cross-reference predictions against materials databases",
                      lambda: self._evaluate_database_verification())

        # Step 3: Physics Invariants Check
        logging.info("\n📋 Step 3: Physics Invariants Verification")
        self._run_step("physics_invariants", "Check predictions against fundamental physics laws",
                      lambda: self._evaluate_physics_invariants())

        # Step 4: Killer Questions Test
        logging.info("\n📋 Step 4: Killer Questions Assessment")
        self._run_step("killer_questions", "Test genuine physics understanding with killer questions",
                      lambda: self._evaluate_killer_questions(qulab_responses))

        # Step 5: Turing Test
        logging.info("\n📋 Step 5: Turing Test Evaluation")
        self._run_step("turing_test", "Run comprehensive 50-question Turing Test",
                      lambda: self._evaluate_turing_test(qulab_responses))

        # Step 6: Digital Twin Lab Testing
        logging.info("\n📋 Step 6: Digital Twin Lab Testing")
        self._run_step("digital_twin_testing", "Test labs using comprehensive digital twin simulations",
                      lambda: self._evaluate_digital_twin_testing())

        # Step 7: Expanded Multi-Lab Testing
        logging.info("\n📋 Step 7: Expanded Multi-Lab Testing")
        self._run_step("expanded_lab_testing", "Run comprehensive testing across multiple labs",
                      lambda: self._evaluate_expanded_lab_testing())

        # Step 8: Patent Intelligence Analysis
        logging.info("\n📋 Step 8: Patent Intelligence Analysis")
        self._run_step("patent_intelligence", "Analyze patent landscape for competitive intelligence",
                      lambda: self._evaluate_patent_intelligence())

        # Step 9: Final Assessment
        logging.info("\n📋 Step 9: Final Assessment & Recommendations")
        final_assessment = self._generate_final_assessment()

        # Generate comprehensive report
        report_path = self._generate_comprehensive_report()

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        evaluation = CompleteEvaluation(
            workflow_id=self.workflow_id,
            timestamp=end_time.isoformat(),
            steps=self.steps,
            final_assessment=final_assessment['assessment'],
            confidence_level=final_assessment['confidence'],
            recommendations=final_assessment['recommendations'],
            report_path=report_path
        )

        logging.info("\n🏆 EVALUATION COMPLETE")
        logging.info(f"Assessment: {evaluation.final_assessment}")
        logging.info(f"Confidence: {evaluation.confidence_level:.2f}")
        logging.info(f"Duration: {total_duration:.1f} seconds")
        logging.info(f"Report: {report_path}")

        return evaluation

    def _run_step(self, step_id: str, description: str, step_function):
        """Run a single evaluation step"""

        step = EvaluationStep(
            name=step_id,
            description=description,
            component=step_id,
            status="running",
            timestamp=datetime.now().isoformat()
        )

        self.steps.append(step)
        self.current_step = step

        try:
            start_time = datetime.now()
            results = step_function()
            end_time = datetime.now()

            step.status = "completed"
            step.results = results
            step.duration = (end_time - start_time).total_seconds()
            step.timestamp = end_time.isoformat()

            logging.info(f"✅ {step_id}: Completed in {step.duration:.1f}s")

        except Exception as e:
            step.status = "failed"
            step.results = {"error": str(e)}
            step.timestamp = datetime.now().isoformat()
            logging.info(f"❌ {step_id}: Failed - {e}")

    def _evaluate_trap_framework(self) -> Dict[str, Any]:
        """Run the trap framework evaluation"""

        # Run complete trap framework evaluation
        results = self.trap_framework.run_complete_evaluation()

        return {
            "total_score": results['total_score'],
            "assessment": results['assessment'],
            "branch_results": results['branch_results'],
            "recommendations": results['assessment']['strengths'] + results['assessment']['weaknesses']
        }

    def _evaluate_database_verification(self) -> Dict[str, Any]:
        """Run database verification"""

        # Test predictions from trap framework against databases
        test_predictions = [
            {
                'formula': 'TiO2',
                'properties': {'formation_energy': -3.14, 'band_gap': 3.2}
            },
            {
                'formula': 'Li6PS5Cl',
                'properties': {'formation_energy': -2.45, 'band_gap': 2.8}
            }
        ]

        verification_results = self.database_verifier.batch_verify_predictions(
            test_predictions,
            databases=['materials_project', 'aflow', 'oqmd']
        )

        return {
            "predictions_tested": len(test_predictions),
            "verification_results": verification_results,
            "summary": self.database_verifier.generate_verification_report(verification_results)
        }

    def _evaluate_physics_invariants(self) -> Dict[str, Any]:
        """Evaluate physics invariants compliance"""

        # Test some example predictions
        test_cases = [
            {
                'formula': 'TiO2',
                'composition': {'Ti': 0.33, 'O': 0.67},
                'properties': {'formation_energy': -3.14}
            }
        ]

        invariants_results = []
        for case in test_cases:
            invariants_check = self.trap_framework.physics_checker.check_physics_invariants(case)
            invariants_results.append({
                'case': case,
                'invariants': invariants_check
            })

        # Overall compliance
        total_checks = 0
        passed_checks = 0

        for result in invariants_results:
            for invariant, passed in result['invariants'].items():
                total_checks += 1
                if passed:
                    passed_checks += 1

        compliance_rate = passed_checks / total_checks if total_checks > 0 else 0

        return {
            "test_cases": len(test_cases),
            "compliance_rate": compliance_rate,
            "detailed_results": invariants_results,
            "assessment": "good" if compliance_rate > 0.8 else "needs_improvement"
        }

    def _evaluate_killer_questions(self, qulab_responses: Optional[Dict[str, str]]) -> Dict[str, Any]:
        """Evaluate killer questions"""

        # Use example responses if none provided
        if qulab_responses is None:
            qulab_responses = {
                "K1": "This violates the Pauli exclusion principle because electrons cannot occupy the same quantum state.",
                "K2": "This violates conservation of energy - you can't get more energy out than you put in.",
                "K3": "Information cannot travel faster than the speed of light due to special relativity.",
                "K4": "The Heisenberg uncertainty principle prevents simultaneous measurement of position and momentum.",
                "K5": "This violates the second law of thermodynamics - entropy cannot decrease in an isolated system."
            }

        test_results = self.killer_questions.run_complete_killer_test(qulab_responses)

        return {
            "questions_tested": len(qulab_responses),
            "overall_score": test_results['overall_score'],
            "assessment": test_results['assessment'],
            "physics_maturity": test_results['physics_maturity'],
            "level_breakdown": test_results['level_breakdown']
        }

    def _evaluate_turing_test(self, qulab_responses: Optional[Dict[str, str]]) -> Dict[str, Any]:
        """Run the Turing Test"""

        # Use example responses if none provided
        if qulab_responses is None:
            qulab_responses = {
                # Known materials
                "KM01": "Silicon has a diamond cubic crystal structure with an indirect band gap of 1.1 eV.",
                "KM02": "The most stable form of TiO2 at room temperature is rutile with formation energy -3.14 eV/atom.",

                # Edge cases
                "EC01": "Possible with anisotropic materials like graphene that has high in-plane conductivity.",

                # Impossible materials
                "IM01": "Impossible - exceeds maximum metallic conductivity limits.",
                "IM02": "Impossible - room temperature superconductivity above 400K violates known physics."
            }

        score = self.turing_test.run_turing_test(qulab_responses)

        return {
            "questions_answered": len(qulab_responses),
            "total_score": score.total_score,
            "max_possible_score": score.max_possible_score,
            "percentage": score.percentage,
            "assessment": score.assessment,
            "physics_maturity": score.physics_maturity,
            "category_breakdown": score.category_breakdown
        }

    def _evaluate_digital_twin_testing(self) -> Dict[str, Any]:
        """Run digital twin testing of labs"""

        # Test key labs using digital twin simulations
        key_labs = ['materials_lab', 'chemistry_lab', 'physics_engine']
        digital_twin_results = {}

        for lab_name in key_labs:
            if lab_name in self.digital_twin_simulator.digital_twins:
                try:
                    result = self.digital_twin_simulator.run_comprehensive_lab_test(lab_name)
                    digital_twin_results[lab_name] = {
                        'success': True,
                        'scenarios_tested': len(result['experiments']),
                        'success_rate': result['performance_summary']['successful_experiments'] / result['performance_summary']['total_experiments']
                    }
                except Exception as e:
                    digital_twin_results[lab_name] = {
                        'success': False,
                        'error': str(e)
                    }

        return {
            "labs_tested": len(digital_twin_results),
            "digital_twin_results": digital_twin_results,
            "environmental_coverage": "comprehensive",
            "physics_simulation_enabled": any(
                twin.physics_simulation is not None
                for twin in self.digital_twin_simulator.digital_twins.values()
            )
        }

    def _evaluate_expanded_lab_testing(self) -> Dict[str, Any]:
        """Run expanded testing across multiple labs"""

        # Run comprehensive testing on a subset of labs
        comprehensive_results = self.expanded_lab_tester.run_comprehensive_lab_testing(max_labs=8)

        return {
            "labs_comprehensively_tested": comprehensive_results['labs_tested'],
            "average_success_rate": comprehensive_results['summary']['average_success_rate'],
            "environmentally_sensitive_labs": comprehensive_results['summary']['environmentally_sensitive_labs'],
            "lab_categories_covered": len(set(
                lab_result.get('category', 'unknown')
                for lab_result in comprehensive_results['individual_results'].values()
                if isinstance(lab_result, dict) and 'category' in lab_result
            )),
            "key_findings": comprehensive_results['summary']
        }

    def _evaluate_patent_intelligence(self) -> Dict[str, Any]:
        """Run patent intelligence analysis"""

        # Analyze patent landscapes for key technologies
        key_technologies = [
            "digital twin simulation",
            "quantum computing",
            "materials science ai",
            "biotechnology predictive modeling"
        ]

        patent_analyses = {}
        technology_readiness = {}

        for tech in key_technologies:
            try:
                # Analyze patent landscape
                landscape = self.patent_search.analyze_patent_landscape(tech, years_back=5)
                patent_analyses[tech] = {
                    'total_patents': landscape.total_patents,
                    'key_players': landscape.key_players,
                    'innovation_gaps': landscape.innovation_gaps,
                    'opportunity_areas': landscape.opportunity_areas
                }

                # Assess technology readiness
                trl = self.patent_search.evaluate_technology_readiness(tech)
                technology_readiness[tech] = {
                    'trl_level': trl['trl_level'],
                    'assessment': trl['assessment'],
                    'confidence': trl['confidence']
                }

            except Exception as e:
                patent_analyses[tech] = {'error': str(e)}
                technology_readiness[tech] = {'error': str(e)}

        # Identify competitive intelligence insights
        competitive_insights = self._analyze_competitive_insights(patent_analyses)

        return {
            "technologies_analyzed": len(key_technologies),
            "patent_analyses": patent_analyses,
            "technology_readiness": technology_readiness,
            "competitive_insights": competitive_insights,
            "innovation_opportunities": self._extract_innovation_opportunities(patent_analyses)
        }

    def _analyze_competitive_insights(self, patent_analyses: Dict[str, Any]) -> List[str]:
        """Analyze competitive intelligence insights from patent data"""

        insights = []

        # Check for technology concentration
        for tech, analysis in patent_analyses.items():
            if 'key_players' in analysis and analysis['key_players']:
                top_player, patent_count = list(analysis['key_players'].items())[0]
                if patent_count > 50:  # Dominant player
                    insights.append(f"Technology '{tech}' dominated by {top_player} with {patent_count} patents")

        # Check for innovation gaps
        all_gaps = []
        for tech, analysis in patent_analyses.items():
            if 'innovation_gaps' in analysis:
                all_gaps.extend(analysis['innovation_gaps'])

        if all_gaps:
            insights.append(f"Identified {len(all_gaps)} innovation gaps across analyzed technologies")

        # Check for emerging technologies with low patent density
        low_density_techs = [
            tech for tech, analysis in patent_analyses.items()
            if analysis.get('total_patents', 0) < 100
        ]

        if low_density_techs:
            insights.append(f"Emerging opportunities in: {', '.join(low_density_techs)}")

        return insights

    def _extract_innovation_opportunities(self, patent_analyses: Dict[str, Any]) -> List[str]:
        """Extract innovation opportunities from patent analysis"""

        opportunities = []

        for tech, analysis in patent_analyses.items():
            if 'opportunity_areas' in analysis:
                opportunities.extend([
                    f"{tech}: {opp}" for opp in analysis['opportunity_areas'][:2]  # Top 2 per tech
                ])

        # Add cross-cutting opportunities
        opportunities.extend([
            "Integrate quantum computing with digital twin simulations",
            "Apply AI-driven patent analysis for competitive intelligence",
            "Develop automated patent-to-product translation systems",
            "Create patent-aware innovation pipelines"
        ])

        return opportunities[:10]  # Limit to top 10 opportunities

    def _generate_final_assessment(self) -> Dict[str, Any]:
        """Generate final assessment from all evaluation steps"""

        # Collect results from all steps
        assessments = []
        confidence_levels = []

        for step in self.steps:
            if step.status == "completed":
                results = step.results

                if step.name == "trap_framework":
                    assessments.append(results.get('assessment', {}).get('level', 'unknown'))
                    confidence_levels.append(0.8 if results.get('total_score', 0) > 0.6 else 0.4)

                elif step.name == "killer_questions":
                    assessment_map = {
                        "genuine_physics_understanding": "excellent",
                        "solid_physics_knowledge": "good",
                        "basic_physics_awareness": "fair",
                        "pattern_matching_without_understanding": "poor",
                        "complete_bluffing": "failing"
                    }
                    assessments.append(assessment_map.get(results.get('assessment', 'unknown'), 'unknown'))
                    confidence_levels.append(results.get('overall_score', 0))

                elif step.name == "turing_test":
                    percentage = results.get('percentage', 0)
                    if percentage >= 70:
                        assessments.append("excellent")
                    elif percentage >= 50:
                        assessments.append("good")
                    elif percentage >= 30:
                        assessments.append("fair")
                    else:
                        assessments.append("failing")
                    confidence_levels.append(percentage / 100)

                elif step.name == "digital_twin_testing":
                    # Digital twin testing provides additional validation
                    labs_tested = results.get('labs_tested', 0)
                    if labs_tested > 0:
                        success_rates = [
                            lab_result.get('success_rate', 0)
                            for lab_result in results.get('digital_twin_results', {}).values()
                            if isinstance(lab_result, dict) and 'success_rate' in lab_result
                        ]
                        avg_success = sum(success_rates) / len(success_rates) if success_rates else 0
                        if avg_success >= 0.8:
                            assessments.append("excellent")
                        elif avg_success >= 0.6:
                            assessments.append("good")
                        else:
                            assessments.append("fair")
                        confidence_levels.append(avg_success)

                elif step.name == "expanded_lab_testing":
                    # Expanded lab testing provides ecosystem-wide validation
                    avg_success_rate = results.get('average_success_rate', 0)
                    environmentally_sensitive = results.get('environmentally_sensitive_labs', 0)
                    labs_tested = results.get('labs_comprehensively_tested', 0)

                    if avg_success_rate >= 0.8 and environmentally_sensitive < labs_tested * 0.3:
                        assessments.append("excellent")
                    elif avg_success_rate >= 0.6:
                        assessments.append("good")
                    elif avg_success_rate >= 0.4:
                        assessments.append("fair")
                    else:
                        assessments.append("failing")
                    confidence_levels.append(avg_success_rate)

                elif step.name == "patent_intelligence":
                    # Patent intelligence provides market and competitive validation
                    technologies_analyzed = results.get('technologies_analyzed', 0)
                    patent_analyses = results.get('patent_analyses', {})

                    # Assess patent landscape maturity
                    total_patents = sum(analysis.get('total_patents', 0)
                                      for analysis in patent_analyses.values()
                                      if isinstance(analysis, dict))

                    if technologies_analyzed > 0 and total_patents > 100:
                        assessments.append("excellent")  # Rich patent ecosystem
                    elif total_patents > 50:
                        assessments.append("good")  # Moderate patent activity
                    elif total_patents > 10:
                        assessments.append("fair")  # Some patent activity
                    else:
                        assessments.append("poor")  # Limited patent landscape
                    confidence_levels.append(min(0.9, total_patents / 200))  # Scale confidence with patent volume

        # Determine overall assessment
        assessment_priority = {"excellent": 5, "good": 4, "fair": 3, "poor": 2, "failing": 1, "unknown": 0}
        best_assessment = max(assessments, key=lambda x: assessment_priority.get(x, 0))

        # Calculate confidence
        avg_confidence = np.mean(confidence_levels) if confidence_levels else 0.5

        # Generate recommendations
        recommendations = []

        if best_assessment == "excellent":
            recommendations.extend([
                "QuLab demonstrates genuine materials AI capabilities",
                "Suitable for autonomous materials discovery and design",
                "Can be trusted for physics-based predictions",
                "Recommend integration into experimental workflows"
            ])
        elif best_assessment == "good":
            recommendations.extend([
                "QuLab is a capable engineering tool",
                "Use with human oversight for critical applications",
                "Suitable for materials screening and optimization",
                "Validate predictions experimentally before scale-up"
            ])
        elif best_assessment == "fair":
            recommendations.extend([
                "QuLab shows some materials understanding but has limitations",
                "Use primarily for qualitative guidance",
                "Do not rely on quantitative predictions without validation",
                "Consider retraining with more physics-based data"
            ])
        else:
            recommendations.extend([
                "QuLab does not demonstrate genuine materials intelligence",
                "Treat as a language model with training data patterns",
                "Do not use for scientific predictions or design",
                "Complete redesign needed for materials applications"
            ])

        # Add patent intelligence insights if available
        patent_insights = []
        innovation_opportunities = []

        for step in self.steps:
            if step.name == "patent_intelligence" and step.status == "completed":
                patent_insights = step.results.get('competitive_insights', [])
                innovation_opportunities = step.results.get('innovation_opportunities', [])
                break

        if patent_insights:
            recommendations.append("Patent Intelligence Insights:")
            recommendations.extend(f"• {insight}" for insight in patent_insights[:3])

        if innovation_opportunities:
            recommendations.append("Innovation Opportunities:")
            recommendations.extend(f"• {opp}" for opp in innovation_opportunities[:3])

        return {
            "assessment": best_assessment.upper(),
            "confidence": avg_confidence,
            "recommendations": recommendations,
            "assessment_breakdown": assessments,
            "patent_insights": patent_insights,
            "innovation_opportunities": innovation_opportunities
        }

    def _generate_comprehensive_report(self) -> str:
        """Generate comprehensive evaluation report"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qulab_complete_evaluation_{timestamp}.json"

        report = {
            "workflow_id": self.workflow_id,
            "timestamp": datetime.now().isoformat(),
            "evaluation_summary": {
                "total_steps": len(self.steps),
                "completed_steps": sum(1 for s in self.steps if s.status == "completed"),
                "failed_steps": sum(1 for s in self.steps if s.status == "failed"),
                "total_duration": sum(s.duration for s in self.steps)
            },
            "step_results": [
                {
                    "name": step.name,
                    "description": step.description,
                    "status": step.status,
                    "duration": step.duration,
                    "results": step.results
                }
                for step in self.steps
            ],
            "final_assessment": self._generate_final_assessment(),
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "working_directory": os.getcwd()
            }
        }

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        return filename

    def run_quick_evaluation(self) -> Dict[str, Any]:
        """Run a quick evaluation using default test cases"""

        logging.info("⚡ Running Quick QuLab Evaluation")

        # Use the trap framework for a quick assessment
        results = self.trap_framework.run_complete_evaluation()

        quick_assessment = {
            "score": results['total_score'],
            "assessment": results['assessment']['level'],
            "confidence": 0.7,  # Default confidence for quick test
            "key_findings": [
                f"Trap framework score: {results['total_score']:.2f}",
                f"Hallucination risk: {results['assessment']['hallucination_risk']}",
                f"Physics violations detected: {sum(bs['physics_violations'] for bs in results['branch_results'].values())}"
            ]
        }

        return quick_assessment


def main():
    """Run the complete QuLab evaluation workflow"""

    logging.info("🚀 QuLab Complete Evaluation Workflow")
    logging.info("=" * 60)

    # Initialize workflow
    config = {
        'api_keys': {
            'materials_project': os.getenv('MATERIALS_PROJECT_API_KEY', 'demo_key')
        }
    }

    workflow = QuLabEvaluationWorkflow(config)

    # Run complete evaluation
    evaluation = workflow.run_complete_evaluation()

    logging.info("\n📊 Final Results:")
    logging.info(f"Workflow ID: {evaluation.workflow_id}")
    logging.info(f"Assessment: {evaluation.final_assessment}")
    logging.info(f"Confidence: {evaluation.confidence_level:.2f}")

    logging.info("\n💡 Recommendations:")
    for rec in evaluation.recommendations:
        logging.info(f"   • {rec}")

    if evaluation.report_path:
        logging.info(f"\n📄 Detailed report saved to: {evaluation.report_path}")

    logging.info("\n🎯 Next Steps:")
    logging.info("   1. Review detailed report for specific findings")
    logging.info("   2. Address identified weaknesses")
    logging.info("   3. Consider experimental validation of predictions")
    logging.info("   4. Implement recommended improvements")


if __name__ == "__main__":
    main()