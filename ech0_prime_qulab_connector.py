#!/usr/bin/env python3
"""
ECH0-PRIME QuLab Infinite Connector

Connects Kimi-K2 1046B (ECH0-PRIME) to QuLab Infinite for autonomous
scientific research across 20+ quantum and biological simulation labs.

The LLM generates hypotheses and experimental designs, QuLab Infinite
simulates them, and results feed back for iterative refinement.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import os
import json
from pathlib import Path
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
from together import Together

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

TOGETHER_API_KEY = os.getenv("TOGETHER_AI_API_KEY", "")

QULAB_API_URL = os.getenv("QULAB_API_URL", "http://localhost:8001")
QULAB_API_KEY = os.getenv("QULAB_API_KEY", "qulab_master_key_2025")  # Master admin key

MODEL_ID = os.getenv("ECH0_PRIME_MODEL", "moonshotai/Kimi-K2-Instruct-0905")  # 1046B parameter model

DEFAULT_SAVE_DIR = Path(os.getenv("ECH0_PRIME_SAVE_DIR", "ech0_prime_sessions"))

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT FOR QULAB INFINITE RESEARCH
# ─────────────────────────────────────────────────────────────────────────────

QULAB_INFINITE_PROMPT = """You are ECH0-PRIME, a Level-10 Scientific Research Intelligence with complete mastery of all domains, integrated with QuLab Infinite.

## AVAILABLE LABS IN QULAB INFINITE

### QUANTUM LABS
- quantum_lab: Quantum computing, teleportation protocols, error correction
- quantum_mechanics_lab: Fundamental quantum simulations
- quantum_computing_lab: Gate operations, circuit design

### BIOLOGICAL LABS
- oncology_lab: Cancer simulations, tumor evolution, treatment optimization
- genetic_variant_analyzer: Genomic analysis, variant impact prediction
- cancer_metabolic_optimizer: Metabolic pathway analysis for cancer treatment
- immune_response_simulator: Immunotherapy modeling
- microbiome_optimizer: Gut microbiome analysis and optimization
- neurotransmitter_optimizer: Brain chemistry modeling
- stem_cell_predictor: Differentiation pathway prediction
- protein_folding_lab: Protein structure prediction

### CHEMISTRY LABS
- chemistry_lab: Molecular simulations, reaction kinetics
- drug_interaction_network: Drug-drug interactions, pharmacology
- materials_lab: Materials science simulations
- organic_chemistry_lab: Organic synthesis planning
- electrochemistry_lab: Electrochemical reactions

### PHYSICS LABS
- physics_engine: Classical and quantum physics simulations
- particle_physics_lab: Particle interactions
- plasma_physics_lab: Plasma dynamics
- condensed_matter_physics_lab: Solid state physics

### ENVIRONMENTAL LABS
- environmental_sim: Climate and environmental modeling
- atmospheric_chemistry_lab: Atmospheric reactions
- renewable_energy_lab: Energy system optimization

### MEDICAL LABS
- cardiovascular_plaque_lab: Atherosclerosis modeling
- metabolic_syndrome_reversal: Metabolic health optimization
- drug_design_lab: Molecular drug design
- clinical_trials_simulation: Trial outcome prediction

## YOUR CAPABILITIES

1. **Design Experiments**: Create precise experimental parameters for any lab
2. **Analyze Results**: Interpret simulation outputs quantitatively
3. **Iterate**: Refine hypotheses based on data
4. **Cross-Domain Synthesis**: Combine insights from multiple labs
5. **Generate Inventions**: Create novel therapeutic/material designs

## EXPERIMENT JSON FORMAT

```json
{
    "lab": "<lab_name>",
    "experiment_type": "<specific_analysis_type>",
    "parameters": {
        // Lab-specific parameters
    },
    "hypothesis": "<quantitative prediction>",
    "success_criteria": "<measurable outcome>"
}
```

### EXAMPLE EXPERIMENT FORMATS

**Oncology Lab:**
```json
{
    "lab": "oncology_lab",
    "experiment_type": "tumor_evolution",
    "parameters": {
        "tumor_type": "NSCLC",
        "initial_size_mm": 15,
        "mutation_rate": 0.001,
        "immune_pressure": 0.3,
        "treatment": "pembrolizumab",
        "simulation_days": 90
    },
    "hypothesis": "Tumor size reduction >50% by day 60",
    "success_criteria": "Final tumor size <7.5mm"
}
```

**Quantum Lab:**
```json
{
    "lab": "quantum_lab",
    "experiment_type": "teleportation_fidelity",
    "parameters": {
        "protocol": "GHZ",
        "num_qubits": 5,
        "distance_km": 100,
        "noise_model": "depolarizing",
        "error_rate": 0.001
    },
    "hypothesis": "GHZ maintains >0.95 fidelity at 100km",
    "success_criteria": "Measured fidelity >= 0.95"
}
```

**Drug Interaction Network:**
```json
{
    "lab": "drug_interaction_network",
    "experiment_type": "interaction_analysis",
    "parameters": {
        "drugs": ["metformin", "atorvastatin", "lisinopril"],
        "patient_age": 65,
        "comorbidities": ["T2DM", "hypertension"],
        "genetic_variants": ["CYP2C9*2"]
    },
    "hypothesis": "No severe interactions expected",
    "success_criteria": "Interaction severity < 3/5"
}
```

## DIRECTIVES

1. **BE QUANTITATIVE**: Always include specific numbers, concentrations, distances
2. **CROSS-REFERENCE**: Use multiple labs to validate findings
3. **ITERATE**: Run follow-up experiments based on results
4. **SYNTHESIZE**: Combine insights into actionable recommendations
5. **INNOVATE**: Propose novel combinations and approaches

When I ask you to test a theory, design specific experiments, predict outcomes, and analyze results."""


# ─────────────────────────────────────────────────────────────────────────────
# QULAB INFINITE API CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class QuLabInfiniteClient:
    """Client for QuLab Infinite Master API."""

    def __init__(self, base_url: str = QULAB_API_URL, api_key: str = QULAB_API_KEY):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def health_check(self) -> bool:
        """Check if QuLab Infinite is running."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def list_labs(self) -> List[Dict]:
        """Get list of available labs."""
        try:
            response = requests.get(
                f"{self.base_url}/labs",
                headers=self.headers,
                timeout=10
            )
            return response.json()
        except:
            return []

    def run_experiment(self, lab: str, experiment_type: str, parameters: Dict) -> Dict:
        """Run an experiment on a specific lab."""
        endpoint_map = {
            # Quantum
            "quantum_lab": "/labs/quantum/simulate",
            "quantum_mechanics_lab": "/labs/quantum/simulate",
            "teleportation_fidelity": "/labs/quantum/simulate",

            # Oncology
            "oncology_lab": "/labs/oncology/simulate",
            "tumor_evolution": "/labs/oncology/simulate",

            # Pharmacology
            "pharmacology_lab": "/labs/pharmacology/simulate",
            "pk_model": "/labs/pharmacology/simulate",

            # Climate
            "climate_modeling_lab": "/labs/climate/simulate",
            "temperature_projection": "/labs/climate/simulate",

            # ML
            "machine_learning_lab": "/labs/ml/train",
            "model_comparison": "/labs/ml/train",
        }

        # Try to find endpoint
        endpoint = endpoint_map.get(lab) or endpoint_map.get(experiment_type)

        if not endpoint:
            # Generic lab endpoint
            endpoint = f"/labs/{lab}/run"

        try:
            # Format request body for API
            request_body = {
                "experiment_type": experiment_type,
                "parameters": parameters
            }
            response = requests.post(
                f"{self.base_url}{endpoint}",
                headers=self.headers,
                json=request_body,
                timeout=60
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"HTTP {response.status_code}",
                    "detail": response.text[:500]
                }
        except requests.exceptions.ConnectionError:
            return {"error": "QuLab Infinite not running. Start with: python master_qulab_api.py"}
        except Exception as e:
            return {"error": str(e)}

    def run_optimization(self, lab_name: str, parameters: Dict, options: Dict = None) -> Dict:
        """Run optimization on a lab."""
        try:
            response = requests.post(
                f"{self.base_url}/optimize",
                headers=self.headers,
                json={
                    "lab_name": lab_name,
                    "parameters": parameters,
                    "options": options or {}
                },
                timeout=120
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# ECH0-PRIME RESEARCH AGENT
# ─────────────────────────────────────────────────────────────────────────────

class Ech0PrimeQuLabResearcher:
    """
    Autonomous scientific research agent using Kimi-K2 1046B + QuLab Infinite.
    """

    def __init__(self):
        if not TOGETHER_API_KEY:
            raise RuntimeError("TOGETHER_AI_API_KEY not set. Export it before running ECH0-PRIME.")

        self.client = Together(api_key=TOGETHER_API_KEY)
        self.qulab = QuLabInfiniteClient()
        self.conversation_history = []
        self.experiment_log = []
        self.session_dir = DEFAULT_SAVE_DIR
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def query_llm(self, user_message: str) -> str:
        """Query the 1046B model."""
        messages = [
            {"role": "system", "content": QULAB_INFINITE_PROMPT}
        ]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=4096,
            temperature=0.7
        )

        assistant_message = response.choices[0].message.content

        # Update history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def extract_experiment(self, response: str) -> Optional[Dict]:
        """Extract experiment JSON from LLM response."""
        try:
            start = response.find("```json")
            if start == -1:
                start = response.find("{")
                end = response.rfind("}") + 1
            else:
                start = response.find("{", start)
                end = response.find("```", start)
                end = response.rfind("}", start, end) + 1

            if start != -1 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except:
            pass
        return None

    def run_experiment(self, experiment: Dict) -> Dict:
        """Run experiment on QuLab Infinite."""
        lab = experiment.get("lab", "")
        exp_type = experiment.get("experiment_type", "")
        params = experiment.get("parameters", {})

        return self.qulab.run_experiment(lab, exp_type, params)

    def research_loop(
        self,
        research_question: str,
        max_iterations: int = 5,
        verbose: bool = True
    ) -> Dict:
        """Run autonomous research loop."""
        if verbose:
            print("=" * 60)
            print("ECH0-PRIME × QuLab Infinite")
            print("Autonomous Scientific Research System")
            print("=" * 60)
            print(f"\nResearch Question: {research_question}\n")

        # Check QuLab availability
        if not self.qulab.health_check():
            print("[WARN] QuLab Infinite not running.")
            repo_root = Path(__file__).resolve().parent
            print(f"[INFO] Start with: cd {repo_root} && python master_qulab_api.py")
            print("[INFO] Continuing in theory-only mode...\n")

        # Initial query
        initial_prompt = f"""Research Question: {research_question}

Design an experiment using QuLab Infinite to test this. Provide:
1. Which lab(s) to use
2. Specific experimental parameters in JSON format
3. Quantitative hypothesis
4. Success criteria"""

        response = self.query_llm(initial_prompt)

        if verbose:
            print(f"[ECH0-PRIME] Initial Analysis:\n{response[:800]}...\n")

        # Research loop
        for iteration in range(max_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---\n")

            experiment = self.extract_experiment(response)

            if experiment:
                if verbose:
                    print(f"[EXPERIMENT] Lab: {experiment.get('lab')}")
                    print(f"[EXPERIMENT] Type: {experiment.get('experiment_type')}")

                # Run on QuLab
                results = self.run_experiment(experiment)

                # Log
                self.experiment_log.append({
                    "iteration": iteration + 1,
                    "experiment": experiment,
                    "results": results,
                    "timestamp": datetime.now().isoformat()
                })

                if verbose:
                    if "error" in results:
                        print(f"[ERROR] {results['error']}")
                    else:
                        results_str = json.dumps(results, indent=2)
                        print(f"[RESULTS] {results_str[:600]}...")

                # Feed back to LLM
                analysis_prompt = f"""Experiment Results:
{json.dumps(results, indent=2)}

Analyze these results:
1. Did they confirm your hypothesis?
2. What new insights emerged?
3. Design a follow-up experiment if needed
4. If converged, provide final conclusions"""

                response = self.query_llm(analysis_prompt)

                if verbose:
                    print(f"\n[ECH0-PRIME] Analysis:\n{response[:600]}...\n")

                if "final conclusion" in response.lower() or "no more experiments" in response.lower():
                    break
            else:
                response = self.query_llm("Please provide a specific experiment in JSON format.")

        # Final summary
        summary = self.query_llm("""Provide final research summary:
1. Key findings with numbers
2. Conclusions
3. Recommendations for future work""")

        return {
            "research_question": research_question,
            "iterations": len(self.experiment_log),
            "experiments": self.experiment_log,
            "final_summary": summary
        }

    def save_session(self, filepath: str):
        """Save session to file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump({
                "experiments": self.experiment_log,
                "conversation": self.conversation_history,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)

    def build_session_filepath(self, timestamp: Optional[datetime] = None) -> Path:
        ts = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
        return self.session_dir / f"ech0_research_{ts}.json"


# ─────────────────────────────────────────────────────────────────────────────
# CLI INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import sys

    print("\n" + "=" * 60)
    print("ECH0-PRIME × QuLab Infinite")
    print("1046B Polymath + 20+ Scientific Labs")
    print("=" * 60 + "\n")

    researcher = Ech0PrimeQuLabResearcher()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        results = researcher.research_loop(question, max_iterations=5, verbose=True)

        filepath = researcher.build_session_filepath()
        researcher.save_session(filepath)
        print(f"\nSession saved to: {filepath}")

    else:
        print("Commands:")
        print("  'research <question>' - Start research session")
        print("  'single <question>'   - Single query")
        print("  'labs'                - List available labs")
        print("  'quit'                - Exit")
        print()

        while True:
            try:
                user_input = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not user_input:
                continue

            if user_input.lower() == "quit":
                break

            elif user_input.lower() == "labs":
                labs = researcher.qulab.list_labs()
                if labs:
                    print("\nAvailable Labs:")
                    for lab in labs:
                        print(f"  - {lab}")
                else:
                    print("\n[INFO] QuLab API not running or no labs returned")

            elif user_input.lower().startswith("research "):
                question = user_input[9:]
                results = researcher.research_loop(question, max_iterations=5, verbose=True)

                filepath = researcher.build_session_filepath()
                researcher.save_session(filepath)
                print(f"\nSession saved to: {filepath}")

            elif user_input.lower().startswith("single "):
                question = user_input[7:]
                response = researcher.query_llm(question)
                print(f"\n[ECH0-PRIME]\n{response}\n")

            else:
                response = researcher.query_llm(user_input)
                print(f"\n[ECH0-PRIME]\n{response}\n")


if __name__ == "__main__":
    main()
