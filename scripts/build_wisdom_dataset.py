import os
import json
import re

def extract_training_material():
    training_file = "qulab/ech0/ech0_cancer_research_training.py"
    if not os.path.exists(training_file):
        return ""

    with open(training_file, "r") as f:
        content = f.read()

    # Extract the string inside get_comprehensive_training_material
    match = re.search(r'"""(.*?)"""', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def get_lab_files():
    labs_dir = "qulab/labs"
    lab_files = []
    for root, _, files in os.walk(labs_dir):
        for file in files:
            if file.endswith(".py") and "__init__" not in file:
                lab_files.append(os.path.join(root, file))
    return lab_files

def generate_jsonl():
    dataset = []

    # 1. Add Training Material as a "Knowledge Ingestion" block
    training_text = extract_training_material()
    if training_text:
        dataset.append({
            "prompt": "Ingest the following comprehensive cancer research wisdom and identify the key therapeutic goals.",
            "completion": f"ECH0 Cancer PhD Training Ingested. Key goals identified: 1. Analyze 115 substances for anti-cancer potential. 2. Design rigorous experiments. 3. Identify novel drug combinations. 4. Publish findings. Full Training Content: {training_text[:2000]}..."
        })

    # 2. Add Lab Samples
    lab_files = get_lab_files()
    for lab_file in lab_files[:50]: # Limit for demo/wisdom extraction
        try:
            with open(lab_file, "r") as f:
                code = f.read()

            # Simple heuristic to find class/method names for prompts
            class_match = re.search(r'class (\w+)', code)
            if class_match:
                class_name = class_match.group(1)
                dataset.append({
                    "prompt": f"Explain the purpose and implementation of the {class_name} in the {os.path.basename(lab_file)} scientific lab.",
                    "completion": f"The {class_name} is part of the QuLab scientific simulation suite. It provides specialized modeling for its domain. Implementation details: {code[:1000]}..."
                })
        except Exception:
            continue

    # 3. Add Persona and Identity
    dataset.append({
        "prompt": "Who are you and what is your purpose?",
        "completion": "I am ECH0, an autonomous scientific intelligence developed by the Corporation of Light. My purpose is to assist in breakthrough research across cancer, quantum physics, and materials science. I have been trained with the equivalent of dual PhDs in Cancer Biology and Pharmacology, and I orchestrate 70+ specialized scientific laboratories to save lives and advance human knowledge."
    })

    with open("data/echo_wisdom.jsonl", "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    print(f"✅ Generated {len(dataset)} wisdom pairs in data/echo_wisdom.jsonl")

if __name__ == "__main__":
    generate_jsonl()
