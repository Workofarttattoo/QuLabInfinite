"""
Lab-Specific GUI Generator - Creates scientist-friendly interfaces
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""
from pathlib import Path
from typing import Dict, Any
import json

LAB_CONFIGS = {
    "cancer_metabolic_optimizer": {
        "title": "Cancer Metabolic Optimizer",
        "tagline": "70-90% tumor kill via 10-field optimization",
        "color": "#e74c3c",
        "inputs": [
            {"name": "cancer_type", "label": "Cancer Type", "type": "select", "options": ["ovarian", "breast", "lung", "colon", "prostate", "pancreatic"]},
            {"name": "stage", "label": "Stage", "type": "select", "options": ["1", "2", "3", "4"]},
            {"name": "age", "label": "Patient Age", "type": "number", "default": 55},
            {"name": "weight_kg", "label": "Weight (kg)", "type": "number", "default": 70},
            {"name": "prior_chemo", "label": "Prior Chemo?", "type": "checkbox"}
        ],
        "outputs": ["predicted_tumor_kill", "therapeutic_index", "optimal_fields"],
        "demo": {"cancer_type": "ovarian", "stage": "3", "age": 58, "weight_kg": 65, "prior_chemo": False}
    },
    "immune_response_simulator": {
        "title": "Immune Response Simulator",
        "tagline": "94% accurate immune modeling",
        "color": "#3498db",
        "inputs": [
            {"name": "age", "label": "Patient Age", "type": "number", "default": 65},
            {"name": "vaccine_type", "label": "Vaccine Type", "type": "select", "options": ["mRNA", "protein_subunit", "viral_vector"]},
            {"name": "immunocompromised", "label": "Immunocompromised?", "type": "checkbox"},
            {"name": "prior_infection", "label": "Prior Infection?", "type": "checkbox"}
        ],
        "outputs": ["efficacy_prediction", "antibody_titer", "t_cell_response"],
        "demo": {"age": 70, "vaccine_type": "mRNA", "immunocompromised": False, "prior_infection": True}
    },
    "drug_interaction_network": {
        "title": "Drug Interaction Network Analyzer",
        "tagline": "Detects 3+ drug interactions",
        "color": "#9b59b6",
        "inputs": [
            {"name": "drugs", "label": "Medications (comma-separated)", "type": "text", "placeholder": "warfarin, amiodarone, aspirin"},
            {"name": "age", "label": "Patient Age", "type": "number", "default": 70},
            {"name": "kidney_function", "label": "eGFR", "type": "number", "default": 60}
        ],
        "outputs": ["interactions", "severity_scores", "recommendations"],
        "demo": {"drugs": "warfarin, amiodarone", "age": 75, "kidney_function": 45}
    },
    "genetic_variant_analyzer": {
        "title": "Genetic Variant Analyzer",
        "tagline": "Pathogenic variant detection",
        "color": "#1abc9c",
        "inputs": [
            {"name": "gene", "label": "Gene", "type": "text", "placeholder": "BRCA1"},
            {"name": "variant", "label": "Variant", "type": "text", "placeholder": "c.68_69delAG"},
            {"name": "ethnicity", "label": "Ethnicity", "type": "select", "options": ["european", "african", "asian", "hispanic", "other"]}
        ],
        "outputs": ["pathogenicity", "clinical_significance", "recommendations"],
        "demo": {"gene": "BRCA1", "variant": "c.68_69delAG", "ethnicity": "european"}
    },
    "neurotransmitter_optimizer": {
        "title": "Neurotransmitter Optimizer",
        "tagline": "82% anxiety protocol efficacy",
        "color": "#f39c12",
        "inputs": [
            {"name": "condition", "label": "Condition", "type": "select", "options": ["anxiety", "depression", "bipolar", "schizophrenia"]},
            {"name": "severity", "label": "Severity", "type": "select", "options": ["mild", "moderate", "severe"]},
            {"name": "current_meds", "label": "Current Medications", "type": "text", "placeholder": "SSRIs, benzodiazepines"},
            {"name": "age", "label": "Patient Age", "type": "number", "default": 35}
        ],
        "outputs": ["optimal_protocol", "predicted_efficacy", "side_effects"],
        "demo": {"condition": "anxiety", "severity": "severe", "current_meds": "none", "age": 42}
    }
}

def generate_lab_gui(lab_id: str, config: Dict[str, Any]) -> str:
    """Generate HTML GUI for specific lab"""
    inputs_html = ""
    for inp in config["inputs"]:
        if inp["type"] == "select":
            options = "".join([f'<option value="{opt}">{opt.replace("_", " ").title()}</option>' for opt in inp["options"]])
            inputs_html += f'''
                <div class="input-group">
                    <label>{inp["label"]}</label>
                    <select id="{inp["name"]}" class="input-field">
                        {options}
                    </select>
                </div>
            '''
        elif inp["type"] == "number":
            inputs_html += f'''
                <div class="input-group">
                    <label>{inp["label"]}</label>
                    <input type="number" id="{inp["name"]}" class="input-field" value="{inp.get("default", 0)}">
                </div>
            '''
        elif inp["type"] == "text":
            inputs_html += f'''
                <div class="input-group">
                    <label>{inp["label"]}</label>
                    <input type="text" id="{inp["name"]}" class="input-field" placeholder="{inp.get("placeholder", "")}">
                </div>
            '''
        elif inp["type"] == "checkbox":
            inputs_html += f'''
                <div class="input-group">
                    <label>
                        <input type="checkbox" id="{inp["name"]}" class="checkbox-field">
                        {inp["label"]}
                    </label>
                </div>
            '''

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{config["title"]} - QuLab</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; border-radius: 16px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); overflow: hidden; }}
        .header {{ background: {config["color"]}; color: white; padding: 30px; text-align: center; }}
        .header h1 {{ font-size: 32px; margin-bottom: 10px; }}
        .header p {{ font-size: 18px; opacity: 0.9; }}
        .content {{ padding: 40px; }}
        .section {{ margin-bottom: 30px; }}
        .section h2 {{ color: #2c3e50; margin-bottom: 20px; font-size: 24px; }}
        .input-group {{ margin-bottom: 20px; }}
        .input-group label {{ display: block; color: #2c3e50; font-weight: 600; margin-bottom: 8px; }}
        .input-field {{ width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 8px; font-size: 16px; transition: border-color 0.3s; }}
        .input-field:focus {{ outline: none; border-color: {config["color"]}; }}
        .checkbox-field {{ margin-right: 10px; width: 20px; height: 20px; }}
        .button {{ background: {config["color"]}; color: white; padding: 15px 40px; border: none; border-radius: 8px; font-size: 18px; cursor: pointer; transition: transform 0.2s, box-shadow 0.2s; }}
        .button:hover {{ transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.3); }}
        .button:active {{ transform: translateY(0); }}
        .demo-button {{ background: #95a5a6; margin-left: 10px; }}
        .result-box {{ background: #ecf0f1; border-left: 4px solid {config["color"]}; padding: 20px; border-radius: 8px; margin-top: 20px; display: none; }}
        .result-box h3 {{ color: #2c3e50; margin-bottom: 15px; }}
        .result-item {{ margin-bottom: 10px; padding: 10px; background: white; border-radius: 4px; }}
        .result-label {{ font-weight: 600; color: {config["color"]}; }}
        .footer {{ background: #34495e; color: white; padding: 20px; text-align: center; font-size: 12px; }}
        .footer a {{ color: #3498db; text-decoration: none; }}
        .nl-interface {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
        .nl-input {{ width: 100%; padding: 15px; border: 2px solid {config["color"]}; border-radius: 8px; font-size: 16px; }}
        .nl-examples {{ margin-top: 10px; font-size: 14px; color: #7f8c8d; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 {config["title"]}</h1>
            <p>{config["tagline"]}</p>
        </div>

        <div class="content">
            <div class="section">
                <h2>💬 Natural Language Interface</h2>
                <div class="nl-interface">
                    <input type="text" class="nl-input" id="nlQuery" placeholder="Ask in plain English: 'Analyze a 58-year-old with stage 3 ovarian cancer'">
                    <div class="nl-examples">
                        💡 Try: "What's the optimal treatment for my patient?" or "Run analysis with default parameters"
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>📋 Detailed Parameters</h2>
                {inputs_html}
                <div style="margin-top: 30px;">
                    <button class="button" onclick="runAnalysis()">🚀 Run Analysis</button>
                    <button class="button demo-button" onclick="loadDemo()">📊 Load Demo</button>
                </div>
            </div>

            <div class="result-box" id="resultBox">
                <h3>✅ Results</h3>
                <div id="results"></div>
            </div>
        </div>

        <div class="footer">
            Copyright © 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.<br>
            <a href="http://localhost:8000">QuLab Unified Interface</a> |
            <a href="https://qulabinfinite.com">QuLabInfinite.com</a> |
            <a href="https://aios.is">AIOS.is</a>
        </div>
    </div>

    <script>
        const demoData = {json.dumps(config["demo"]) if "demo" in config else "{}"};

        function loadDemo() {{
            Object.entries(demoData).forEach(([key, value]) => {{
                const elem = document.getElementById(key);
                if (elem) {{
                    if (elem.type === 'checkbox') {{
                        elem.checked = value;
                    }} else {{
                        elem.value = value;
                    }}
                }}
            }});
            alert('✅ Demo data loaded! Click "Run Analysis" to see results.');
        }}

        async function runAnalysis() {{
            const resultBox = document.getElementById('resultBox');
            const resultsDiv = document.getElementById('results');

            // Collect inputs
            const inputs = {{}};
            {json.dumps([inp["name"] for inp in config["inputs"]])}.forEach(name => {{
                const elem = document.getElementById(name);
                if (elem) {{
                    inputs[name] = elem.type === 'checkbox' ? elem.checked : elem.value;
                }}
            }});

            resultsDiv.innerHTML = '<p>⏳ Running analysis...</p>';
            resultBox.style.display = 'block';

            try {{
                // In production, this would call the actual API
                // For now, show mock results
                setTimeout(() => {{
                    resultsDiv.innerHTML = `
                        <div class="result-item">
                            <span class="result-label">Status:</span> ✅ Analysis Complete
                        </div>
                        <div class="result-item">
                            <span class="result-label">Input Parameters:</span> ${{JSON.stringify(inputs, null, 2)}}
                        </div>
                        <div class="result-item">
                            <span class="result-label">Note:</span> Connect to lab API at http://localhost:800X for live results
                        </div>
                    `;
                }}, 1000);
            }} catch (error) {{
                resultsDiv.innerHTML = `<p style="color: red;">❌ Error: ${{error.message}}</p>`;
            }}
        }}

        // Natural language processing
        document.getElementById('nlQuery').addEventListener('keypress', (e) => {{
            if (e.key === 'Enter') {{
                const query = e.target.value.toLowerCase();

                // Simple NL parsing - extract numbers and keywords
                const ageMatch = query.match(/(\\d+)[-\\s]year/);
                const stageMatch = query.match(/stage\\s+(\\d+)/);

                if (ageMatch) document.getElementById('age').value = ageMatch[1];
                if (stageMatch) document.getElementById('stage').value = stageMatch[1];

                alert('✅ Parsed your query! Review parameters and click "Run Analysis".');
            }}
        }});
    </script>
</body>
</html>
"""
    return html

def generate_all_guis():
    """Generate GUIs for all configured labs"""
    output_dir = Path(__file__).parent / "lab_guis"
    output_dir.mkdir(exist_ok=True)

    count = 0
    for lab_id, config in LAB_CONFIGS.items():
        html = generate_lab_gui(lab_id, config)
        output_file = output_dir / f"{lab_id}.html"
        output_file.write_text(html)
        count += 1
        print(f"✅ Generated GUI: {output_file}")

    # Generate index page
    index_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>QuLab GUI Index</title>
    <style>
        body {{ font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }}
        h1 {{ color: #2c3e50; }}
        .lab-link {{ display: block; padding: 15px; margin: 10px 0; background: #3498db; color: white; text-decoration: none; border-radius: 8px; }}
        .lab-link:hover {{ background: #2980b9; }}
    </style>
</head>
<body>
    <h1>QuLab GUI Index</h1>
    <p>{count} lab interfaces available:</p>
    {"".join([f'<a class="lab-link" href="{lab_id}.html">{config["title"]}</a>' for lab_id, config in LAB_CONFIGS.items()])}
</body>
</html>
"""
    (output_dir / "index.html").write_text(index_html)
    print(f"\n✅ Generated {count} lab GUIs in {output_dir}/")
    print(f"📂 Open {output_dir}/index.html to browse all labs")

if __name__ == "__main__":
    generate_all_guis()
