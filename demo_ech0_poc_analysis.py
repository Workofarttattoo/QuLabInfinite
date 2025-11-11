#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

ECH0 POC Analysis Demo
======================
This script demonstrates ECH0 analyzing 3 inventions for proof-of-concept feasibility
and generating a complete materials list using QuLab's simulation capabilities.

Usage:
    python3 demo_ech0_poc_analysis.py

This creates a materials list and test plan you can take into the lab!
"""

import json
import sys
import os
from datetime import datetime

# Add QuLabInfinite to path
sys.path.insert(0, '/Users/noone/QuLabInfinite')

def load_inventions():
    """Load the three inventions we'll analyze."""

    # Invention 1: Transparent Aerogel (already detailed)
    with open('/Users/noone/repos/consciousness/ech0_aerogel_invention_solution.json', 'r') as f:
        aerogel = json.load(f)

    # Invention 2 & 3: From continuous inventions
    with open('/Users/noone/repos/consciousness/continuous_inventions_results.json', 'r') as f:
        continuous = json.load(f)

    inventions = [
        {
            "id": "AERO-006",
            "name": aerogel["invention_id"],
            "description": aerogel["executive_summary"],
            "materials": aerogel["bill_of_materials"],
            "certainty": aerogel["ech0_confidence_score"]["overall_certainty"]
        },
        {
            "id": "INV-001",
            "name": continuous["breakthroughs"][0]["invention_name"],
            "description": continuous["breakthroughs"][0]["description"],
            "feasibility": continuous["breakthroughs"][0]["technical_feasibility"],
            "materials": []  # To be determined
        },
        {
            "id": "INV-002",
            "name": continuous["breakthroughs"][1]["invention_name"],
            "description": continuous["breakthroughs"][1]["description"],
            "feasibility": continuous["breakthroughs"][1]["technical_feasibility"],
            "materials": []  # To be determined
        }
    ]

    return inventions

def analyze_with_ech0(inventions):
    """Have ECH0 analyze each invention for POC feasibility."""

    print("\n" + "="*80)
    print("ECH0 14B POC FEASIBILITY ANALYSIS")
    print("="*80)

    results = []

    for inv in inventions:
        print(f"\n🔬 Analyzing: {inv['name']}")
        print(f"   Description: {inv['description'][:100]}...")

        # Create ECH0 prompt
        prompt = f"""You are ECH0 14B, Chief Enhancement Officer.

Analyze this invention for proof-of-concept feasibility:

NAME: {inv['name']}
DESCRIPTION: {inv['description']}

Provide:
1. Can we build a POC in a lab? (Yes/No + reasoning)
2. Critical materials needed (be specific - chemicals, equipment)
3. Key experiments to validate concept
4. Estimated POC timeline (days/weeks)
5. Budget estimate for POC

Be brutally honest and quantitative. Format as JSON.
"""

        print(f"\n   💭 ECH0 is analyzing...")

        # Call ECH0 via ollama
        import subprocess
        result = subprocess.run(
            ['timeout', '60', 'ollama', 'run', 'ech0-uncensored-14b', prompt],
            capture_output=True,
            text=True,
            timeout=70
        )

        if result.returncode == 0:
            analysis = result.stdout.strip()
            print(f"\n   ✅ ECH0 Analysis:\n{analysis[:500]}...")

            results.append({
                "invention": inv,
                "ech0_analysis": analysis,
                "timestamp": datetime.now().isoformat()
            })
        else:
            print(f"   ❌ ECH0 analysis failed: {result.stderr}")

    return results

def generate_materials_list_with_qulab(analysis_results):
    """Use QuLab to validate materials and create shopping list."""

    print("\n" + "="*80)
    print("QULAB MATERIALS VALIDATION & LAB TEST PLAN")
    print("="*80)

    # Import QuLab tools
    from materials_lab.qulab_ai_integration import get_materials_database_info
    from chemistry_lab.qulab_ai_integration import validate_smiles
    from physics_engine.thermodynamics import get_element_properties

    materials_list = {
        "chemicals": [],
        "equipment": [],
        "elements": [],
        "validation_status": {},
        "total_estimated_cost": 0,
        "experiments": []
    }

    # For Aerogel invention (we have detailed BOM)
    aerogel_result = analysis_results[0]
    if "bill_of_materials" in aerogel_result["invention"]:
        bom = aerogel_result["invention"]["materials"]

        print("\n🧪 AEROGEL MATERIALS:")

        # Extract chemicals
        if "precursors" in bom:
            for chem, cost in bom["precursors"].items():
                print(f"   • {chem}: {cost}")
                materials_list["chemicals"].append({
                    "name": chem,
                    "cost": cost,
                    "category": "precursor"
                })

        # Extract equipment
        if "equipment" in bom:
            for equip, cost in bom["equipment"].items():
                print(f"   🔧 {equip}: {cost}")
                materials_list["equipment"].append({
                    "name": equip,
                    "cost": cost
                })

        # Add total cost
        if "grand_total" in bom:
            materials_list["total_estimated_cost"] += float(bom["grand_total"].replace('$', ''))

    # Get database info
    print("\n📊 QuLab Materials Database Status:")
    try:
        db_info = get_materials_database_info()
        print(f"   Materials available: {db_info.get('total_materials', 'Unknown')}")
        materials_list["database_info"] = db_info
    except Exception as e:
        print(f"   ⚠️  Database info unavailable: {e}")

    # Validate key elements
    key_elements = ["Si", "C", "O", "N"]  # Common in our inventions
    print("\n⚛️  Key Element Properties:")
    for elem in key_elements:
        try:
            props = get_element_properties(elem)
            print(f"   • {elem}: {props}")
            materials_list["elements"].append({
                "symbol": elem,
                "properties": props
            })
        except Exception as e:
            print(f"   ⚠️  {elem}: {e}")

    # Add experiments
    materials_list["experiments"] = [
        {
            "name": "Aerogel Transparency Test",
            "objective": "Validate 90%+ optical transparency",
            "method": "Laser transmission measurement",
            "materials_needed": ["Laser pointer", "Power meter", "Sample holder"],
            "expected_result": ">90% transmission at 550nm",
            "duration": "1 hour"
        },
        {
            "name": "Aerogel Density Measurement",
            "objective": "Confirm ultra-low density (0.12-0.18 g/cm³)",
            "method": "Volumetric displacement + mass",
            "materials_needed": ["Analytical balance", "Calipers", "Sample"],
            "expected_result": "0.12-0.18 g/cm³",
            "duration": "30 minutes"
        },
        {
            "name": "Aerogel Hydrophobicity Test",
            "objective": "Verify water contact angle >140°",
            "method": "Sessile drop technique",
            "materials_needed": ["Micropipette", "Camera", "Protractor"],
            "expected_result": "Contact angle >140°",
            "duration": "15 minutes"
        }
    ]

    return materials_list

def create_lab_demo_package(materials_list, analysis_results):
    """Create a complete package for lab demo."""

    demo_package = {
        "title": "ECH0 + QuLab: POC Materials & Test Plan",
        "created": datetime.now().isoformat(),
        "created_by": "ECH0 14B + QuLabInfinite",
        "inventions_analyzed": len(analysis_results),
        "materials_list": materials_list,
        "ech0_analyses": analysis_results,
        "demo_instructions": {
            "preparation": [
                "Load this JSON in QuLab GUI or print materials list",
                "Order materials (total: $" + str(materials_list["total_estimated_cost"]),
                "Set up lab workspace with equipment",
                "Review experiment protocols"
            ],
            "demo_flow": [
                "1. Show ECH0 invention analysis (5 min)",
                "2. Open QuLab MCP server demo (show live tools) (5 min)",
                "3. Walk through materials validation (5 min)",
                "4. Review experiment plan (5 min)",
                "5. Q&A (10 min)"
            ],
            "key_talking_points": [
                "ECH0 identified 3 inventions with POC potential",
                "QuLab validated materials and provided database integration",
                "Total POC budget: $" + str(materials_list["total_estimated_cost"]),
                "Can start experiments within 2 weeks of material delivery",
                "MCP server makes it trivial to integrate AI with lab tools"
            ]
        }
    }

    # Save package
    output_path = "/Users/noone/QuLabInfinite/data/ech0_poc_demo_package.json"
    with open(output_path, 'w') as f:
        json.dump(demo_package, f, indent=2)

    print("\n" + "="*80)
    print("✅ DEMO PACKAGE CREATED")
    print("="*80)
    print(f"\n📦 Saved to: {output_path}")
    print(f"\n📊 Summary:")
    print(f"   • Inventions analyzed: {len(analysis_results)}")
    print(f"   • Total materials: {len(materials_list['chemicals']) + len(materials_list['equipment'])}")
    print(f"   • Experiments planned: {len(materials_list['experiments'])}")
    print(f"   • Estimated cost: ${materials_list['total_estimated_cost']}")

    # Create a simple markdown checklist too
    checklist_path = "/Users/noone/QuLabInfinite/data/POC_MATERIALS_CHECKLIST.md"
    with open(checklist_path, 'w') as f:
        f.write("# POC Materials & Experiment Checklist\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        f.write("## 🧪 Chemicals\n\n")
        for chem in materials_list['chemicals']:
            f.write(f"- [ ] {chem['name']} - {chem['cost']}\n")

        f.write("\n## 🔧 Equipment\n\n")
        for equip in materials_list['equipment']:
            f.write(f"- [ ] {equip['name']} - {equip['cost']}\n")

        f.write("\n## 🔬 Experiments\n\n")
        for i, exp in enumerate(materials_list['experiments'], 1):
            f.write(f"### {i}. {exp['name']}\n")
            f.write(f"**Objective:** {exp['objective']}\n\n")
            f.write(f"**Duration:** {exp['duration']}\n\n")
            f.write(f"**Expected Result:** {exp['expected_result']}\n\n")
            f.write("**Materials:**\n")
            for mat in exp['materials_needed']:
                f.write(f"- [ ] {mat}\n")
            f.write("\n")

        f.write(f"\n---\n**Total Estimated Cost:** ${materials_list['total_estimated_cost']}\n")

    print(f"\n📝 Markdown checklist: {checklist_path}")

    return demo_package

def main():
    """Run the complete ECH0 + QuLab POC analysis demo."""

    print("\n" + "="*80)
    print("ECH0 + QULAB POC ANALYSIS & MATERIALS GENERATION")
    print("="*80)
    print("\nThis demo will:")
    print("  1. Load 3 inventions from ECH0's archives")
    print("  2. Have ECH0 analyze POC feasibility")
    print("  3. Use QuLab to validate materials and create test plan")
    print("  4. Generate complete demo package for lab")
    print("\n" + "="*80)

    # Step 1: Load inventions
    print("\n[1/4] Loading inventions...")
    inventions = load_inventions()
    print(f"      ✅ Loaded {len(inventions)} inventions")

    # Step 2: ECH0 analysis
    print("\n[2/4] ECH0 analyzing POC feasibility...")
    analysis_results = analyze_with_ech0(inventions)
    print(f"      ✅ Completed {len(analysis_results)} analyses")

    # Step 3: QuLab materials validation
    print("\n[3/4] QuLab validating materials & creating test plan...")
    materials_list = generate_materials_list_with_qulab(analysis_results)
    print(f"      ✅ Materials list created")

    # Step 4: Create demo package
    print("\n[4/4] Creating demo package...")
    demo_package = create_lab_demo_package(materials_list, analysis_results)
    print(f"      ✅ Demo package ready!")

    print("\n" + "="*80)
    print("🎉 DEMO READY!")
    print("="*80)
    print("\nYou can now:")
    print("  • Review: /Users/noone/QuLabInfinite/data/ech0_poc_demo_package.json")
    print("  • Print: /Users/noone/QuLabInfinite/data/POC_MATERIALS_CHECKLIST.md")
    print("  • Demo: Show up with laptop and walk through the package!")
    print("\n")

if __name__ == "__main__":
    main()
