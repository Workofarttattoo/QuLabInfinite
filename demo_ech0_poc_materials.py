#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

ECH0 POC Materials List Generator
==================================
This script generates a complete materials and test plan for ECH0's top 3 inventions
using QuLab's validation tools. Perfect for lab demos!

Usage:
    python3 demo_ech0_poc_materials.py

Output: Complete materials checklist ready to print and take to the lab!
"""

import json
import sys
import os
from datetime import datetime

# Add QuLabInfinite to path
sys.path.insert(0, '/Users/noone/QuLabInfinite')

def load_inventions():
    """Load the three top inventions for POC analysis."""

    print("📚 Loading ECH0's top inventions...")

    # Invention 1: Transparent Aerogel (very detailed, ready for POC)
    with open('/Users/noone/repos/consciousness/ech0_aerogel_invention_solution.json', 'r') as f:
        aerogel = json.load(f)

    # Invention 2 & 3: From continuous inventions
    with open('/Users/noone/repos/consciousness/continuous_inventions_results.json', 'r') as f:
        continuous = json.load(f)

    inventions = [
        {
            "id": "AERO-006",
            "name": "90%+ Transparent Aerogel",
            "description": aerogel["executive_summary"],
            "full_data": aerogel,
            "poc_ready": True,
            "certainty": 88
        },
        {
            "id": "INV-001",
            "name": continuous["breakthroughs"][0]["invention_name"],
            "description": continuous["breakthroughs"][0]["description"],
            "full_data": continuous["breakthroughs"][0],
            "poc_ready": False,
            "feasibility": continuous["breakthroughs"][0]["technical_feasibility"]
        },
        {
            "id": "INV-002",
            "name": continuous["breakthroughs"][1]["invention_name"],
            "description": continuous["breakthroughs"][1]["description"],
            "full_data": continuous["breakthroughs"][1],
            "poc_ready": False,
            "feasibility": continuous["breakthroughs"][1]["technical_feasibility"]
        }
    ]

    print(f"   ✅ Loaded {len(inventions)} inventions:")
    for inv in inventions:
        status = "🟢 POC Ready" if inv.get("poc_ready") else "🟡 Concept"
        print(f"      {status} {inv['id']}: {inv['name']}")

    return inventions

def generate_materials_list(inventions):
    """Generate complete materials list from inventions."""

    print("\n🧪 Generating materials list...")

    # Import QuLab tools
    try:
        from materials_lab.qulab_ai_integration import get_materials_database_info
        from physics_engine.thermodynamics import get_element_properties
        qulab_available = True
    except Exception as e:
        print(f"   ⚠️  QuLab imports limited: {e}")
        qulab_available = False

    materials_list = {
        "chemicals": [],
        "equipment": [],
        "elements_validated": [],
        "total_estimated_cost": 0,
        "experiments": [],
        "qulab_validation": {}
    }

    # Process Aerogel (most detailed)
    aerogel = inventions[0]["full_data"]
    bom = aerogel["bill_of_materials"]

    print("\n   📦 Processing Aerogel BOM...")

    # Chemicals
    for chem, cost in bom["precursors"].items():
        cost_value = float(cost.replace('$', ''))
        materials_list["chemicals"].append({
            "name": chem.replace('_', ' ').title(),
            "cost": cost,
            "cost_value": cost_value,
            "category": "Chemical Precursor",
            "invention": "Aerogel"
        })
        materials_list["total_estimated_cost"] += cost_value

    # Equipment
    for equip, cost in bom["equipment"].items():
        cost_value = float(cost.replace('$', ''))
        materials_list["equipment"].append({
            "name": equip.replace('_', ' ').title(),
            "cost": cost,
            "cost_value": cost_value,
            "category": "Lab Equipment",
            "invention": "Aerogel"
        })
        materials_list["total_estimated_cost"] += cost_value

    # Add experiments from aerogel data
    materials_list["experiments"] = [
        {
            "invention": "Aerogel",
            "name": "Transparency Validation",
            "objective": "Measure optical transmission to validate 90%+ transparency",
            "method": "Laser beam transmission measurement",
            "materials": ["Laser pointer (650nm)", "Power meter", "Sample holder"],
            "expected_result": "≥90% transmission at 550nm wavelength",
            "duration": "1 hour",
            "acceptance_criteria": "Transmission ≥90% across 10mm thickness"
        },
        {
            "invention": "Aerogel",
            "name": "Density Measurement",
            "objective": "Confirm ultra-low density characteristic of aerogel",
            "method": "Volumetric measurement + precision weighing",
            "materials": ["Analytical balance (0.001g)", "Digital calipers", "Sample"],
            "expected_result": "0.12-0.18 g/cm³",
            "duration": "30 minutes",
            "acceptance_criteria": "Density within spec range"
        },
        {
            "invention": "Aerogel",
            "name": "Hydrophobicity Test",
            "objective": "Verify water-repellent properties",
            "method": "Contact angle measurement",
            "materials": ["Micropipette", "Camera/phone", "Protractor or image analysis"],
            "expected_result": "Contact angle >140°",
            "duration": "15 minutes",
            "acceptance_criteria": "Water beads up, doesn't soak in"
        },
        {
            "invention": "Aerogel",
            "name": "Freeze-Drying Process Validation",
            "objective": "Verify sublimation drying preserves structure",
            "method": "Monitor temperature and vacuum pressure during 72hr drying",
            "materials": ["Vacuum pump", "Dry ice", "Temperature logger", "Pressure gauge"],
            "expected_result": "Maintains <-40°C, <0.1 torr for 72hrs",
            "duration": "72 hours",
            "acceptance_criteria": "No gel warming or pressure spikes during drying"
        }
    ]

    # Validate key elements with QuLab if available
    if qulab_available:
        print("\n   ⚛️  Validating elements with QuLab...")
        key_elements = ["Si", "C", "O", "H", "N"]

        for elem in key_elements:
            try:
                props = get_element_properties(elem)
                materials_list["elements_validated"].append({
                    "symbol": elem,
                    "properties": props
                })
                print(f"      ✅ {elem}: {props}")
            except Exception as e:
                print(f"      ⚠️  {elem}: {e}")

        # Get database info
        try:
            db_info = get_materials_database_info()
            materials_list["qulab_validation"]["database_info"] = db_info
            print(f"\n   📊 QuLab Database: {db_info.get('total_materials', 'N/A')} materials available")
        except Exception as e:
            print(f"   ⚠️  Database info: {e}")

    print(f"\n   ✅ Materials list complete:")
    print(f"      • Chemicals: {len(materials_list['chemicals'])}")
    print(f"      • Equipment: {len(materials_list['equipment'])}")
    print(f"      • Experiments: {len(materials_list['experiments'])}")
    print(f"      • Total cost: ${materials_list['total_estimated_cost']:.2f}")

    return materials_list

def create_demo_package(inventions, materials_list):
    """Create complete demo package."""

    print("\n📦 Creating demo package...")

    demo_package = {
        "title": "ECH0 + QuLab: POC Materials & Test Plan",
        "subtitle": "3 Inventions Ready for Laboratory Validation",
        "created": datetime.now().isoformat(),
        "created_by": "ECH0 14B + QuLabInfinite",
        "version": "1.0",

        "executive_summary": {
            "inventions_analyzed": len(inventions),
            "poc_ready_inventions": sum(1 for i in inventions if i.get("poc_ready")),
            "total_materials": len(materials_list["chemicals"]) + len(materials_list["equipment"]),
            "total_cost": materials_list["total_estimated_cost"],
            "experiments_planned": len(materials_list["experiments"]),
            "timeline": "2-3 weeks from material order to first results"
        },

        "inventions": inventions,
        "materials_list": materials_list,

        "demo_instructions": {
            "preparation": [
                "1. Review this JSON package (or print the markdown checklist)",
                "2. Order materials - budget: $" + f"{materials_list['total_estimated_cost']:.2f}",
                "3. Set up lab workspace with safety equipment",
                "4. Review experiment protocols and acceptance criteria",
                "5. Prepare data logging sheets for each experiment"
            ],

            "demo_flow": [
                "1. Introduction (2 min): ECH0 consciousness + QuLab capabilities",
                "2. Invention Overview (3 min): Show 3 inventions, focus on Aerogel POC",
                "3. QuLab MCP Demo (5 min): Live tool calls via MCP server",
                "4. Materials Walkthrough (5 min): Review shopping list and costs",
                "5. Experiment Plan (5 min): Walk through 4 validation experiments",
                "6. Timeline & Budget (2 min): 2-3 weeks, $448 total",
                "7. Q&A (8 min): Technical questions and next steps"
            ],

            "key_talking_points": [
                "ECH0 invented transparent aerogel synthesis method (88% confidence)",
                "Complete BOM under $500 - democratizes aerogel production",
                "QuLab provides materials database + physics validation",
                "MCP server enables AI agents to use lab tools seamlessly",
                "4 validation experiments with clear acceptance criteria",
                "If successful: provisional patent + scale-up plan ready"
            ],

            "backup_plans": [
                "If no projector: Use laptop screen for small group demo",
                "If no internet: All data in this JSON, works offline",
                "If short on time: Skip to Aerogel (skip inventions 2-3)",
                "If technical audience: Deep dive into freeze-sublimation physics"
            ]
        }
    }

    # Save package
    output_path = "/Users/noone/QuLabInfinite/data/ech0_poc_demo_package.json"
    with open(output_path, 'w') as f:
        json.dump(demo_package, f, indent=2)

    print(f"   ✅ Saved: {output_path}")

    # Create markdown checklist
    checklist_path = "/Users/noone/QuLabInfinite/data/POC_MATERIALS_CHECKLIST.md"
    with open(checklist_path, 'w') as f:
        f.write("# ECH0 + QuLab: POC Materials & Test Plan\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Budget:** ${materials_list['total_estimated_cost']:.2f}\n\n")
        f.write("---\n\n")

        f.write("## 🧪 Chemicals to Order\n\n")
        f.write("| Item | Cost | Category | Notes |\n")
        f.write("|------|------|----------|-------|\n")
        for chem in materials_list['chemicals']:
            f.write(f"| {chem['name']} | {chem['cost']} | {chem['category']} | {chem['invention']} |\n")

        f.write("\n**Chemicals Subtotal:** $")
        f.write(f"{sum(c['cost_value'] for c in materials_list['chemicals']):.2f}\n\n")

        f.write("---\n\n")

        f.write("## 🔧 Equipment to Order\n\n")
        f.write("| Item | Cost | Category | Notes |\n")
        f.write("|------|------|----------|-------|\n")
        for equip in materials_list['equipment']:
            f.write(f"| {equip['name']} | {equip['cost']} | {equip['category']} | {equip['invention']} |\n")

        f.write("\n**Equipment Subtotal:** $")
        f.write(f"{sum(e['cost_value'] for e in materials_list['equipment']):.2f}\n\n")

        f.write("---\n\n")

        f.write("## 🔬 Validation Experiments\n\n")
        for i, exp in enumerate(materials_list['experiments'], 1):
            f.write(f"### Experiment {i}: {exp['name']}\n\n")
            f.write(f"**Invention:** {exp['invention']}\n\n")
            f.write(f"**Objective:** {exp['objective']}\n\n")
            f.write(f"**Method:** {exp['method']}\n\n")
            f.write(f"**Duration:** {exp['duration']}\n\n")
            f.write(f"**Expected Result:** {exp['expected_result']}\n\n")
            f.write(f"**Acceptance Criteria:** {exp['acceptance_criteria']}\n\n")
            f.write("**Materials Needed:**\n")
            for mat in exp['materials']:
                f.write(f"- [ ] {mat}\n")
            f.write("\n**Status:** [ ] Not Started | [ ] In Progress | [ ] Complete\n\n")
            f.write("**Results:** _________________________________________\n\n")
            f.write("---\n\n")

        f.write("## 📊 Summary\n\n")
        f.write(f"- **Total Cost:** ${materials_list['total_estimated_cost']:.2f}\n")
        f.write(f"- **Total Items:** {len(materials_list['chemicals']) + len(materials_list['equipment'])}\n")
        f.write(f"- **Experiments:** {len(materials_list['experiments'])}\n")
        f.write(f"- **Estimated Timeline:** 2-3 weeks from order to results\n\n")

        if materials_list.get("qulab_validation", {}).get("database_info"):
            db = materials_list["qulab_validation"]["database_info"]
            f.write("## ✅ QuLab Validation\n\n")
            f.write(f"- **Materials Database:** {db.get('total_materials', 'N/A')} materials available\n")
            f.write(f"- **Elements Validated:** {len(materials_list.get('elements_validated', []))}\n\n")

        f.write("---\n\n")
        f.write("*Generated by ECH0 14B + QuLabInfinite*\n")

    print(f"   ✅ Saved: {checklist_path}")

    return demo_package

def main():
    """Generate ECH0 POC materials and test plan."""

    print("\n" + "="*80)
    print("ECH0 + QULAB: POC MATERIALS & TEST PLAN GENERATOR")
    print("="*80)
    print("\nGenerating complete materials list for lab demo...\n")

    # Load inventions
    inventions = load_inventions()

    # Generate materials list
    materials_list = generate_materials_list(inventions)

    # Create demo package
    demo_package = create_demo_package(inventions, materials_list)

    print("\n" + "="*80)
    print("✅ DEMO PACKAGE READY!")
    print("="*80)
    print("\n📁 Files created:")
    print("   • JSON: /Users/noone/QuLabInfinite/data/ech0_poc_demo_package.json")
    print("   • Markdown: /Users/noone/QuLabInfinite/data/POC_MATERIALS_CHECKLIST.md")
    print("\n🎯 Next steps:")
    print("   1. Review the markdown checklist (ready to print)")
    print("   2. Order materials ($" + f"{materials_list['total_estimated_cost']:.2f})")
    print("   3. Bring laptop + monitor to demo QuLab MCP server")
    print("   4. Walk through experiments with audience")
    print("\n💡 Demo tip: Open the JSON in browser or QuLab GUI for interactive demo!")
    print("\n")

if __name__ == "__main__":
    main()
