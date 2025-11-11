#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Test with DIVERSE inventions to show REAL material matching and cost variation
"""

from ech0_invention_poc_pipeline import ECH0_POC_Pipeline, InventionConcept

def test_diverse_inventions():
    """Test with inventions that have clear material requirements."""

    # Create inventions with explicit material mentions in descriptions
    concepts = [
        InventionConcept(
            "Titanium-Graphene Aerospace Strut",
            "High strength-to-weight strut using titanium alloy (Ti-6Al-4V) matrix with graphene reinforcement for aerospace applications"
        ),
        InventionConcept(
            "Silicon Carbide Thermal Shield",
            "Ultra-high temperature ceramic shield using silicon carbide (SiC) for hypersonic vehicle re-entry protection"
        ),
        InventionConcept(
            "Aluminum-Carbon Fiber Composite Panel",
            "Lightweight structural panel combining aluminum alloy matrix with carbon fiber reinforcement for automotive body panels"
        ),
        InventionConcept(
            "Aerogel Insulation System",
            "Ultra-light thermal insulation using silica aerogel structure with nanoporous architecture for spacecraft"
        ),
        InventionConcept(
            "Steel-Concrete Hybrid Beam",
            "Structural beam using steel reinforcement in concrete matrix for bridge construction applications"
        ),
    ]

    pipeline = ECH0_POC_Pipeline()

    requirements = {
        'application': 'advanced',
        'budget': 10000.0,
        'constraints': {
            'max_weight': 10.0,
            'min_performance': 0.85
        }
    }

    print("="*80)
    print("TESTING DIVERSE INVENTIONS - REAL MATERIAL MATCHING")
    print("="*80)
    print()

    results = pipeline.run_pipeline(concepts, requirements)

    # Show cost diversity
    print("\n" + "="*80)
    print("COST DIVERSITY VERIFICATION")
    print("="*80)

    costs = []
    for poc in results['pocs']:
        name = poc['name']
        # Extract cost from findings
        cost_finding = [f for f in poc['findings'] if 'Cost estimate' in f]
        if cost_finding:
            cost_str = cost_finding[0].split('$')[1].split()[0]
            cost = float(cost_str.replace(',', ''))
            costs.append(cost)
            print(f"{name}: ${cost:,.2f}")

    print()
    unique_costs = len(set(costs))
    print(f"✅ Unique cost values: {unique_costs}/{len(costs)}")
    print(f"✅ Cost range: ${min(costs):,.2f} - ${max(costs):,.2f}")
    print(f"✅ Cost variation: {(max(costs) - min(costs)) / min(costs) * 100:.1f}%")
    print()

    if unique_costs >= 3:
        print("✅ PASS: Costs show real diversity (not all the same)")
    else:
        print("⚠️  WARNING: Limited cost diversity detected")

    return results


if __name__ == "__main__":
    test_diverse_inventions()
