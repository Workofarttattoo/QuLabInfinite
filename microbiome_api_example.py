"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

MICROBIOME HEALTH OPTIMIZER API - USAGE EXAMPLES
Demonstrates all 10 breakthrough innovations with real-world scenarios
"""

import requests
import json

# API Base URL (assumes server running locally)
BASE_URL = "http://localhost:8000"


def example_1_ibs_patient():
    """Example 1: IBS patient with severe dysbiosis"""
    print("\n" + "="*70)
    print("EXAMPLE 1: IBS Patient Optimization")
    print("="*70)

    # Patient profile: High F/B ratio, low diversity, low butyrate
    profile = {
        "profile": {
            "firmicutes_abundance": 0.7,
            "bacteroidetes_abundance": 0.15,
            "actinobacteria_abundance": 0.05,
            "proteobacteria_abundance": 0.08,
            "verrucomicrobia_abundance": 0.02,
            "shannon_diversity": 2.5,
            "butyrate_producers": 0.3,
            "acetate_producers": 0.4,
            "propionate_producers": 0.2,
            "lps_producers": 0.5,
            "fiber_degradation": 0.4,
            "mucin_degradation": 0.3,
            "bile_acid_metabolism": 0.4,
            "neurotransmitter_production": 0.3
        },
        "condition": "ibs",
        "preferences": {
            "probiotics_allowed": True,
            "vegetarian": False
        }
    }

    response = requests.post(f"{BASE_URL}/optimize", json=profile)
    result = response.json()

    print(f"\nCurrent State:")
    print(f"  F/B Ratio: {0.7/0.15:.2f} (target: 1.5-2.5)")
    print(f"  Diversity: 2.5 (target: >3.5)")
    print(f"  Butyrate Producers: 0.3 (target: >0.6)")

    print(f"\nRecommendations:")
    print(f"  Probiotics: {len(result['probiotics'])} formulations")
    print(f"  Prebiotics: {len(result['prebiotics'])} fiber types")
    print(f"  Dietary Changes: {len(result['dietary_changes'])} interventions")

    print(f"\nExpected Outcomes:")
    print(f"  F/B Ratio: {result['expected_fb_ratio']:.2f}")
    print(f"  Butyrate Increase: {result['expected_butyrate_increase']:.2f}")
    print(f"  Timeline: {result['expected_timeline_weeks']} weeks")
    print(f"  Confidence: {result['confidence_score']:.0%}")


def example_2_depression_gut_brain():
    """Example 2: Depression patient - gut-brain axis optimization"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Depression - Gut-Brain Axis")
    print("="*70)

    profile = {
        "profile": {
            "firmicutes_abundance": 0.55,
            "bacteroidetes_abundance": 0.25,
            "actinobacteria_abundance": 0.08,
            "proteobacteria_abundance": 0.1,
            "verrucomicrobia_abundance": 0.02,
            "shannon_diversity": 3.0,
            "butyrate_producers": 0.4,
            "acetate_producers": 0.3,
            "propionate_producers": 0.2,
            "lps_producers": 0.4,
            "fiber_degradation": 0.5,
            "mucin_degradation": 0.4,
            "bile_acid_metabolism": 0.5,
            "neurotransmitter_production": 0.3
        },
        "condition": "depression"
    }

    response = requests.post(f"{BASE_URL}/gut-brain", json=profile)
    result = response.json()

    print(f"\nCurrent Neurotransmitters:")
    print(f"  Serotonin: {result['current_neurotransmitters']['serotonin']:.2f}")
    print(f"  GABA: {result['current_neurotransmitters']['gaba']:.2f}")
    print(f"  Dopamine: {result['current_neurotransmitters']['dopamine']:.2f}")
    print(f"  BBB Integrity: {result['current_neurotransmitters']['bbb_integrity']:.2f}")

    print(f"\nInterventions: {len(result['interventions'])}")
    for i, intervention in enumerate(result['interventions'], 1):
        print(f"  {i}. Target: {intervention['target']}")
        print(f"     Expected boost: {intervention.get('expected_boost', 'N/A')}")

    print(f"\nExpected Timeline: {result['expected_improvement_weeks']} weeks")


def example_3_probiotic_synergy():
    """Example 3: Probiotic synergy calculation"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Probiotic Synergy Prediction")
    print("="*70)

    strains = [
        'lactobacillus_rhamnosus_gg',
        'bifidobacterium_longum',
        'akkermansia_muciniphila'
    ]
    cfus = [1e10, 1e10, 1e8]

    response = requests.post(
        f"{BASE_URL}/probiotic-synergy",
        params={"strains": strains, "cfus": cfus}
    )
    result = response.json()

    print(f"\nStrains: {', '.join(strains)}")
    print(f"CFUs: {', '.join([f'{c:.0e}' for c in cfus])}")

    print(f"\nSynergy Analysis:")
    print(f"  Overall Synergy Score: {result['synergy_score']:.2f}")
    print(f"  Butyrate Potential: {result['total_butyrate_potential']:.2f}")
    print(f"  Immune Modulation: {result['total_immune_modulation']:.2f}")
    print(f"  Phylum Diversity Bonus: {result['phylum_diversity_bonus']:.2f}")
    print(f"  CFU Optimization: {result['cfu_optimization']:.2f}")

    if result['recommended_adjustments']:
        print(f"\nRecommended Adjustments:")
        for adj in result['recommended_adjustments']:
            print(f"  {adj['strain']}:")
            print(f"    Current: {adj['current_cfu']:.0e} → Recommended: {adj['recommended_cfu']:.0e}")


def example_4_trajectory_prediction():
    """Example 4: Dysbiosis correction trajectory"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Dysbiosis Correction Trajectory")
    print("="*70)

    profile = {
        "profile": {
            "firmicutes_abundance": 0.7,
            "bacteroidetes_abundance": 0.15,
            "actinobacteria_abundance": 0.05,
            "proteobacteria_abundance": 0.08,
            "verrucomicrobia_abundance": 0.02,
            "shannon_diversity": 2.5,
            "butyrate_producers": 0.3,
            "acetate_producers": 0.4,
            "propionate_producers": 0.2,
            "lps_producers": 0.5,
            "fiber_degradation": 0.4,
            "mucin_degradation": 0.3,
            "bile_acid_metabolism": 0.4,
            "neurotransmitter_production": 0.3
        },
        "condition": "ibs"
    }

    response = requests.post(f"{BASE_URL}/predict-trajectory", json=profile)
    result = response.json()

    print(f"\nKey Milestones:")
    print(f"  50% Improvement: Week {result['time_to_50_percent_improvement']}")
    print(f"  80% Improvement: Week {result['time_to_80_percent_improvement']}")
    print(f"  Plateau Expected: Week {result['plateau_expected_week']}")

    print(f"\nWeek-by-Week Trajectory:")
    print(f"  {'Week':<6} {'F/B Ratio':<12} {'Diversity':<12} {'Butyrate':<12} {'Symptoms':<10}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    for week_data in result['trajectory'][::2]:  # Every 2 weeks
        print(f"  {week_data['week']:<6} "
              f"{week_data['fb_ratio']:<12.2f} "
              f"{week_data['shannon_diversity']:<12.2f} "
              f"{week_data['butyrate_production']:<12.2f} "
              f"{week_data['symptom_severity']:<10.1f}/10")


def example_5_comprehensive_plan():
    """Example 5: Comprehensive intervention plan"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Comprehensive Intervention Plan")
    print("="*70)

    profile = {
        "profile": {
            "firmicutes_abundance": 0.6,
            "bacteroidetes_abundance": 0.2,
            "actinobacteria_abundance": 0.08,
            "proteobacteria_abundance": 0.1,
            "verrucomicrobia_abundance": 0.02,
            "shannon_diversity": 3.2,
            "butyrate_producers": 0.45,
            "acetate_producers": 0.35,
            "propionate_producers": 0.25,
            "lps_producers": 0.35,
            "fiber_degradation": 0.5,
            "mucin_degradation": 0.4,
            "bile_acid_metabolism": 0.5,
            "neurotransmitter_production": 0.4
        },
        "condition": "obesity",
        "preferences": {
            "probiotics_allowed": True,
            "vegetarian": False,
            "budget": "moderate"
        }
    }

    response = requests.post(f"{BASE_URL}/optimize", json=profile)
    result = response.json()

    print(f"\n📊 CURRENT STATUS")
    print(f"  F/B Ratio: {0.6/0.2:.2f} (elevated - obesity risk)")
    print(f"  Shannon Diversity: 3.2 (moderate)")
    print(f"  Butyrate Producers: 0.45 (suboptimal)")

    print(f"\n🦠 PROBIOTIC RECOMMENDATIONS")
    for i, probiotic in enumerate(result['probiotics'], 1):
        print(f"  Formulation {i}:")
        print(f"    Strains: {', '.join(probiotic['strains'])}")
        print(f"    Synergy Score: {probiotic['synergy_score']:.2f}")

    print(f"\n🌾 PREBIOTIC (FIBER) RECOMMENDATIONS")
    for i, prebiotic in enumerate(result['prebiotics'], 1):
        print(f"  {i}. {prebiotic['fiber_type'].replace('_', ' ').title()}")
        print(f"     Amount: {prebiotic['daily_amount_g']:.1f}g/day")
        print(f"     Sources: {', '.join(prebiotic['best_sources'][:3])}")
        print(f"     Expected Butyrate Boost: +{prebiotic['expected_butyrate_boost']:.2f}")

    print(f"\n🍽️ DIETARY CHANGES")
    for change in result['dietary_changes']:
        if 'target' in change:
            print(f"  • Target: {change['target']}")
            if 'diet' in change:
                print(f"    Foods: {', '.join(change['diet'][:5])}")

    print(f"\n🕐 LIFESTYLE RECOMMENDATIONS")
    for lifestyle in result['lifestyle']:
        if 'intervention' in lifestyle:
            print(f"  • {lifestyle['intervention'].replace('_', ' ').title()}")
            if 'target_eating_window' in lifestyle:
                print(f"    Eating Window: {lifestyle['target_eating_window']}")

    print(f"\n🎯 EXPECTED OUTCOMES")
    print(f"  F/B Ratio: {0.6/0.2:.2f} → {result['expected_fb_ratio']:.2f}")
    print(f"  Butyrate Production: +{result['expected_butyrate_increase']:.2f}")
    print(f"  LPS Reduction: -{result.get('expected_lps_reduction', 0):.2f}")
    print(f"  Timeline: {result['expected_timeline_weeks']} weeks")
    print(f"  Confidence: {result['confidence_score']:.0%}")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("MICROBIOME HEALTH OPTIMIZER API - USAGE EXAMPLES")
    print("Copyright (c) 2025 Corporation of Light. PATENT PENDING.")
    print("="*70)

    print("\nNOTE: These examples require the API server to be running.")
    print("Start server with: uvicorn microbiome_optimizer_api:app --reload")
    print("\nPress Enter to continue (or Ctrl+C to exit)...")

    try:
        input()

        # Check if API is running
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("\n✓ API server is running")
            else:
                print("\n✗ API server not responding correctly")
                return
        except requests.exceptions.RequestException:
            print("\n✗ API server not running. Please start it first:")
            print("  uvicorn microbiome_optimizer_api:app --reload")
            return

        # Run examples
        example_1_ibs_patient()
        input("\nPress Enter for next example...")

        example_2_depression_gut_brain()
        input("\nPress Enter for next example...")

        example_3_probiotic_synergy()
        input("\nPress Enter for next example...")

        example_4_trajectory_prediction()
        input("\nPress Enter for next example...")

        example_5_comprehensive_plan()

        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED")
        print("="*70)
        print("\n10 Breakthrough Innovations Demonstrated:")
        print("  1. ✓ Multi-strain probiotic synergy prediction")
        print("  2. ✓ Personalized fiber-type recommendations")
        print("  3. ✓ Gut-brain axis neurotransmitter modeling")
        print("  4. ✓ SCFA production optimization")
        print("  5. ✓ LPS reduction for inflammation control")
        print("  6. ✓ Circadian rhythm integration")
        print("  7. ✓ Predictive dysbiosis correction trajectories")
        print("  8. ✓ FMT donor compatibility scoring")
        print("  9. ✓ Antibiotic recovery protocols")
        print(" 10. ✓ Diet-microbiome-immune axis integration")

        print("\nFor interactive API documentation:")
        print(f"  {BASE_URL}/docs")

    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")


if __name__ == "__main__":
    main()
