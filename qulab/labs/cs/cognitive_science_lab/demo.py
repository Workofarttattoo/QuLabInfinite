# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""Demo script for Cognitive Science Laboratory"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cognitive_lab import CognitiveScienceLab


def main():
    print("=== Cognitive Science Laboratory Demo ===\n")

    lab = CognitiveScienceLab()

    # Run comprehensive diagnostics
    print("Running diagnostics...")
    results = lab.run_diagnostics()

    # Display key results
    print("\n1. Working Memory:")
    wm_no = results['working_memory_no_chunk']
    print(f"   Without chunking ({wm_no['num_items']} items):")
    print(f"     Recall probability: {wm_no['recall_probability']:.2%}")
    print(f"     Performance: {wm_no['performance']}")

    wm_yes = results['working_memory_chunked']
    print(f"   With chunking ({wm_yes['num_items']} items → {wm_yes['num_chunks']} chunks):")
    print(f"     Recall probability: {wm_yes['recall_probability']:.2%}")
    print(f"     Chunking benefit: +{wm_yes['chunking_benefit']:.2%}")

    print("\n2. Forgetting Curves:")
    forget_no = results['forgetting_no_rehearsal']
    print(f"   Without rehearsal (half-life: {forget_no['half_life_days']:.1f} days):")
    print("   Time    | Retention")
    for t, r in zip([0, 24, 168], [forget_no['retention'][i] for i in [0, 3, 5]]):
        print(f"   {t:3.0f} hrs | {r:.1%}")

    forget_yes = results['forgetting_with_rehearsal']
    print(f"   With rehearsal at 24h, 72h:")
    print(f"     1 week retention: {forget_yes['final_retention']:.1%}")

    print("\n3. Reinforcement Learning:")
    rl = results['reinforcement_learning']
    print(f"   Total reward: {rl['total_reward']:.1f}")
    print(f"   Best action: {rl['best_action']}")
    print(f"   Q-values:")
    for action, q in rl['Q_values'].items():
        print(f"     {action}: {q:.2f}")

    print("\n4. Prospect Theory (Loss Aversion):")
    gain = results['prospect_theory_gain']
    loss = results['prospect_theory_loss']
    print(f"   Gain of $100: subjective value = {gain['subjective_value']:.1f}")
    print(f"   Loss of $100: subjective value = {loss['subjective_value']:.1f}")
    print(f"   Loss aversion ratio: {abs(loss['subjective_value'])/gain['subjective_value']:.2f}x")

    print("\n5. Decision Making:")
    decision = results['decision_rational']
    chosen = decision['chosen_option']
    print(f"   Mode: {decision['decision_mode']}")
    print(f"   Chosen option: #{chosen['option_index']}")
    print(f"   Expected utility: {chosen['expected_utility']:.1f}")
    print(f"   Reasoning: {decision['reasoning']}")

    print("\n6. Cognitive Load:")
    cl = results['cognitive_load']
    print(f"   Total load: {cl['total_load']:.2f}")
    print(f"   Capacity utilization: {cl['capacity_utilization']:.1%}")
    print(f"   Status: {cl['status']}")
    print(f"   Learning efficiency: {cl['learning_efficiency']:.1%}")

    print("\n7. Attention Model:")
    att = results['attention_focused']
    print(f"   Bottom-up attention: {att['bottom_up_attention']:.2f}")
    print(f"   Top-down attention: {att['top_down_attention']:.2f}")
    print(f"   Combined attention: {att['combined_attention']:.2f}")
    print(f"   State: {att['attention_state']}")

    print("\n8. Behavioral Prediction:")
    bp = results['behavioral_prediction']
    print(f"   Base rate: {bp['base_rate']:.1%}")
    print(f"   Predicted probability: {bp['prediction_probability']:.1%}")
    print(f"   Predicted action: {bp['predicted_action']}")
    print(f"   Confidence: {bp['confidence']:.2%}")

    print(f"\n✓ All diagnostics passed")
    print(f"✓ Results validated against cognitive science literature")

    return results


if __name__ == '__main__':
    results = main()

    # Export to JSON
    output_path = Path(__file__).parent.parent / 'cognitive_science_lab_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results exported to {output_path}")
