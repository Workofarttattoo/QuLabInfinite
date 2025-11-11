"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Cardiology Laboratory Demo
"""

from cardiology_lab import CardiologyLaboratory, CardiacDrug
import numpy as np


def main():
    """Run cardiology lab demonstration"""
    print("QuLabInfinite Cardiology Laboratory - Demo")
    print("=" * 70)

    lab = CardiologyLaboratory(seed=42)

    # Demo 1: Cardiac cycle at different heart rates
    print("\n1. Cardiac Cycle Simulation")
    print("-" * 70)

    heart_rates = [60, 70, 100, 150]
    for hr in heart_rates:
        cycle = lab.simulate_cardiac_cycle(heart_rate=hr, contractility=1.0, duration_s=2)
        print(f"\nHeart rate: {hr} bpm")
        print(f"  Stroke volume: {cycle['stroke_volume']:.1f} mL")
        print(f"  Cardiac output: {cycle['cardiac_output']:.2f} L/min")
        print(f"  Ejection fraction: {cycle['ejection_fraction']:.1f}%")
        print(f"  Blood pressure: {cycle['systolic_bp']:.0f}/{cycle['diastolic_bp']:.0f} mmHg")

    # Demo 2: Contractility effects (heart failure vs normal)
    print("\n2. Contractility Effects")
    print("-" * 70)

    contractilities = [0.5, 0.75, 1.0, 1.25]
    for contractility in contractilities:
        cycle = lab.simulate_cardiac_cycle(heart_rate=70, contractility=contractility, duration_s=2)

        condition = "Heart failure" if contractility < 0.75 else \
                   "Below normal" if contractility < 1.0 else \
                   "Normal" if contractility == 1.0 else "Hypercontractile"

        print(f"\nContractility: {contractility:.2f} ({condition})")
        print(f"  Stroke volume: {cycle['stroke_volume']:.1f} mL")
        print(f"  Ejection fraction: {cycle['ejection_fraction']:.1f}%")
        print(f"  Cardiac output: {cycle['cardiac_output']:.2f} L/min")

    # Demo 3: ECG rhythm analysis
    print("\n3. ECG Rhythm Analysis")
    print("-" * 70)

    rhythms = [
        ('normal_sinus', 70),
        ('atrial_fib', 120),
        ('ventricular_tach', 180)
    ]

    for rhythm, hr in rhythms:
        ecg = lab.generate_ecg_signal(heart_rate=hr, rhythm=rhythm, duration_s=10)

        print(f"\n{rhythm.replace('_', ' ').title()}:")
        print(f"  Heart rate: {ecg.heart_rate:.0f} bpm")
        print(f"  PR interval: {ecg.pr_interval:.0f} ms " +
              ("(normal 120-200)" if rhythm == 'normal_sinus' else ""))
        print(f"  QRS duration: {ecg.qrs_duration:.0f} ms " +
              ("(normal <120)" if ecg.qrs_duration < 120 else "(WIDE - abnormal)"))
        print(f"  QT interval: {ecg.qt_interval:.0f} ms")

        # Clinical interpretation
        if rhythm == 'normal_sinus':
            print("  ✓ Normal sinus rhythm")
        elif rhythm == 'atrial_fib':
            print("  ⚠ Irregular rhythm, no P waves - Atrial Fibrillation")
        else:
            print("  ⚠ CRITICAL: Wide complex tachycardia - Ventricular Tachycardia")

    # Demo 4: Blood flow in different vessels
    print("\n4. Blood Flow Dynamics")
    print("-" * 70)

    vessels = [
        ("Aorta", 12, 10, 30),
        ("Femoral artery", 4, 20, 40),
        ("Coronary artery", 2, 5, 50),
        ("Arteriole", 0.05, 0.5, 30),
        ("Capillary", 0.004, 0.05, 15)
    ]

    print(f"\n{'Vessel':<20} {'Radius':<10} {'Flow Rate':<15} {'Velocity':<15} {'Re':<10} {'Type'}")
    print("-" * 90)

    for name, radius, length, pressure in vessels:
        flow = lab.calculate_blood_flow(radius, length, pressure)
        print(f"{name:<20} {radius:>8.3f} mm {flow.flow_rate:>12.2f} mL/s "
              f"{flow.velocity:>12.2f} cm/s {flow.reynolds_number:>8.0f}  {flow.flow_type}")

    # Demo 5: Cardiac drug effects
    print("\n5. Cardiac Drug Effects Over Time")
    print("-" * 70)

    drugs = [
        (CardiacDrug.BETA_BLOCKER, 50, "Metoprolol"),
        (CardiacDrug.ACE_INHIBITOR, 20, "Lisinopril"),
        (CardiacDrug.CALCIUM_CHANNEL_BLOCKER, 10, "Amlodipine")
    ]

    for drug, dose, name in drugs:
        effect = lab.simulate_drug_effect(drug, dose_mg=dose, duration_hours=24)

        print(f"\n{name} ({drug.value}):")
        print(f"  Dose: {dose} mg")
        print(f"  Half-life: {effect['half_life_hours']:.1f} hours")
        print(f"  Peak effect: {effect['peak_effect_time']:.1f} hours")

        # Effects at peak
        peak_idx = int(effect['peak_effect_time'] * 100 / 24)
        if peak_idx < len(effect['heart_rate']):
            print(f"  Baseline HR: 70 bpm → Peak effect: {effect['heart_rate'][peak_idx]:.0f} bpm")
            print(f"  Baseline BP: 120/80 → Peak effect: "
                  f"{effect['systolic_bp'][peak_idx]:.0f}/{effect['diastolic_bp'][peak_idx]:.0f} mmHg")

    # Demo 6: Stenosis simulation (narrowed vessel)
    print("\n6. Coronary Artery Stenosis")
    print("-" * 70)

    print("\nCoronary artery (5 cm length, 50 mmHg pressure drop):")
    stenosis_levels = [
        (2.0, "Normal (0% stenosis)"),
        (1.5, "Mild (25% diameter reduction)"),
        (1.0, "Moderate (50% diameter reduction)"),
        (0.7, "Severe (65% diameter reduction)")
    ]

    for radius, description in stenosis_levels:
        flow = lab.calculate_blood_flow(radius, 5, 50)
        reduction = (1 - flow.flow_rate / 12.57) * 100  # Compared to normal

        print(f"\n{description}:")
        print(f"  Radius: {radius:.1f} mm")
        print(f"  Flow rate: {flow.flow_rate:.2f} mL/s ({reduction:+.0f}% vs normal)")
        print(f"  Velocity: {flow.velocity:.1f} cm/s")
        print(f"  Flow: {flow.flow_type}")

        if reduction > 75:
            print("  ⚠ CRITICAL: Severely reduced flow - immediate intervention needed")
        elif reduction > 50:
            print("  ⚠ Significant flow reduction - revascularization recommended")

    print("\n" + "=" * 70)
    print("Demo complete!")


if __name__ == "__main__":
    main()
