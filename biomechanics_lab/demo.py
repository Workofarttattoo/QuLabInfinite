# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Biomechanics Lab Demo
Comprehensive demonstrations of all biomechanics simulation capabilities
"""

import numpy as np
import json
from biomechanics_lab import (
    TissueMechanics,
    BiomaterialTesting,
    ProstheticDesign,
    MusculoskeletalDynamics
)


def run_all_demos():
    """Run all biomechanics demonstrations"""
    results = {
        'lab_name': 'Biomechanics Laboratory',
        'demonstrations': {}
    }

    # 1. Tissue Mechanics
    print("=" * 60)
    print("TISSUE MECHANICS SIMULATION")
    print("=" * 60)

    tissue = TissueMechanics()

    # Mooney-Rivlin hyperelastic (soft tissue)
    lambda_stretch = np.linspace(1.0, 2.0, 50)
    mooney_result = tissue.mooney_rivlin_hyperelastic(
        lambda_stretch=lambda_stretch,
        C10=0.05,  # MPa
        C01=0.02   # MPa
    )
    print(f"Soft Tissue Hyperelastic Response:")
    print(f"Stretch 1.5x stress: {mooney_result['cauchy_stress_MPa'][25]:.3f} MPa")
    print(f"Stretch 2.0x stress: {mooney_result['cauchy_stress_MPa'][-1]:.3f} MPa")

    results['demonstrations']['hyperelastic_tissue'] = mooney_result

    # Maxwell viscoelastic
    time_s = np.linspace(0, 100, 100)
    maxwell_result = tissue.maxwell_viscoelastic(
        time_s=time_s,
        stress_MPa=1.0,
        elastic_modulus_MPa=10.0,
        viscosity_MPa_s=100.0
    )
    print(f"\nViscoelastic Relaxation:")
    print(f"Relaxation time: {maxwell_result['relaxation_time_s']:.1f} s")
    print(f"Half-life: {maxwell_result['half_life_s']:.1f} s")

    results['demonstrations']['viscoelastic'] = maxwell_result

    # Von Mises failure
    failure_result = tissue.von_mises_failure_criterion(
        sigma_1=50,
        sigma_2=30,
        sigma_3=10,
        yield_strength_MPa=60
    )
    print(f"\nFailure Analysis:")
    print(f"Von Mises stress: {failure_result['von_mises_stress_MPa']:.2f} MPa")
    print(f"Safety factor: {failure_result['safety_factor']:.2f}")
    print(f"Failure predicted: {failure_result['failure_predicted']}")

    results['demonstrations']['von_mises_failure'] = failure_result

    # 2. Biomaterial Testing
    print("\n" + "=" * 60)
    print("BIOMATERIAL TESTING")
    print("=" * 60)

    bio_test = BiomaterialTesting()

    # Tensile test
    tensile_result = bio_test.tensile_test_simulation(
        youngs_modulus_GPa=110,  # Titanium alloy
        yield_strength_MPa=880,
        ultimate_strength_MPa=950,
        elongation_at_break_percent=14
    )
    print(f"Titanium Alloy Tensile Test:")
    print(f"Yield strength: {tensile_result['yield_strength_MPa']:.0f} MPa")
    print(f"Ultimate strength: {tensile_result['ultimate_strength_MPa']:.0f} MPa")
    print(f"Toughness: {tensile_result['toughness_MJ_per_m3']:.1f} MJ/m³")

    results['demonstrations']['tensile_test'] = tensile_result

    # Fatigue analysis
    stress_amplitude = np.array([400, 500, 600, 700, 800])
    fatigue_result = bio_test.sn_fatigue_curve(
        stress_amplitude_MPa=stress_amplitude,
        ultimate_strength_MPa=950
    )
    print(f"\nFatigue Analysis:")
    print(f"Endurance limit: {fatigue_result['endurance_limit_MPa']:.1f} MPa")
    print(f"Cycles at 600 MPa: {fatigue_result['cycles_to_failure'][2]:.2e}")

    results['demonstrations']['fatigue'] = fatigue_result

    # 3. Prosthetic Design
    print("\n" + "=" * 60)
    print("PROSTHETIC DESIGN")
    print("=" * 60)

    prosthetic = ProstheticDesign()

    # Socket pressure
    socket_result = prosthetic.socket_pressure_distribution(
        body_weight_kg=75,
        socket_area_cm2=150,
        load_bearing_percent=100
    )
    print(f"Prosthetic Socket Analysis:")
    print(f"Average pressure: {socket_result['average_pressure_kPa']:.1f} kPa")
    print(f"Comfort level: {socket_result['comfort_level']}")

    results['demonstrations']['socket_pressure'] = socket_result

    # Gait GRF
    gait_result = prosthetic.gait_ground_reaction_forces(
        body_weight_kg=75,
        stride_length_m=1.4,
        cadence_steps_per_min=110
    )
    print(f"\nGait Ground Reaction Forces:")
    print(f"Body weight: {gait_result['body_weight_N']:.0f} N")
    print(f"Peak vertical force: {gait_result['peak_vertical_force_N']:.0f} N")
    print(f"Peak A-P force: {gait_result['peak_ap_force_N']:.0f} N")

    results['demonstrations']['gait_grf'] = gait_result

    # 4. Musculoskeletal Dynamics
    print("\n" + "=" * 60)
    print("MUSCULOSKELETAL DYNAMICS")
    print("=" * 60)

    musculo = MusculoskeletalDynamics()

    # Hill muscle model
    velocity = np.linspace(-1, 2, 50)
    hill_result = musculo.hill_muscle_model(
        velocity_m_per_s=velocity,
        max_force_N=1000,
        max_velocity_m_per_s=2.0
    )
    print(f"Hill Muscle Model:")
    print(f"Isometric force: {hill_result['max_force_N']:.0f} N")
    print(f"Max velocity: {hill_result['max_velocity_m_per_s']:.1f} m/s")

    results['demonstrations']['hill_muscle'] = hill_result

    # Bone bending stress
    bone_result = musculo.bone_stress_from_bending(
        length_m=0.4,  # Femur length
        outer_diameter_mm=28,
        inner_diameter_mm=18,
        applied_force_N=2000
    )
    print(f"\nFemur Bending Stress:")
    print(f"Max stress: {bone_result['max_bending_stress_MPa']:.1f} MPa")
    print(f"Safety factor: {bone_result['safety_factor']:.2f}")

    results['demonstrations']['bone_stress'] = bone_result

    # Joint moment
    joint_result = musculo.joint_moment_calculation(
        muscle_force_N=500,
        moment_arm_cm=4.0,
        joint_angle_deg=90
    )
    print(f"\nJoint Mechanics:")
    print(f"Joint moment: {joint_result['joint_moment_Nm']:.1f} N·m")
    print(f"Mechanical advantage: {joint_result['mechanical_advantage']:.2f}")

    results['demonstrations']['joint_moment'] = joint_result

    print("\n" + "=" * 60)
    print("BIOMECHANICS LAB DEMO COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = run_all_demos()

    # Save results to JSON
    with open('biomechanics_lab_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to: biomechanics_lab_results.json")
