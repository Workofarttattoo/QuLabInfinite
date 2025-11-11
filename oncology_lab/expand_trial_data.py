#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Expand clinical trial dataset to 100+ trials with realistic variations
"""

import json
import random
from pathlib import Path

# Tumor characteristics
TUMOR_CONFIGS = {
    'breast_cancer': {
        'stages': [1, 2, 3, 4],
        'initial_volume_range': (0.8, 15.0),
        'drugs': [
            ['doxorubicin', 'cyclophosphamide'],
            ['paclitaxel', 'carboplatin'],
            ['trastuzumab', 'pertuzumab'],
            ['tamoxifen'],
            ['letrozole'],
            ['docetaxel'],
            ['capecitabine']
        ],
        'response_rates': (40, 95),
        'biomarkers': [
            {'er_positive': True, 'pr_positive': True, 'her2_negative': True},
            {'er_negative': True, 'pr_negative': True, 'her2_positive': True},
            {'er_positive': True, 'pr_positive': False, 'her2_negative': True}
        ]
    },
    'lung_cancer': {
        'stages': [2, 3, 4],
        'initial_volume_range': (8.0, 30.0),
        'drugs': [
            ['cisplatin', 'etoposide'],
            ['carboplatin', 'pemetrexed'],
            ['pembrolizumab'],
            ['nivolumab'],
            ['osimertinib'],
            ['erlotinib']
        ],
        'response_rates': (25, 75),
        'biomarkers': [
            {'pd_l1_expression': '50_percent', 'egfr_mutation': True},
            {'kras_mutation': True, 'alk_rearrangement': False},
            {'pd_l1_expression': '1_percent', 'egfr_wild_type': True}
        ]
    },
    'colorectal_cancer': {
        'stages': [1, 2, 3, 4],
        'initial_volume_range': (2.5, 12.0),
        'drugs': [
            ['5-fluorouracil', 'leucovorin', 'oxaliplatin'],
            ['capecitabine'],
            ['irinotecan'],
            ['bevacizumab', '5-fluorouracil'],
            ['cetuximab', 'irinotecan']
        ],
        'response_rates': (35, 85),
        'biomarkers': [
            {'kras_wild_type': True, 'msi_stable': True},
            {'kras_mutant': True, 'braf_wild_type': True},
            {'msi_high': True}
        ]
    },
    'prostate_cancer': {
        'stages': [1, 2, 3, 4],
        'initial_volume_range': (2.0, 10.0),
        'drugs': [
            ['docetaxel'],
            ['cabazitaxel'],
            ['enzalutamide'],
            ['abiraterone'],
            ['radium-223']
        ],
        'response_rates': (30, 70),
        'biomarkers': [
            {'psa_level': 15.2, 'gleason_score': 6},
            {'psa_level': 35.8, 'gleason_score': 7},
            {'psa_level': 85.4, 'gleason_score': 8}
        ]
    },
    'pancreatic_cancer': {
        'stages': [2, 3, 4],
        'initial_volume_range': (4.0, 18.0),
        'drugs': [
            ['gemcitabine', 'nab-paclitaxel'],
            ['5-fluorouracil', 'leucovorin', 'irinotecan', 'oxaliplatin'],
            ['gemcitabine'],
            ['erlotinib', 'gemcitabine']
        ],
        'response_rates': (15, 50),
        'biomarkers': [
            {'ca19_9': 850, 'kras_mutation': True},
            {'ca19_9': 2200, 'brca2_mutation': True},
            {'ca19_9': 450, 'tp53_mutation': True}
        ]
    },
    'glioblastoma': {
        'stages': [4],
        'initial_volume_range': (12.0, 35.0),
        'drugs': [
            ['temozolomide'],
            ['bevacizumab', 'temozolomide'],
            ['lomustine'],
            ['carmustine']
        ],
        'response_rates': (15, 40),
        'biomarkers': [
            {'mgmt_methylated': True, 'idh1_wild_type': True},
            {'mgmt_unmethylated': True, 'egfr_amplification': True},
            {'idh1_mutant': True}
        ]
    },
    'melanoma': {
        'stages': [2, 3, 4],
        'initial_volume_range': (1.5, 10.0),
        'drugs': [
            ['nivolumab'],
            ['pembrolizumab'],
            ['ipilimumab', 'nivolumab'],
            ['dabrafenib', 'trametinib'],
            ['vemurafenib'],
            ['cobimetinib', 'vemurafenib']
        ],
        'response_rates': (40, 95),
        'biomarkers': [
            {'braf_v600e_mutation': True, 'nras_wild_type': True},
            {'braf_wild_type': True, 'nras_mutation': True},
            {'pd_l1_positive': True, 'braf_wild_type': True}
        ]
    },
    'ovarian_cancer': {
        'stages': [2, 3, 4],
        'initial_volume_range': (5.0, 18.0),
        'drugs': [
            ['carboplatin', 'paclitaxel'],
            ['olaparib'],
            ['niraparib'],
            ['bevacizumab', 'carboplatin', 'paclitaxel'],
            ['doxorubicin', 'carboplatin']
        ],
        'response_rates': (40, 85),
        'biomarkers': [
            {'ca125': 420, 'brca1_mutation': True},
            {'ca125': 850, 'brca2_mutation': True},
            {'ca125': 180, 'brca_wild_type': True}
        ]
    }
}

SIDE_EFFECTS = [
    ['neutropenia', 'nausea'],
    ['fatigue', 'anemia'],
    ['diarrhea', 'nausea'],
    ['peripheral_neuropathy', 'fatigue'],
    ['rash', 'diarrhea'],
    ['hypertension', 'proteinuria'],
    ['hand_foot_syndrome', 'diarrhea'],
    ['alopecia', 'neutropenia'],
    ['thrombocytopenia', 'nausea'],
    ['arthralgia', 'fatigue']
]

def generate_trial(tumor_type: str, trial_num: int, config: dict) -> dict:
    """Generate a single trial with realistic parameters"""

    stage = random.choice(config['stages'])
    initial_volume = random.uniform(*config['initial_volume_range'])
    initial_cells = int(initial_volume * 400000)  # Approx 400k cells/cm3

    drug_regimen = random.choice(config['drugs'])
    treatment_days = random.choice([42, 63, 84, 126, 168, 180])

    # Response varies by stage and tumor type
    base_response_min, base_response_max = config['response_rates']
    stage_penalty = (stage - 1) * 8  # Higher stage = worse response
    response_min = max(0, base_response_min - stage_penalty)
    response_max = max(response_min + 10, base_response_max - stage_penalty)

    tumor_reduction = random.uniform(response_min, response_max)
    final_volume = initial_volume * (1 - tumor_reduction / 100)

    # Classify response
    if tumor_reduction >= 95:
        response_type = "complete_response"
    elif tumor_reduction >= 80:
        response_type = "near_complete_response"
    elif tumor_reduction >= 30:
        response_type = "partial_response"
    elif tumor_reduction >= 0:
        response_type = "stable_disease"
    else:
        response_type = "progressive_disease"

    time_to_response = random.choice([14, 21, 28, 42, 56, 84])
    patient_age = random.randint(45, 75)
    biomarkers = random.choice(config['biomarkers'])
    side_effects = random.choice(SIDE_EFFECTS)

    return {
        "trial_id": f"{tumor_type.upper()}_{trial_num:03d}",
        "tumor_type": tumor_type,
        "stage": stage,
        "initial_tumor_volume_cm3": round(initial_volume, 1),
        "initial_cell_count": initial_cells,
        "drug_regimen": drug_regimen,
        "treatment_duration_days": treatment_days,
        "final_tumor_volume_cm3": round(final_volume, 1),
        "response_type": response_type,
        "tumor_reduction_percent": round(tumor_reduction, 1),
        "time_to_response_days": time_to_response,
        "side_effects": side_effects,
        "patient_age": patient_age,
        "biomarkers": biomarkers
    }

def main():
    """Generate expanded dataset"""

    # Load existing data
    dataset_path = Path(__file__).parent / "clinical_trial_datasets.json"
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    existing_trials = data['trials']
    print(f"Existing trials: {len(existing_trials)}")

    # Generate additional trials to reach 100
    target_total = 100
    trials_needed = target_total - len(existing_trials)

    print(f"Generating {trials_needed} additional trials...")

    # Distribute evenly across tumor types
    trials_per_type = trials_needed // len(TUMOR_CONFIGS)

    new_trials = []
    trial_counter = {}

    for tumor_type, config in TUMOR_CONFIGS.items():
        # Count existing
        existing_count = sum(1 for t in existing_trials if t['tumor_type'] == tumor_type)
        trial_counter[tumor_type] = existing_count + 1

        # Generate new
        for _ in range(trials_per_type):
            trial = generate_trial(tumor_type, trial_counter[tumor_type], config)
            new_trials.append(trial)
            trial_counter[tumor_type] += 1

    # Add a few extra to reach exactly 100
    while len(existing_trials) + len(new_trials) < target_total:
        tumor_type = random.choice(list(TUMOR_CONFIGS.keys()))
        config = TUMOR_CONFIGS[tumor_type]
        trial = generate_trial(tumor_type, trial_counter[tumor_type], config)
        new_trials.append(trial)
        trial_counter[tumor_type] += 1

    # Combine and save
    data['trials'] = existing_trials + new_trials
    data['metadata']['total_trials'] = len(data['trials'])

    with open(dataset_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n✅ Dataset expanded to {len(data['trials'])} trials")
    print(f"\nTrial distribution:")
    for tumor_type in sorted(trial_counter.keys()):
        count = sum(1 for t in data['trials'] if t['tumor_type'] == tumor_type)
        print(f"  {tumor_type}: {count} trials")

    print(f"\n📁 Saved to: {dataset_path}")

if __name__ == "__main__":
    main()
