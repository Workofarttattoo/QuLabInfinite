#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Pharmacology Training System - Interactive Course on Drug Science

Comprehensive training on:
- Pharmacology fundamentals
- Pharmacokinetics (ADME)
- Pharmacodynamics (dose-response)
- Drug interactions
- Practical applications
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from oncology_lab.drug_response import DRUG_DATABASE
from typing import Dict, List, Tuple


class PharmacologyTrainer:
    """Interactive pharmacology training system"""

    def __init__(self):
        self.db = DRUG_DATABASE
        self.current_lesson = 0
        self.quiz_scores = []

    def start_course(self):
        """Start the interactive course"""
        print("\n" + "=" * 80)
        print("  💊 PHARMACOLOGY TRAINING SYSTEM")
        print("  Learn Drug Science from Fundamentals to Advanced")
        print("=" * 80)

        print("\n📚 Course Modules:")
        print("  1. Pharmacology Fundamentals")
        print("  2. Pharmacokinetics (ADME) - What the Body Does to the Drug")
        print("  3. Pharmacodynamics (PD) - What the Drug Does to the Body")
        print("  4. Dose-Response Relationships")
        print("  5. Drug Interactions & Combinations")
        print("  6. Practical Applications with Real Drugs")
        print("  7. Final Assessment")
        print("  8. Quick Reference Guide")
        print("  0. Exit")

        while True:
            choice = input("\n📖 Select module (0-8): ").strip()

            if choice == '0':
                self._show_completion_summary()
                break
            elif choice == '1':
                self.lesson_fundamentals()
            elif choice == '2':
                self.lesson_pharmacokinetics()
            elif choice == '3':
                self.lesson_pharmacodynamics()
            elif choice == '4':
                self.lesson_dose_response()
            elif choice == '5':
                self.lesson_drug_interactions()
            elif choice == '6':
                self.lesson_practical_applications()
            elif choice == '7':
                self.final_assessment()
            elif choice == '8':
                self.quick_reference()
            else:
                print("Invalid choice. Try again.")

    # ========================================================================
    # MODULE 1: FUNDAMENTALS
    # ========================================================================

    def lesson_fundamentals(self):
        """Module 1: Pharmacology Fundamentals"""
        print("\n" + "=" * 80)
        print("  MODULE 1: PHARMACOLOGY FUNDAMENTALS")
        print("=" * 80)

        print("""
🔬 What is Pharmacology?

Pharmacology is the science of drugs and their effects on living systems.
It combines two major disciplines:

1. PHARMACOKINETICS (PK) - "What the body does to the drug"
   - How drugs are absorbed, distributed, metabolized, and eliminated (ADME)
   - Determines drug concentration over time

2. PHARMACODYNAMICS (PD) - "What the drug does to the body"
   - How drugs produce their effects
   - Dose-response relationships
   - Drug-receptor interactions

🎯 Key Concepts:

• DRUG: Any substance that changes biological function
• DOSE: Amount of drug administered
• CONCENTRATION: Amount of drug per volume (e.g., µM, mg/L)
• BIOAVAILABILITY: Fraction of drug that reaches systemic circulation
• HALF-LIFE: Time for drug concentration to decrease by 50%
• IC50: Concentration for 50% inhibition (measures potency)
• EC50: Concentration for 50% of maximum effect
• Emax: Maximum possible effect of a drug
""")

        self._wait_for_continue()

        print("""
📊 Drug Classification:

Drugs are classified by:

1. CHEMICAL STRUCTURE
   - Small molecules (most drugs, <900 Da)
   - Biologics (proteins, antibodies, >5000 Da)

2. MECHANISM OF ACTION
   - DNA damaging (chemotherapy)
   - Enzyme inhibitors (kinase inhibitors)
   - Receptor modulators (hormone therapies)
   - Immune modulators (checkpoint inhibitors)

3. THERAPEUTIC USE
   - Chemotherapy
   - Targeted therapy
   - Immunotherapy
   - Hormone therapy
   - Supportive care

💡 Example from Database:

Cisplatin:
  - Small molecule (300 Da)
  - DNA crosslinker (mechanism)
  - Chemotherapy (therapeutic class)
  - IC50 = 1.5 µM (potency)
  - Half-life = 0.8 hours (short-acting)

Pembrolizumab:
  - Biologic antibody (149,000 Da)
  - PD-1 inhibitor (mechanism)
  - Immunotherapy (therapeutic class)
  - IC50 = 0.0001 µM (very potent)
  - Half-life = 26 days (long-acting)
""")

        self._wait_for_continue()
        self._quiz_fundamentals()

    def _quiz_fundamentals(self):
        """Quiz on fundamentals"""
        print("\n📝 QUIZ: Fundamentals")
        print("-" * 80)

        score = 0
        total = 3

        # Q1
        print("\nQ1: What does pharmacokinetics study?")
        print("  a) What the drug does to the body")
        print("  b) What the body does to the drug")
        print("  c) Drug-receptor interactions")
        ans = input("Your answer (a/b/c): ").strip().lower()
        if ans == 'b':
            print("✓ Correct! PK = body does to drug")
            score += 1
        else:
            print("✗ Incorrect. Answer: b (body → drug)")

        # Q2
        print("\nQ2: What does IC50 measure?")
        print("  a) Maximum effect")
        print("  b) Half-life")
        print("  c) Concentration for 50% inhibition (potency)")
        ans = input("Your answer (a/b/c): ").strip().lower()
        if ans == 'c':
            print("✓ Correct! IC50 measures potency")
            score += 1
        else:
            print("✗ Incorrect. Answer: c (IC50 = potency)")

        # Q3
        print("\nQ3: Which has longer half-life?")
        print("  a) Small molecule (cisplatin, 0.8 hours)")
        print("  b) Antibody (pembrolizumab, 26 days)")
        ans = input("Your answer (a/b): ").strip().lower()
        if ans == 'b':
            print("✓ Correct! Antibodies have much longer half-lives")
            score += 1
        else:
            print("✗ Incorrect. Answer: b (antibodies last weeks)")

        self.quiz_scores.append(('Fundamentals', score, total))
        print(f"\n📊 Score: {score}/{total} ({score/total*100:.0f}%)")

    # ========================================================================
    # MODULE 2: PHARMACOKINETICS
    # ========================================================================

    def lesson_pharmacokinetics(self):
        """Module 2: Pharmacokinetics (ADME)"""
        print("\n" + "=" * 80)
        print("  MODULE 2: PHARMACOKINETICS (PK) - ADME")
        print("=" * 80)

        print("""
💊 ADME: The Journey of a Drug

Pharmacokinetics describes the 4 stages of drug movement:

┌─────────────────────────────────────────────────────────┐
│  A - ABSORPTION    → Drug enters bloodstream             │
│  D - DISTRIBUTION  → Drug spreads through body           │
│  M - METABOLISM    → Drug is chemically modified         │
│  E - EXCRETION     → Drug leaves the body                │
└─────────────────────────────────────────────────────────┘

────────────────────────────────────────────────────────────

1️⃣ ABSORPTION

How drug enters systemic circulation.

Routes & Bioavailability (F):
• IV (intravenous):   F = 100% (directly into blood)
• Oral:               F = 5-99% (must pass through gut & liver)
• Subcutaneous/IM:    F = 50-100%

Example: Curcumin
  - Oral bioavailability = 1% (very poor!)
  - Most is destroyed in gut/liver (first-pass metabolism)
  - Solution: Take with piperine (increases absorption)

────────────────────────────────────────────────────────────

2️⃣ DISTRIBUTION

How drug spreads through body compartments.

Key Parameter: Volume of Distribution (Vd)

Vd = Total drug amount / Plasma concentration

Low Vd (3-20 L):     Stays in blood
  Example: Antibodies (trastuzumab, Vd = 3 L)

High Vd (>100 L):    Distributes widely into tissues
  Example: Doxorubicin (Vd = 800 L)

Very high Vd (>1000 L): Accumulates in fat/tissues
  Example: Hydroxychloroquine (Vd = 5000 L!)

Vd affects:
• How long drug stays in body
• How much drug reaches tumor
• Dosing intervals

────────────────────────────────────────────────────────────

3️⃣ METABOLISM

Chemical modification of drugs (usually in liver).

Phases:
• Phase I: Oxidation (CYP450 enzymes)
  - Makes drug more water-soluble
  - Usually inactivates drug

• Phase II: Conjugation
  - Adds large groups (glucuronide, sulfate)
  - Prepares for excretion

Drug Interactions via Metabolism:
• CYP450 Inhibitors → Drug levels increase
  Example: Grapefruit juice inhibits CYP3A4

• CYP450 Inducers → Drug levels decrease
  Example: Rifampin induces CYP3A4

────────────────────────────────────────────────────────────

4️⃣ EXCRETION

How drug leaves the body.

Main Routes:
• Kidneys (urine) - 70% of drugs
• Liver (bile → feces) - 20% of drugs
• Lungs (volatile drugs)
• Other (sweat, breast milk)

Key Parameter: Clearance (CL)

CL = Volume of plasma cleared per time (L/h)

High CL = Drug removed quickly
Low CL = Drug stays longer

Half-Life (t½):
t½ = 0.693 × Vd / CL

This determines dosing interval:
• Short t½ (< 6h): Dose multiple times daily
• Long t½ (> 24h): Once daily or less

Example: Cisplatin
  t½ = 0.8 hours → Need to keep giving it?
  No! It binds to DNA (irreversible), so effect lasts
  despite short half-life
""")

        self._wait_for_continue()
        self._demonstrate_pk_curves()
        self._quiz_pharmacokinetics()

    def _demonstrate_pk_curves(self):
        """Show PK concentration curves"""
        print("\n📊 Generating PK concentration curves...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Single dose IV
        ax = axes[0, 0]
        t = np.linspace(0, 24, 100)

        # Short half-life drug
        c_short = 100 * np.exp(-0.693 * t / 2.0)  # t½ = 2h
        # Long half-life drug
        c_long = 100 * np.exp(-0.693 * t / 12.0)  # t½ = 12h

        ax.plot(t, c_short, 'r-', linewidth=2, label='Short t½ (2h)')
        ax.plot(t, c_long, 'b-', linewidth=2, label='Long t½ (12h)')
        ax.axhline(y=10, color='g', linestyle='--', label='Therapeutic level')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Concentration (µM)')
        ax.set_title('Single IV Dose - Effect of Half-Life')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # 2. Oral vs IV
        ax = axes[0, 1]
        t = np.linspace(0, 24, 100)

        # IV - immediate peak
        c_iv = 100 * np.exp(-0.693 * t / 4.0)

        # Oral - delayed peak, lower bioavailability
        ka = 2.0  # absorption rate
        ke = 0.693 / 4.0  # elimination rate
        c_oral = 50 * (ka / (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))

        ax.plot(t, c_iv, 'r-', linewidth=2, label='IV (F=100%)')
        ax.plot(t, c_oral, 'b-', linewidth=2, label='Oral (F=50%)')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Concentration (µM)')
        ax.set_title('IV vs Oral - Same Dose')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Multiple dosing - steady state
        ax = axes[1, 0]
        t = np.linspace(0, 96, 500)

        # Simulate multiple doses every 8 hours
        c_multi = np.zeros_like(t)
        dose_interval = 8.0
        t_half = 8.0

        for dose_time in np.arange(0, 96, dose_interval):
            mask = t >= dose_time
            c_multi[mask] += 50 * np.exp(-0.693 * (t[mask] - dose_time) / t_half)

        ax.plot(t, c_multi, 'b-', linewidth=2)
        ax.axhline(y=25, color='g', linestyle='--', linewidth=2, label='Therapeutic window')
        ax.axhline(y=75, color='r', linestyle='--', linewidth=2, label='Toxic level')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Concentration (µM)')
        ax.set_title('Multiple Dosing - Reaches Steady State')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.fill_between(t, 25, 75, alpha=0.2, color='green', label='Therapeutic range')

        # 4. Effect of Vd on concentration
        ax = axes[1, 1]
        dose = 100  # mg
        vd_values = [5, 20, 100, 500]  # L

        for vd in vd_values:
            c0 = dose / vd  # Initial concentration
            t = np.linspace(0, 24, 100)
            c = c0 * np.exp(-0.693 * t / 6.0)  # t½ = 6h
            ax.plot(t, c, linewidth=2, label=f'Vd = {vd} L')

        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Plasma Concentration (mg/L)')
        ax.set_title('Effect of Vd on Drug Concentration')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('pharmacology_pk_curves.png', dpi=150)
        print("✓ Saved to: pharmacology_pk_curves.png")
        plt.close()

    def _quiz_pharmacokinetics(self):
        """Quiz on PK"""
        print("\n📝 QUIZ: Pharmacokinetics")
        print("-" * 80)

        score = 0
        total = 4

        # Q1
        print("\nQ1: If a drug has high first-pass metabolism, what happens?")
        print("  a) High oral bioavailability")
        print("  b) Low oral bioavailability")
        print("  c) No effect on bioavailability")
        ans = input("Your answer: ").strip().lower()
        if ans == 'b':
            print("✓ Correct! First-pass = destroyed in liver → low F")
            score += 1
        else:
            print("✗ Incorrect. Answer: b")

        # Q2
        print("\nQ2: Drug A: Vd=5L, Drug B: Vd=500L. Same dose. Which has higher plasma concentration?")
        print("  a) Drug A (low Vd)")
        print("  b) Drug B (high Vd)")
        ans = input("Your answer: ").strip().lower()
        if ans == 'a':
            print("✓ Correct! Low Vd = stays in blood = higher concentration")
            score += 1
        else:
            print("✗ Incorrect. Answer: a (C = Dose/Vd)")

        # Q3
        print("\nQ3: Drug has t½ = 24 hours. How often should you dose it?")
        print("  a) Every 6 hours")
        print("  b) Once daily")
        print("  c) Once weekly")
        ans = input("Your answer: ").strip().lower()
        if ans == 'b':
            print("✓ Correct! Long t½ → less frequent dosing")
            score += 1
        else:
            print("✗ Incorrect. Answer: b (t½=24h → daily dosing)")

        # Q4
        print("\nQ4: What does clearance (CL) measure?")
        print("  a) Total drug in body")
        print("  b) Volume of plasma cleared per time")
        print("  c) Drug concentration")
        ans = input("Your answer: ").strip().lower()
        if ans == 'b':
            print("✓ Correct! CL = volume cleared/time (L/h)")
            score += 1
        else:
            print("✗ Incorrect. Answer: b")

        self.quiz_scores.append(('Pharmacokinetics', score, total))
        print(f"\n📊 Score: {score}/{total} ({score/total*100:.0f}%)")

    # ========================================================================
    # MODULE 3: PHARMACODYNAMICS
    # ========================================================================

    def lesson_pharmacodynamics(self):
        """Module 3: Pharmacodynamics"""
        print("\n" + "=" * 80)
        print("  MODULE 3: PHARMACODYNAMICS (PD)")
        print("=" * 80)

        print("""
🎯 What the Drug Does to the Body

Pharmacodynamics studies drug effects and mechanisms.

────────────────────────────────────────────────────────────

1️⃣ DRUG-RECEPTOR INTERACTIONS

Most drugs work by binding to receptors:

Drug + Receptor ⇌ Drug-Receptor Complex → Effect

Types of Drug Action:
• AGONIST: Activates receptor (mimics natural ligand)
• ANTAGONIST: Blocks receptor (prevents activation)
• PARTIAL AGONIST: Weak activation
• INVERSE AGONIST: Reduces baseline activity

Example: Tamoxifen
  - SERM (Selective Estrogen Receptor Modulator)
  - Antagonist in breast tissue (blocks estrogen)
  - Agonist in bone (protects bone density)
  - Mixed effects depending on tissue!

────────────────────────────────────────────────────────────

2️⃣ KEY PD PARAMETERS

IC50 (Inhibitory Concentration 50%)
  - Concentration needed for 50% inhibition
  - Measures POTENCY (how much drug you need)
  - Lower IC50 = more potent

  Example:
    Vincristine IC50 = 0.001 µM (very potent!)
    Vitamin C IC50 = 500 µM (need high doses)

EC50 (Effective Concentration 50%)
  - Concentration for 50% of maximum effect
  - Similar to IC50 but for activation

Emax (Maximum Effect)
  - Highest effect achievable
  - Measures EFFICACY (how well drug works)
  - Emax = 1.0 means 100% kill

  Example:
    Vincristine: Emax = 0.85 (85% max kill)
    Pembrolizumab: Emax = 0.6 (60% response rate)

Hill Coefficient (n)
  - Steepness of dose-response curve
  - n > 1: Cooperative binding (steep curve)
  - n = 1: Simple binding
  - n < 1: Negative cooperativity

────────────────────────────────────────────────────────────

3️⃣ THERAPEUTIC INDEX

Measures drug safety:

TI = Toxic Dose / Therapeutic Dose

High TI (>10):  Safe drug
Low TI (<3):    Narrow therapeutic window (dangerous!)

Example:
  Metformin: TI ~ 20 (very safe)
  Vincristine: TI ~ 2 (narrow window, easy to overdose)

────────────────────────────────────────────────────────────

4️⃣ DRUG SELECTIVITY

How specific is the drug?

Selective Drugs:
  - Hit one target
  - Fewer side effects
  - Example: Imatinib (BCR-ABL specific)

Non-Selective Drugs:
  - Hit multiple targets
  - More side effects, but sometimes more effective
  - Example: Sorafenib (hits RAF, VEGFR, PDGFR)

────────────────────────────────────────────────────────────

5️⃣ DOSE-RESPONSE RELATIONSHIP

Effect = (Emax × [Drug]^n) / (EC50^n + [Drug]^n)

This is the Hill equation!

At low dose:     Steep response
At EC50:         Half-maximal effect
At high dose:    Plateau (can't exceed Emax)

Key Points:
• Doubling dose ≠ double effect
• There's a ceiling (Emax)
• Shape determined by Hill coefficient
""")

        self._wait_for_continue()
        self._demonstrate_pd_curves()
        self._quiz_pharmacodynamics()

    def _demonstrate_pd_curves(self):
        """Show PD dose-response curves"""
        print("\n📊 Generating dose-response curves...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Basic dose-response
        ax = axes[0, 0]
        conc = np.logspace(-3, 2, 100)  # 0.001 to 100 µM

        ec50 = 1.0
        emax = 1.0
        hill = 2.0

        effect = (emax * conc**hill) / (ec50**hill + conc**hill)

        ax.plot(conc, effect * 100, 'b-', linewidth=3)
        ax.axhline(y=50, color='r', linestyle='--', label='EC50')
        ax.axvline(x=ec50, color='r', linestyle='--')
        ax.set_xscale('log')
        ax.set_xlabel('Drug Concentration (µM)')
        ax.set_ylabel('Effect (%)')
        ax.set_title('Dose-Response Curve (EC50=1.0 µM)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 2. Potency comparison (different EC50)
        ax = axes[0, 1]

        # Potent drug
        effect_potent = (1.0 * conc**2) / (0.01**2 + conc**2)
        # Moderate drug
        effect_moderate = (1.0 * conc**2) / (1.0**2 + conc**2)
        # Weak drug
        effect_weak = (1.0 * conc**2) / (100.0**2 + conc**2)

        ax.plot(conc, effect_potent * 100, 'r-', linewidth=2, label='Potent (EC50=0.01)')
        ax.plot(conc, effect_moderate * 100, 'g-', linewidth=2, label='Moderate (EC50=1.0)')
        ax.plot(conc, effect_weak * 100, 'b-', linewidth=2, label='Weak (EC50=100)')
        ax.set_xscale('log')
        ax.set_xlabel('Drug Concentration (µM)')
        ax.set_ylabel('Effect (%)')
        ax.set_title('Potency Comparison (same Emax)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Efficacy comparison (different Emax)
        ax = axes[1, 0]
        conc = np.logspace(-2, 2, 100)

        # Full agonist
        effect_full = (1.0 * conc**2) / (1.0**2 + conc**2)
        # Partial agonist
        effect_partial = (0.5 * conc**2) / (1.0**2 + conc**2)
        # Weak partial agonist
        effect_weak = (0.2 * conc**2) / (1.0**2 + conc**2)

        ax.plot(conc, effect_full * 100, 'r-', linewidth=2, label='Full agonist (Emax=1.0)')
        ax.plot(conc, effect_partial * 100, 'g-', linewidth=2, label='Partial (Emax=0.5)')
        ax.plot(conc, effect_weak * 100, 'b-', linewidth=2, label='Weak partial (Emax=0.2)')
        ax.set_xscale('log')
        ax.set_xlabel('Drug Concentration (µM)')
        ax.set_ylabel('Effect (%)')
        ax.set_title('Efficacy Comparison (same EC50)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Hill coefficient effect
        ax = axes[1, 1]

        for hill in [0.5, 1.0, 2.0, 4.0]:
            effect = (1.0 * conc**hill) / (1.0**hill + conc**hill)
            ax.plot(conc, effect * 100, linewidth=2, label=f'Hill = {hill}')

        ax.set_xscale('log')
        ax.set_xlabel('Drug Concentration (µM)')
        ax.set_ylabel('Effect (%)')
        ax.set_title('Effect of Hill Coefficient (Cooperativity)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('pharmacology_pd_curves.png', dpi=150)
        print("✓ Saved to: pharmacology_pd_curves.png")
        plt.close()

    def _quiz_pharmacodynamics(self):
        """Quiz on PD"""
        print("\n📝 QUIZ: Pharmacodynamics")
        print("-" * 80)

        score = 0
        total = 4

        # Q1
        print("\nQ1: Drug A: IC50=0.1 µM, Drug B: IC50=10 µM. Which is more potent?")
        print("  a) Drug A (lower IC50)")
        print("  b) Drug B (higher IC50)")
        ans = input("Your answer: ").strip().lower()
        if ans == 'a':
            print("✓ Correct! Lower IC50 = more potent = need less drug")
            score += 1
        else:
            print("✗ Incorrect. Answer: a (lower IC50 = higher potency)")

        # Q2
        print("\nQ2: Drug A: Emax=1.0, Drug B: Emax=0.5. Which has better efficacy?")
        print("  a) Drug A (higher Emax)")
        print("  b) Drug B (lower Emax)")
        ans = input("Your answer: ").strip().lower()
        if ans == 'a':
            print("✓ Correct! Higher Emax = better efficacy = works better")
            score += 1
        else:
            print("✗ Incorrect. Answer: a (higher Emax = better efficacy)")

        # Q3
        print("\nQ3: Therapeutic Index (TI) = 2. Is this drug safe?")
        print("  a) Yes, very safe")
        print("  b) No, narrow therapeutic window")
        ans = input("Your answer: ").strip().lower()
        if ans == 'b':
            print("✓ Correct! Low TI (<3) = dangerous, easy to overdose")
            score += 1
        else:
            print("✗ Incorrect. Answer: b (TI<3 is dangerous)")

        # Q4
        print("\nQ4: A drug binds one receptor. What type is this?")
        print("  a) Selective (specific)")
        print("  b) Non-selective (hits many targets)")
        ans = input("Your answer: ").strip().lower()
        if ans == 'a':
            print("✓ Correct! One target = selective")
            score += 1
        else:
            print("✗ Incorrect. Answer: a")

        self.quiz_scores.append(('Pharmacodynamics', score, total))
        print(f"\n📊 Score: {score}/{total} ({score/total*100:.0f}%)")

    # ========================================================================
    # MODULE 4: DOSE-RESPONSE
    # ========================================================================

    def lesson_dose_response(self):
        """Module 4: Dose-Response Relationships"""
        print("\n" + "=" * 80)
        print("  MODULE 4: DOSE-RESPONSE RELATIONSHIPS")
        print("=" * 80)

        print("""
📈 Understanding Drug Effects vs Dose

The relationship between drug dose and biological effect.

────────────────────────────────────────────────────────────

1️⃣ THE HILL EQUATION

Effect = (Emax × [Drug]^n) / (EC50^n + [Drug]^n)

Where:
• Emax = Maximum possible effect
• [Drug] = Drug concentration
• EC50 = Concentration for 50% effect
• n = Hill coefficient (cooperativity)

────────────────────────────────────────────────────────────

2️⃣ CURVE REGIONS

┌────────────────────────────────────────┐
│ LOW DOSE (< 0.1 × EC50)                │
│ • Linear relationship                   │
│ • Doubling dose ≈ doubles effect       │
│ • Most sensitive to dose changes        │
│                                        │
│ MIDDLE DOSE (0.1-10 × EC50)            │
│ • Steep response                       │
│ • Small dose changes = big effect      │
│ • Clinical dosing usually here         │
│                                        │
│ HIGH DOSE (> 10 × EC50)                │
│ • Plateau region                       │
│ • Increasing dose = minimal effect     │
│ • Already at Emax                      │
└────────────────────────────────────────┘

────────────────────────────────────────────────────────────

3️⃣ REAL EXAMPLES FROM DATABASE

Vincristine (very potent, steep curve):
  IC50 = 0.001 µM
  Emax = 0.85
  Hill = 3.5 (very steep!)

  Dose    Effect
  0.0001  5%      ← Below threshold
  0.001   50%     ← EC50, steep rise
  0.01    85%     ← Maximum effect

Metformin (less potent, shallow curve):
  IC50 = 50 µM
  Emax = 0.5
  Hill = 1.5

  Dose    Effect
  5       5%      ← Minimal effect
  50      25%     ← EC50
  500     47%     ← Approaching max

Key Insight:
• Vincristine: Small dose range (0.001-0.01 µM)
• Metformin: Large dose range (10-1000 µM)
• Different drugs = different optimal dosing!

────────────────────────────────────────────────────────────

4️⃣ COMBINATION EFFECTS

When combining drugs:

ADDITIVE: 1 + 1 = 2
  • Independent mechanisms
  • Effect = Drug A + Drug B
  • Example: Two different chemos

SYNERGISTIC: 1 + 1 = 3
  • Mechanisms interact
  • Effect > Drug A + Drug B
  • Example: BRAF inhibitor + MEK inhibitor

ANTAGONISTIC: 1 + 1 = 0.5
  • Mechanisms oppose
  • Effect < Drug A + Drug B
  • Avoid these combinations!

────────────────────────────────────────────────────────────

5️⃣ PRACTICAL IMPLICATIONS

Choosing Dose:
1. Start below EC50 (test tolerance)
2. Titrate up to therapeutic range
3. Don't exceed 10× EC50 (no benefit, more toxicity)
4. Consider PK: dose to maintain plasma levels

Monitoring:
• Measure plasma levels (if possible)
• Watch for therapeutic effect
• Monitor for toxicity
• Adjust based on response
""")

        self._wait_for_continue()
        self._demonstrate_real_drug_curves()

    def _demonstrate_real_drug_curves(self):
        """Show real drug dose-response curves"""
        print("\n📊 Generating real drug comparisons...")

        # Select interesting drugs
        drugs_to_plot = [
            ('vincristine', 'Very Potent'),
            ('cisplatin', 'Moderate'),
            ('metformin', 'Weak Potency'),
            ('vitamin_d3', 'Very Weak'),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        for idx, (drug_name, label) in enumerate(drugs_to_plot):
            if drug_name not in self.db:
                continue

            drug = self.db[drug_name]
            ax = axes[idx // 2, idx % 2]

            # Generate concentration range
            conc = np.logspace(np.log10(drug.ic50/100), np.log10(drug.ic50*100), 200)

            # Calculate effect using Hill equation
            effect = (drug.emax * conc**drug.hill_coefficient) / \
                     (drug.ic50**drug.hill_coefficient + conc**drug.hill_coefficient)

            ax.plot(conc, effect * 100, 'b-', linewidth=3)
            ax.axhline(y=drug.emax * 50, color='r', linestyle='--', label=f'EC50 region')
            ax.axvline(x=drug.ic50, color='r', linestyle='--')

            ax.set_xscale('log')
            ax.set_xlabel('Concentration (µM)')
            ax.set_ylabel('Effect (%)')
            ax.set_title(f'{drug.name} - {label}\nIC50={drug.ic50} µM, Emax={drug.emax}, Hill={drug.hill_coefficient}')
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        plt.savefig('pharmacology_real_drugs.png', dpi=150)
        print("✓ Saved to: pharmacology_real_drugs.png")
        plt.close()

    # ========================================================================
    # MODULE 5: DRUG INTERACTIONS
    # ========================================================================

    def lesson_drug_interactions(self):
        """Module 5: Drug Interactions"""
        print("\n" + "=" * 80)
        print("  MODULE 5: DRUG INTERACTIONS & COMBINATIONS")
        print("=" * 80)

        print("""
🔗 How Drugs Interact with Each Other

Drug interactions can be:
• Beneficial (synergy)
• Harmful (toxicity)
• Neutral (additive)

────────────────────────────────────────────────────────────

1️⃣ PHARMACOKINETIC INTERACTIONS

Drugs affecting each other's ADME:

A) Absorption Interactions
   • Food: Decreases absorption of many drugs
   • pH changes: Antacids reduce absorption
   • Chelation: Calcium binds to drugs

   Example: Take iron supplements 2h apart from drugs

B) Distribution Interactions
   • Protein binding displacement
   • One drug kicks another off proteins
   • Increases free (active) drug

   Example: Warfarin + Aspirin (bleeding risk)

C) Metabolism Interactions ⚠️ MOST COMMON

   CYP450 Inhibition:
   • One drug blocks metabolism of another
   • Drug B levels INCREASE (toxicity risk!)

   Example:
     Grapefruit juice + Statins
     → Inhibits CYP3A4
     → Statin levels increase 3-5×
     → Muscle toxicity risk

   CYP450 Induction:
   • One drug speeds up metabolism of another
   • Drug B levels DECREASE (loss of efficacy!)

   Example:
     Rifampin + Birth control pills
     → Induces CYP3A4
     → Hormone levels decrease
     → Pregnancy risk!

D) Excretion Interactions
   • Competition for kidney transporters
   • One drug blocks excretion of another

   Example: Probenecid + Penicillin
   → Blocks penicillin excretion
   → Penicillin levels increase (therapeutic use!)

────────────────────────────────────────────────────────────

2️⃣ PHARMACODYNAMIC INTERACTIONS

Drugs affecting each other's EFFECTS:

A) Additive (1 + 1 = 2)
   • Same mechanism, independent action
   • Total effect = sum of individual effects

   Example: Two beta-blockers
   → Redundant (no benefit)

B) Synergistic (1 + 1 = 3+)
   • Mechanisms complement each other
   • Total effect > sum of parts

   Examples:

   BRAF + MEK inhibitors (melanoma):
   • BRAF inhibitor alone: 50% response
   • MEK inhibitor alone: 30% response
   • Together: 75% response (synergy!)
   • Mechanism: Block same pathway at 2 points

   Cisplatin + Radiation:
   • Cisplatin makes DNA more sensitive to radiation
   • 2× effect compared to either alone

   PD-1 inhibitor + CTLA-4 inhibitor:
   • Different checkpoint targets
   • Release brakes on immune system
   • Higher response than either alone

C) Antagonistic (1 + 1 = 0.5)
   • Mechanisms oppose each other
   • AVOID these combinations!

   Example: Agonist + Antagonist of same receptor
   → They fight each other, reduced effect

────────────────────────────────────────────────────────────

3️⃣ CANCER DRUG COMBINATIONS

Principles of Combination Therapy:

1. Different Mechanisms
   - Hit cancer from multiple angles
   - Harder for resistance to develop

2. Non-Overlapping Toxicities
   - Each drug tolerable alone
   - Don't add toxicities together

3. Proven in Clinical Trials
   - Not all logical combos work!
   - Need evidence

Successful Combinations:

FOLFOX (colorectal cancer):
• 5-FU (antimetabolite)
• Oxaliplatin (DNA crosslinker)
• Leucovorin (enhances 5-FU)
→ Standard of care, proven synergy

R-CHOP (lymphoma):
• Rituximab (CD20 antibody)
• Cyclophosphamide (alkylator)
• Doxorubicin (topoisomerase)
• Vincristine (microtubule)
• Prednisone (steroid)
→ 5 drugs, different mechanisms, curative!

Dabrafenib + Trametinib (melanoma):
• Dabrafenib (BRAF inhibitor)
• Trametinib (MEK inhibitor)
→ Synergistic, delays resistance

────────────────────────────────────────────────────────────

4️⃣ PREDICTING INTERACTIONS

Check for:

✓ Same drug class → Likely additive/toxic
✓ Same enzyme metabolism → PK interaction risk
✓ Same toxicity → Cumulative toxicity
✓ Complementary mechanisms → Potential synergy
✓ Clinical trial data → Proven combinations

Tools:
• Drug interaction databases
• Simulation software (like this oncology lab!)
• Clinical pharmacist consultation

────────────────────────────────────────────────────────────

5️⃣ PRACTICAL EXAMPLE

Patient wants to combine:
• Cisplatin (chemo)
• Metformin (metabolic)
• Curcumin (natural anti-inflammatory)

Analysis:

Cisplatin + Metformin:
• Different mechanisms ✓
• Non-overlapping toxicity ✓
• Evidence: Phase 2 trials show benefit ✓
→ Good combination!

Cisplatin + Curcumin:
• Different mechanisms ✓
• Both cause nausea (additive side effect) ⚠️
• Evidence: Mixed results in trials ⚠️
→ Possible, monitor carefully

Metformin + Curcumin:
• Both affect metabolism ✓
• Minimal toxicity overlap ✓
• Evidence: Pre-clinical synergy ✓
→ Good combination!

Verdict: All three together is reasonable
         Monitor for nausea (cisplatin + curcumin)
""")

        self._wait_for_continue()

    # ========================================================================
    # MODULE 6: PRACTICAL APPLICATIONS
    # ========================================================================

    def lesson_practical_applications(self):
        """Module 6: Practical Applications"""
        print("\n" + "=" * 80)
        print("  MODULE 6: PRACTICAL APPLICATIONS")
        print("=" * 80)

        print("""
🔬 Applying Pharmacology to Real Protocols

Let's analyze actual drugs from the database using pharmacology principles!
""")

        # Show 3 detailed drug profiles
        example_drugs = ['cisplatin', 'pembrolizumab', 'metformin']

        for drug_name in example_drugs:
            if drug_name not in self.db:
                continue

            drug = self.db[drug_name]

            print(f"\n{'=' * 70}")
            print(f"  {drug.name.upper()} - Complete Pharmacology Profile")
            print(f"{'=' * 70}")

            # PK Analysis
            print(f"\n📊 PHARMACOKINETICS:")
            print(f"  Route: {drug.route}")
            print(f"  Bioavailability: {drug.pk_model.bioavailability*100:.0f}%")
            if drug.pk_model.bioavailability < 0.3:
                print(f"    → Poor absorption! Consider higher dose or IV route")
            elif drug.pk_model.bioavailability > 0.8:
                print(f"    → Excellent absorption! Oral dosing effective")

            print(f"\n  Volume of Distribution: {drug.pk_model.volume_of_distribution:.0f} L")
            if drug.pk_model.volume_of_distribution < 20:
                print(f"    → Stays in blood (good for blood cancers)")
            elif drug.pk_model.volume_of_distribution > 100:
                print(f"    → Widely distributed (reaches solid tumors)")

            print(f"\n  Half-Life: {drug.pk_model.half_life:.1f} hours")
            if drug.pk_model.half_life < 6:
                print(f"    → Short-acting! Need frequent dosing or continuous infusion")
            elif drug.pk_model.half_life > 24:
                print(f"    → Long-acting! Once daily or less frequent dosing")

            print(f"\n  Clearance: {drug.pk_model.clearance:.1f} L/h")
            print(f"  Protein Binding: {drug.pk_model.protein_binding*100:.0f}%")

            # PD Analysis
            print(f"\n🎯 PHARMACODYNAMICS:")
            print(f"  Mechanism: {drug.mechanism_of_action}")
            print(f"  Targets: {', '.join(drug.target_proteins[:3])}")

            print(f"\n  IC50: {drug.ic50} µM")
            if drug.ic50 < 0.01:
                print(f"    → Very potent! Low doses effective")
            elif drug.ic50 > 10:
                print(f"    → Weak potency, need high doses")

            print(f"  Emax: {drug.emax}")
            if drug.emax > 0.8:
                print(f"    → High efficacy! Can achieve strong tumor kill")
            elif drug.emax < 0.5:
                print(f"    → Moderate efficacy, may need combinations")

            print(f"  Hill Coefficient: {drug.hill_coefficient}")
            if drug.hill_coefficient > 2:
                print(f"    → Steep dose-response! Small dose changes = big effect")

            # Dosing
            print(f"\n💊 DOSING:")
            print(f"  Standard Dose: {drug.standard_dose_mg} mg")
            print(f"  Interval: Every {drug.dosing_interval_hours/24:.1f} days")

            # Safety
            print(f"\n⚠️ SAFETY:")
            toxicities = []
            if getattr(drug, 'myelosuppression', 0) > 0.3:
                toxicities.append(f"Myelosuppression ({drug.myelosuppression:.1f})")
            if getattr(drug, 'neurotoxicity', 0) > 0.3:
                toxicities.append(f"Neurotoxicity ({drug.neurotoxicity:.1f})")
            if getattr(drug, 'cardiotoxicity', 0) > 0.3:
                toxicities.append(f"Cardiotoxicity ({drug.cardiotoxicity:.1f})")

            if toxicities:
                print(f"  Major toxicities: {', '.join(toxicities)}")
            else:
                print(f"  Generally well-tolerated!")

            # Clinical use
            print(f"\n🏥 CLINICAL USE:")
            if drug.fda_approved:
                print(f"  FDA Approved: {drug.approval_year}")
                print(f"  Indications: {', '.join(drug.approved_indications[:3])}")
            else:
                print(f"  Experimental/Off-label use")

            self._wait_for_continue()

    # ========================================================================
    # FINAL ASSESSMENT
    # ========================================================================

    def final_assessment(self):
        """Comprehensive final exam"""
        print("\n" + "=" * 80)
        print("  🎓 FINAL ASSESSMENT")
        print("=" * 80)

        print("\nComprehensive test on all modules!")

        score = 0
        total = 10

        questions = [
            ("What does ADME stand for?", ["a) Absorption, Distribution, Metabolism, Excretion", "b) Active Drug Molecular Effect", "c) Advanced Drug Manufacturing Equipment"], "a"),
            ("Drug with t½=4h, dosed Q12h. Will it accumulate?", ["a) Yes (dose more frequent than elimination)", "b) No (eliminated between doses)"], "b"),
            ("IC50=0.001 vs IC50=100. Which is more potent?", ["a) 0.001 (lower = more potent)", "b) 100 (higher = more potent)"], "a"),
            ("Therapeutic Index = 1.5. Safe?", ["a) Yes, very safe", "b) No, narrow window"], "b"),
            ("Grapefruit juice inhibits CYP3A4. What happens to drug levels?", ["a) Increase (dangerous!)", "b) Decrease", "c) No change"], "a"),
            ("Synergistic drugs mean:", ["a) 1+1=2 (additive)", "b) 1+1=3 (synergy)", "c) 1+1=0.5 (antagonistic)"], "b"),
            ("Which has longer half-life?", ["a) Small molecule (500 Da)", "b) Antibody (150,000 Da)"], "b"),
            ("Oral bioavailability=5%. Why?", ["a) High first-pass metabolism", "b) Poor absorption", "c) Both"], "c"),
            ("Emax=0.95 vs Emax=0.5. Which has better efficacy?", ["a) 0.95 (higher efficacy)", "b) 0.5"], "a"),
            ("Best combination strategy:", ["a) Same mechanism (additive)", "b) Different mechanisms (synergy potential)", "c) Opposing mechanisms"], "b"),
        ]

        for i, (question, options, correct) in enumerate(questions, 1):
            print(f"\nQ{i}: {question}")
            for option in options:
                print(f"  {option}")
            ans = input("Your answer: ").strip().lower()
            if ans == correct:
                print("✓ Correct!")
                score += 1
            else:
                print(f"✗ Incorrect. Answer: {correct}")

        self.quiz_scores.append(('Final Assessment', score, total))

        print(f"\n{'=' * 80}")
        print(f"  FINAL SCORE: {score}/{total} ({score/total*100:.0f}%)")
        print(f"{'=' * 80}")

        if score >= 8:
            print("\n🎉 EXCELLENT! You have mastered pharmacology!")
            print("   You understand PK, PD, and drug interactions.")
        elif score >= 6:
            print("\n✓ GOOD! You have solid pharmacology knowledge.")
            print("  Review the modules you missed for mastery.")
        else:
            print("\n📚 Keep learning! Review the course modules.")
            print("   Pharmacology takes time to master.")

    # ========================================================================
    # QUICK REFERENCE
    # ========================================================================

    def quick_reference(self):
        """Quick reference guide"""
        print("\n" + "=" * 80)
        print("  📖 PHARMACOLOGY QUICK REFERENCE")
        print("=" * 80)

        reference = """
┌────────────────────────────────────────────────────────────┐
│ PHARMACOKINETICS (PK) - Body → Drug                       │
├────────────────────────────────────────────────────────────┤
│ Absorption (F)    How much drug enters blood              │
│ Distribution (Vd) Where drug goes in body                  │
│ Metabolism        How drug is modified                     │
│ Excretion (CL)    How drug leaves body                     │
│                                                            │
│ Key Equations:                                             │
│   t½ = 0.693 × Vd / CL                                    │
│   C = Dose / Vd                                           │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ PHARMACODYNAMICS (PD) - Drug → Body                        │
├────────────────────────────────────────────────────────────┤
│ IC50    Potency (lower = more potent)                     │
│ Emax    Efficacy (higher = works better)                  │
│ Hill    Steepness (>1 = steep, cooperative)               │
│                                                            │
│ Hill Equation:                                             │
│   E = (Emax × C^n) / (IC50^n + C^n)                       │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ DRUG INTERACTIONS                                          │
├────────────────────────────────────────────────────────────┤
│ Additive:      1 + 1 = 2  (independent)                   │
│ Synergistic:   1 + 1 = 3  (beneficial)                    │
│ Antagonistic:  1 + 1 = 0  (harmful)                       │
│                                                            │
│ CYP450 Inhibitor → Drug levels INCREASE                   │
│ CYP450 Inducer   → Drug levels DECREASE                   │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ INTERPRETATION GUIDE                                       │
├────────────────────────────────────────────────────────────┤
│ F < 0.3          Poor absorption (use IV)                 │
│ F > 0.8          Good absorption (oral OK)                │
│                                                            │
│ Vd < 20 L        Stays in blood                           │
│ Vd > 100 L       Widely distributed                       │
│                                                            │
│ t½ < 6h          Need frequent dosing                     │
│ t½ > 24h         Once daily or less                       │
│                                                            │
│ IC50 < 0.01 µM   Very potent                              │
│ IC50 > 10 µM     Weak potency                             │
│                                                            │
│ Emax > 0.8       High efficacy                            │
│ Emax < 0.5       Moderate efficacy                        │
│                                                            │
│ TI > 10          Safe drug                                │
│ TI < 3           Narrow window (dangerous)                │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ DOSING STRATEGY                                            │
├────────────────────────────────────────────────────────────┤
│ 1. Start at effective dose (around EC50)                  │
│ 2. Adjust for PK (t½ determines interval)                 │
│ 3. Monitor response (PD)                                   │
│ 4. Watch for toxicity                                      │
│ 5. Don't exceed 10× EC50 (no benefit)                     │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ COMBINATION PRINCIPLES                                     │
├────────────────────────────────────────────────────────────┤
│ ✓ Different mechanisms                                     │
│ ✓ Non-overlapping toxicities                              │
│ ✓ Proven in clinical trials                               │
│ ✓ Consider PK interactions                                │
│ ✗ Avoid antagonistic combos                               │
└────────────────────────────────────────────────────────────┘
"""
        print(reference)

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _wait_for_continue(self):
        """Wait for user to continue"""
        input("\nPress Enter to continue...")

    def _show_completion_summary(self):
        """Show final completion summary"""
        if not self.quiz_scores:
            print("\nNo quizzes completed yet. Try the lessons!")
            return

        print("\n" + "=" * 80)
        print("  📊 LEARNING SUMMARY")
        print("=" * 80)

        total_score = 0
        total_possible = 0

        print("\nModule Scores:")
        for module, score, possible in self.quiz_scores:
            pct = score/possible*100
            total_score += score
            total_possible += possible
            status = "✓" if pct >= 70 else "✗"
            print(f"  {status} {module:25s}: {score}/{possible} ({pct:.0f}%)")

        if total_possible > 0:
            final_pct = total_score/total_possible*100
            print(f"\n{'=' * 80}")
            print(f"  OVERALL: {total_score}/{total_possible} ({final_pct:.0f}%)")
            print(f"{'=' * 80}")

            if final_pct >= 80:
                print("\n🎓 CONGRATULATIONS! You've mastered pharmacology!")
                print("   You're ready to design and analyze drug protocols.")
            elif final_pct >= 60:
                print("\n✓ Good progress! Keep practicing.")
            else:
                print("\n📚 Review the course materials and try again.")

        print("\nThank you for learning pharmacology!")


def main():
    """Main entry point"""
    trainer = PharmacologyTrainer()
    trainer.start_course()


if __name__ == "__main__":
    main()
