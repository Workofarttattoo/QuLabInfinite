#!/usr/bin/env python3
"""
Drug Synthesis Training Module - Learn Organic Chemistry and Drug Synthesis

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Interactive training system teaching:
- Organic chemistry fundamentals
- Functional groups and reactions
- Retrosynthetic analysis
- Common drug synthesis pathways
- Natural product extraction
- Pharmaceutical chemistry principles
"""

import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random


@dataclass
class FunctionalGroup:
    """Represents a chemical functional group"""
    name: str
    formula: str
    reactivity: str
    common_reactions: List[str]
    examples: List[str]


@dataclass
class Reaction:
    """Represents a chemical reaction"""
    name: str
    reactants: List[str]
    products: List[str]
    conditions: str
    mechanism_type: str
    example_use: str


@dataclass
class DrugSynthesis:
    """Represents a complete drug synthesis pathway"""
    drug_name: str
    target_structure: str
    synthesis_steps: List[str]
    key_reactions: List[str]
    starting_materials: List[str]
    yield_percent: float
    difficulty: str  # "beginner", "intermediate", "advanced"


class DrugSynthesisTrainer:
    """Interactive drug synthesis and organic chemistry trainer"""

    def __init__(self):
        self.score = 0
        self.total_questions = 0
        self.completed_modules = []

        # Build chemistry knowledge base
        self.functional_groups = self._init_functional_groups()
        self.reactions = self._init_reactions()
        self.syntheses = self._init_drug_syntheses()

    def _init_functional_groups(self) -> Dict[str, FunctionalGroup]:
        """Initialize functional group database"""
        return {
            "alcohol": FunctionalGroup(
                name="Alcohol",
                formula="R-OH",
                reactivity="Nucleophilic, can be oxidized",
                common_reactions=[
                    "Oxidation to aldehyde/ketone",
                    "Esterification",
                    "Dehydration to alkene",
                    "Substitution (SN1/SN2)"
                ],
                examples=["Ethanol", "Cholesterol", "Vitamin A"]
            ),
            "amine": FunctionalGroup(
                name="Amine",
                formula="R-NH2, R2NH, R3N",
                reactivity="Basic, nucleophilic",
                common_reactions=[
                    "Acylation to amide",
                    "Alkylation",
                    "Condensation with carbonyl",
                    "Diazotization (aromatic)"
                ],
                examples=["Dopamine", "Amphetamine", "Morphine"]
            ),
            "carbonyl": FunctionalGroup(
                name="Carbonyl (C=O)",
                formula="R-CO-R' (ketone) or R-CHO (aldehyde)",
                reactivity="Electrophilic carbon, nucleophilic oxygen",
                common_reactions=[
                    "Nucleophilic addition",
                    "Reduction to alcohol",
                    "Condensation reactions",
                    "Oxidation (aldehyde → acid)"
                ],
                examples=["Acetone", "Formaldehyde", "Cortisone"]
            ),
            "carboxylic_acid": FunctionalGroup(
                name="Carboxylic Acid",
                formula="R-COOH",
                reactivity="Acidic, can form esters/amides",
                common_reactions=[
                    "Esterification",
                    "Amide formation",
                    "Decarboxylation",
                    "Reduction to alcohol"
                ],
                examples=["Aspirin (acetylsalicylic acid)", "Ibuprofen", "Vitamin C"]
            ),
            "ester": FunctionalGroup(
                name="Ester",
                formula="R-COO-R'",
                reactivity="Can be hydrolyzed",
                common_reactions=[
                    "Hydrolysis (acidic/basic)",
                    "Transesterification",
                    "Reduction to alcohols",
                    "Claisen condensation"
                ],
                examples=["Aspirin", "Cocaine", "Heroin"]
            ),
            "amide": FunctionalGroup(
                name="Amide",
                formula="R-CO-NH2",
                reactivity="Stable, less reactive than esters",
                common_reactions=[
                    "Hydrolysis (harsh conditions)",
                    "Reduction to amine",
                    "Hofmann rearrangement",
                    "Dehydration to nitrile"
                ],
                examples=["Paracetamol (acetaminophen)", "Penicillin", "Asparagine"]
            ),
            "aromatic": FunctionalGroup(
                name="Aromatic Ring (Benzene)",
                formula="C6H5-",
                reactivity="Electrophilic aromatic substitution",
                common_reactions=[
                    "Nitration",
                    "Halogenation",
                    "Sulfonation",
                    "Friedel-Crafts alkylation/acylation"
                ],
                examples=["Benzene", "Toluene", "Most drugs (70%+ contain aromatic rings)"]
            ),
            "ether": FunctionalGroup(
                name="Ether",
                formula="R-O-R'",
                reactivity="Relatively unreactive",
                common_reactions=[
                    "Cleavage with HI/HBr",
                    "Peroxide formation (dangerous!)",
                    "Williamson ether synthesis (formation)"
                ],
                examples=["Diethyl ether", "Tetrahydrofuran (THF)", "Morphine"]
            ),
            "halide": FunctionalGroup(
                name="Alkyl Halide",
                formula="R-X (X = F, Cl, Br, I)",
                reactivity="Good leaving group",
                common_reactions=[
                    "SN1/SN2 substitution",
                    "E1/E2 elimination",
                    "Grignard reagent formation",
                    "Cross-coupling reactions"
                ],
                examples=["Chloroform", "Haloperidol", "Fluoroquinolones"]
            ),
        }

    def _init_reactions(self) -> Dict[str, Reaction]:
        """Initialize common reaction database"""
        return {
            "esterification": Reaction(
                name="Fischer Esterification",
                reactants=["Carboxylic acid", "Alcohol"],
                products=["Ester", "Water"],
                conditions="Acid catalyst (H2SO4), heat",
                mechanism_type="Nucleophilic acyl substitution",
                example_use="Aspirin synthesis from salicylic acid + acetic anhydride"
            ),
            "amide_formation": Reaction(
                name="Amide Bond Formation",
                reactants=["Carboxylic acid derivative", "Amine"],
                products=["Amide", "Leaving group"],
                conditions="Coupling reagents (DCC, EDC) or acid chloride",
                mechanism_type="Nucleophilic acyl substitution",
                example_use="Peptide synthesis, penicillin derivatives"
            ),
            "reduction": Reaction(
                name="Carbonyl Reduction",
                reactants=["Ketone/Aldehyde", "Reducing agent"],
                products=["Alcohol"],
                conditions="NaBH4 (mild) or LiAlH4 (strong)",
                mechanism_type="Nucleophilic addition",
                example_use="Steroid synthesis, alcohol formation"
            ),
            "grignard": Reaction(
                name="Grignard Reaction",
                reactants=["Alkyl halide + Mg", "Carbonyl compound"],
                products=["Alcohol (after workup)"],
                conditions="Dry ether, anhydrous conditions",
                mechanism_type="Nucleophilic addition",
                example_use="Carbon-carbon bond formation, building complex molecules"
            ),
            "suzuki": Reaction(
                name="Suzuki Cross-Coupling",
                reactants=["Aryl halide", "Aryl boronic acid"],
                products=["Biaryl compound"],
                conditions="Pd catalyst, base",
                mechanism_type="Palladium-catalyzed cross-coupling",
                example_use="Creating biphenyl structures in pharmaceuticals"
            ),
            "diels_alder": Reaction(
                name="Diels-Alder Cycloaddition",
                reactants=["Conjugated diene", "Dienophile (alkene)"],
                products=["Six-membered ring"],
                conditions="Heat (sometimes Lewis acid catalyst)",
                mechanism_type="[4+2] Cycloaddition",
                example_use="Building ring systems in natural product synthesis"
            ),
            "williamson": Reaction(
                name="Williamson Ether Synthesis",
                reactants=["Alkoxide ion", "Alkyl halide"],
                products=["Ether"],
                conditions="Strong base (NaH, NaOH), SN2 mechanism",
                mechanism_type="Nucleophilic substitution",
                example_use="Ether formation in drug synthesis"
            ),
        }

    def _init_drug_syntheses(self) -> Dict[str, DrugSynthesis]:
        """Initialize drug synthesis pathways"""
        return {
            "aspirin": DrugSynthesis(
                drug_name="Aspirin (Acetylsalicylic Acid)",
                target_structure="C9H8O4 (acetyl group on salicylic acid)",
                synthesis_steps=[
                    "1. Start with salicylic acid (from phenol + CO2, Kolbe-Schmitt reaction)",
                    "2. React with acetic anhydride in presence of acid catalyst",
                    "3. Acetylation of hydroxyl group occurs",
                    "4. Crystallize product from solution",
                    "5. Purify by recrystallization"
                ],
                key_reactions=["Esterification", "Kolbe-Schmitt carboxylation"],
                starting_materials=["Salicylic acid", "Acetic anhydride"],
                yield_percent=85.0,
                difficulty="beginner"
            ),
            "paracetamol": DrugSynthesis(
                drug_name="Paracetamol (Acetaminophen)",
                target_structure="C8H9NO2 (para-aminophenol + acetyl)",
                synthesis_steps=[
                    "1. Start with phenol",
                    "2. Nitrate to form para-nitrophenol",
                    "3. Reduce nitro group to amine (para-aminophenol)",
                    "4. Acetylate amine with acetic anhydride",
                    "5. Crystallize and purify"
                ],
                key_reactions=["Nitration", "Reduction", "Acetylation"],
                starting_materials=["Phenol", "Nitric acid", "Acetic anhydride"],
                yield_percent=75.0,
                difficulty="beginner"
            ),
            "ibuprofen": DrugSynthesis(
                drug_name="Ibuprofen",
                target_structure="C13H18O2 (propionic acid derivative)",
                synthesis_steps=[
                    "1. Start with isobutylbenzene",
                    "2. Friedel-Crafts acylation with acetic anhydride",
                    "3. Hydrogenation of carbonyl to alcohol",
                    "4. Dehydration to alkene",
                    "5. Hydroformylation (add CHO group)",
                    "6. Oxidation to carboxylic acid"
                ],
                key_reactions=["Friedel-Crafts", "Hydrogenation", "Oxidation"],
                starting_materials=["Isobutylbenzene", "Acetic anhydride"],
                yield_percent=65.0,
                difficulty="intermediate"
            ),
            "morphine": DrugSynthesis(
                drug_name="Morphine (Natural Product)",
                target_structure="C17H19NO3 (complex pentacyclic structure)",
                synthesis_steps=[
                    "NATURAL EXTRACTION (from opium poppy):",
                    "1. Harvest latex from Papaver somniferum seed pods",
                    "2. Dry latex to form opium (10-15% morphine)",
                    "3. Extract with water/acid to solubilize alkaloids",
                    "4. Basify to precipitate morphine",
                    "5. Purify by recrystallization",
                    "",
                    "TOTAL SYNTHESIS (extremely difficult):",
                    "6. Rice-Stork synthesis (22 steps)",
                    "7. Gates synthesis (14 steps, breakthrough 1952)",
                    "8. Modern syntheses: 10+ steps minimum"
                ],
                key_reactions=[
                    "Natural extraction (primary method)",
                    "Diels-Alder (synthesis)",
                    "Grewe cyclization",
                    "Multiple protecting group strategies"
                ],
                starting_materials=["Opium latex (natural) or complex organic precursors (synthesis)"],
                yield_percent=5.0,  # Total synthesis yield
                difficulty="advanced"
            ),
            "penicillin": DrugSynthesis(
                drug_name="Penicillin G",
                target_structure="β-lactam core + thiazolidine ring",
                synthesis_steps=[
                    "BIOSYNTHESIS (industrial method):",
                    "1. Fermentation of Penicillium chrysogenum fungus",
                    "2. Add side-chain precursor (phenylacetic acid for Pen G)",
                    "3. Extract from culture medium with organic solvent",
                    "4. Purify by chromatography",
                    "",
                    "SEMI-SYNTHESIS (penicillin derivatives):",
                    "5. Enzymatic cleavage of side chain (penicillin acylase)",
                    "6. Chemical coupling of new side chain",
                    "7. Generates ampicillin, amoxicillin, etc."
                ],
                key_reactions=[
                    "Biosynthesis (fermentation)",
                    "Enzymatic acylation/deacylation",
                    "Amide bond formation"
                ],
                starting_materials=["Penicillium culture", "Phenylacetic acid"],
                yield_percent=60.0,  # Fermentation yield
                difficulty="intermediate"
            ),
            "taxol": DrugSynthesis(
                drug_name="Taxol (Paclitaxel)",
                target_structure="C47H51NO14 (extremely complex diterpene)",
                synthesis_steps=[
                    "NATURAL EXTRACTION (original method):",
                    "1. Extract from bark of Pacific yew tree",
                    "2. Multiple chromatography steps",
                    "3. Extremely low yield (0.01% of bark weight)",
                    "",
                    "SEMI-SYNTHESIS (current industrial method):",
                    "4. Extract 10-deacetylbaccatin III from yew needles",
                    "5. Attach synthetic side chain (5-6 steps)",
                    "6. Final yield ~20% from precursor",
                    "",
                    "TOTAL SYNTHESIS (research only):",
                    "7. Holton synthesis (40+ steps)",
                    "8. Nicolaou synthesis (38 steps)",
                    "9. Not commercially viable"
                ],
                key_reactions=[
                    "Natural extraction + semi-synthesis (industrial)",
                    "Ester formation",
                    "Protecting group chemistry",
                    "Stereoselective synthesis"
                ],
                starting_materials=["Yew tree needles (10-deacetylbaccatin III)", "Synthetic side chain"],
                yield_percent=0.01,  # Natural extraction
                difficulty="advanced"
            ),
        }

    # ========== TRAINING MODULES ==========

    def start_course(self):
        """Main training menu"""
        print("\n" + "="*80)
        print("🧪 DRUG SYNTHESIS & ORGANIC CHEMISTRY TRAINING")
        print("="*80)
        print("\nLearn how drugs are made from basic chemistry to complex synthesis!")
        print("\n📚 Course Modules:")
        print("  1. Functional Groups - The Building Blocks")
        print("  2. Chemical Reactions - How Molecules Transform")
        print("  3. Retrosynthetic Analysis - Planning Synthesis")
        print("  4. Drug Synthesis Examples - Real Pathways")
        print("  5. Natural Product Chemistry - Extraction vs Synthesis")
        print("  6. Pharmaceutical Chemistry Principles")
        print("  7. Final Assessment")
        print("  8. Quick Reference Guide")
        print("  9. Exit")

        while True:
            choice = input("\n🎓 Select module (1-9): ").strip()

            if choice == "1":
                self.lesson_functional_groups()
            elif choice == "2":
                self.lesson_reactions()
            elif choice == "3":
                self.lesson_retrosynthesis()
            elif choice == "4":
                self.lesson_drug_syntheses()
            elif choice == "5":
                self.lesson_natural_products()
            elif choice == "6":
                self.lesson_pharmaceutical_principles()
            elif choice == "7":
                self.final_assessment()
            elif choice == "8":
                self.quick_reference()
            elif choice == "9":
                print("\n✅ Training complete! Your score: {}/{} ({:.1f}%)".format(
                    self.score, self.total_questions,
                    100 * self.score / max(self.total_questions, 1)
                ))
                break
            else:
                print("❌ Invalid choice. Please select 1-9.")

    def lesson_functional_groups(self):
        """Module 1: Functional Groups"""
        print("\n" + "="*80)
        print("MODULE 1: FUNCTIONAL GROUPS - The Building Blocks of Drugs")
        print("="*80)

        print("\n📖 Every drug is made of functional groups that determine its properties:")
        print("   • REACTIVITY - How the molecule reacts")
        print("   • SOLUBILITY - Water-soluble vs fat-soluble")
        print("   • BIOAVAILABILITY - How well absorbed")
        print("   • METABOLISM - How the body breaks it down")
        print("   • TOXICITY - Some groups are toxic, others safe")

        print("\n🔍 Let's explore the 9 most important functional groups:\n")

        for i, (key, fg) in enumerate(self.functional_groups.items(), 1):
            print(f"{i}. {fg.name} ({fg.formula})")
            print(f"   Reactivity: {fg.reactivity}")
            print(f"   Key reactions: {', '.join(fg.common_reactions[:2])}")
            print(f"   Found in: {', '.join(fg.examples[:2])}")
            print()

        # Quiz
        print("\n📝 QUIZ: Functional Group Recognition\n")

        questions = [
            ("Which functional group makes a molecule acidic?", "carboxylic_acid", "Carboxylic acid (-COOH)"),
            ("Which functional group is basic and nucleophilic?", "amine", "Amine (-NH2)"),
            ("What functional group is in aspirin?", "ester", "Ester (acetyl group)"),
            ("Which group is present in 70%+ of all drugs?", "aromatic", "Aromatic rings (benzene)"),
        ]

        for question, correct_key, correct_name in questions:
            print(f"❓ {question}")
            user_answer = input("Your answer: ").strip().lower()
            self.total_questions += 1

            if correct_key in user_answer or correct_name.lower() in user_answer:
                print("✅ Correct!\n")
                self.score += 1
            else:
                print(f"❌ Incorrect. Answer: {correct_name}\n")

        self.completed_modules.append("functional_groups")
        print(f"✅ Module 1 complete! Score: {self.score}/{self.total_questions}")

    def lesson_reactions(self):
        """Module 2: Chemical Reactions"""
        print("\n" + "="*80)
        print("MODULE 2: CHEMICAL REACTIONS - How Molecules Transform")
        print("="*80)

        print("\n📖 Drug synthesis requires mastering key reaction types:")
        print("   • Substitution (swap one group for another)")
        print("   • Addition (add groups to double bonds)")
        print("   • Elimination (remove groups to form double bonds)")
        print("   • Oxidation/Reduction (change oxidation state)")
        print("   • Coupling (join two molecules together)")

        print("\n🔍 Essential Reactions for Drug Synthesis:\n")

        for i, (key, rxn) in enumerate(self.reactions.items(), 1):
            print(f"{i}. {rxn.name}")
            print(f"   Reactants: {' + '.join(rxn.reactants)}")
            print(f"   Products: {' + '.join(rxn.products)}")
            print(f"   Conditions: {rxn.conditions}")
            print(f"   Example: {rxn.example_use}")
            print()

        # Quiz
        print("\n📝 QUIZ: Reaction Knowledge\n")

        questions = [
            ("What reaction makes aspirin from salicylic acid?", ["esterification", "fischer"], "Fischer esterification"),
            ("What reaction joins two aromatic rings using palladium?", ["suzuki", "cross-coupling"], "Suzuki cross-coupling"),
            ("What reagent reduces ketones to alcohols (mild)?", ["nabh4", "sodium borohydride"], "NaBH4 (sodium borohydride)"),
            ("What reaction forms peptide bonds?", ["amide", "peptide coupling"], "Amide bond formation"),
        ]

        for question, correct_keywords, correct_answer in questions:
            print(f"❓ {question}")
            user_answer = input("Your answer: ").strip().lower()
            self.total_questions += 1

            if any(kw in user_answer for kw in correct_keywords):
                print("✅ Correct!\n")
                self.score += 1
            else:
                print(f"❌ Incorrect. Answer: {correct_answer}\n")

        self.completed_modules.append("reactions")
        print(f"✅ Module 2 complete! Score: {self.score}/{self.total_questions}")

    def lesson_retrosynthesis(self):
        """Module 3: Retrosynthetic Analysis"""
        print("\n" + "="*80)
        print("MODULE 3: RETROSYNTHETIC ANALYSIS - Planning Drug Synthesis")
        print("="*80)

        print("\n📖 RETROSYNTHESIS: The Art of Working Backwards")
        print("\nInstead of asking 'What can I make from X?', ask 'How do I make Y?'")
        print("\n🎯 The Retrosynthetic Process:")
        print("   1. START with the target molecule (drug)")
        print("   2. IDENTIFY key bonds to break (disconnections)")
        print("   3. WORK BACKWARDS to simpler precursors")
        print("   4. REPEAT until you reach commercially available materials")
        print("   5. REVERSE the plan to get forward synthesis")

        print("\n🔍 Example: Aspirin Retrosynthesis")
        print("\n   TARGET: Aspirin (acetylsalicylic acid)")
        print("   ↓ (disconnect ester bond)")
        print("   PRECURSORS: Salicylic acid + Acetic anhydride")
        print("   ↓ (salicylic acid from...)")
        print("   STARTING MATERIALS: Phenol + CO2")

        print("\n💡 Key Retrosynthetic Strategies:")
        print("   • Functional Group Interconversions (FGI)")
        print("     - Alcohol ↔ Ketone ↔ Alkene")
        print("     - Amine ↔ Nitro ↔ Nitrile")
        print("   • Carbon-Carbon Bond Disconnections")
        print("     - Identify which C-C bonds to break")
        print("     - Use Grignard, aldol, Diels-Alder to reconnect")
        print("   • Protecting Groups")
        print("     - Temporarily block reactive groups")
        print("     - Allows selective reactions elsewhere")

        print("\n🧩 Retrosynthetic Symbols:")
        print("   ⇒ means 'can be made from' (retro arrow)")
        print("   FGI = Functional Group Interconversion")
        print("   TM = Target Molecule")

        # Interactive Example
        print("\n📝 INTERACTIVE: Plan a synthesis")
        print("\nTarget: Paracetamol (acetaminophen)")
        print("Structure: Benzene ring with -OH and -NHCOCH3 groups (para position)")

        print("\nWhat would be a good disconnection?")
        print("A) Break the aromatic ring")
        print("B) Disconnect the acetyl group from the amine")
        print("C) Break the C-N bond")

        answer = input("Your choice (A/B/C): ").strip().upper()
        self.total_questions += 1

        if answer == "B":
            print("✅ Correct! This gives para-aminophenol + acetic anhydride")
            self.score += 1
        else:
            print("❌ Incorrect. Best disconnection: acetyl group from amine (B)")

        print("\nNext step: How to make para-aminophenol?")
        print("A) Direct amination of phenol")
        print("B) Nitration of phenol, then reduction")
        print("C) Halogenation, then substitution")

        answer = input("Your choice (A/B/C): ").strip().upper()
        self.total_questions += 1

        if answer == "B":
            print("✅ Correct! Nitrate phenol → para-nitrophenol → reduce → para-aminophenol")
            self.score += 1
        else:
            print("❌ Incorrect. Best route: Nitration then reduction (B)")

        self.completed_modules.append("retrosynthesis")
        print(f"\n✅ Module 3 complete! Score: {self.score}/{self.total_questions}")

    def lesson_drug_syntheses(self):
        """Module 4: Real Drug Synthesis Examples"""
        print("\n" + "="*80)
        print("MODULE 4: DRUG SYNTHESIS EXAMPLES - Real Pathways")
        print("="*80)

        print("\n📖 Let's examine how real drugs are actually made:")
        print("   • ASPIRIN - Simple synthesis (beginner)")
        print("   • IBUPROFEN - Industrial synthesis (intermediate)")
        print("   • MORPHINE - Natural extraction + total synthesis (advanced)")
        print("   • TAXOL - Semi-synthesis from natural precursor (advanced)")

        print("\n🔍 Detailed Synthesis Pathways:\n")

        for i, (key, synth) in enumerate(self.syntheses.items(), 1):
            print(f"{i}. {synth.drug_name}")
            print(f"   Difficulty: {synth.difficulty.upper()}")
            print(f"   Target: {synth.target_structure}")
            print(f"   Typical Yield: {synth.yield_percent}%")
            print(f"\n   Synthesis Steps:")
            for step in synth.synthesis_steps:
                print(f"      {step}")
            print(f"\n   Key Reactions: {', '.join(synth.key_reactions)}")
            print(f"   Starting Materials: {', '.join(synth.starting_materials)}")
            print()

        # Quiz
        print("\n📝 QUIZ: Synthesis Knowledge\n")

        questions = [
            ("What is the simplest drug to synthesize?", ["aspirin", "acetylsalicylic"], "Aspirin"),
            ("What reaction type is used to make the ester bond in aspirin?", ["esterification", "fischer"], "Esterification"),
            ("How is morphine primarily obtained industrially?", ["extraction", "natural", "opium", "poppy"], "Natural extraction from opium poppy"),
            ("Why is total synthesis of morphine not used commercially?", ["expensive", "difficult", "low yield", "complex"], "Too complex, low yield, not economical"),
            ("How is Taxol (paclitaxel) made today?", ["semi-synthesis", "semisynthesis", "needles"], "Semi-synthesis from yew needle precursor"),
        ]

        for question, correct_keywords, correct_answer in questions:
            print(f"❓ {question}")
            user_answer = input("Your answer: ").strip().lower()
            self.total_questions += 1

            if any(kw in user_answer for kw in correct_keywords):
                print("✅ Correct!\n")
                self.score += 1
            else:
                print(f"❌ Incorrect. Answer: {correct_answer}\n")

        self.completed_modules.append("drug_syntheses")
        print(f"✅ Module 4 complete! Score: {self.score}/{self.total_questions}")

    def lesson_natural_products(self):
        """Module 5: Natural Product Chemistry"""
        print("\n" + "="*80)
        print("MODULE 5: NATURAL PRODUCT CHEMISTRY - Extraction vs Synthesis")
        print("="*80)

        print("\n📖 Many drugs come from nature:")
        print("   • 50%+ of drugs derived from natural products")
        print("   • Plants, fungi, bacteria, marine organisms")
        print("   • Often too complex to synthesize economically")

        print("\n🌿 EXTRACTION METHODS:")
        print("\n1. SOLVENT EXTRACTION")
        print("   • Grind plant material")
        print("   • Soak in solvent (water, ethanol, hexane)")
        print("   • Filter and concentrate")
        print("   • Example: Morphine from opium latex")

        print("\n2. STEAM DISTILLATION")
        print("   • Heat plant material with steam")
        print("   • Volatile compounds evaporate")
        print("   • Condense and collect")
        print("   • Example: Essential oils, menthol")

        print("\n3. SUPERCRITICAL CO2 EXTRACTION")
        print("   • Use CO2 above critical point (31°C, 73 atm)")
        print("   • Acts as solvent but leaves no residue")
        print("   • Example: Caffeine from coffee, CBD from hemp")

        print("\n4. FERMENTATION")
        print("   • Grow microorganisms that produce compound")
        print("   • Extract from culture medium")
        print("   • Example: Penicillin, insulin, many antibiotics")

        print("\n⚖️ EXTRACTION vs TOTAL SYNTHESIS:")

        comparison = [
            ("Complexity", "Extraction: Simple process", "Synthesis: Can be 20-50+ steps"),
            ("Yield", "Extraction: Often low (0.01-10%)", "Synthesis: Variable (1-85%)"),
            ("Purity", "Extraction: Requires purification", "Synthesis: High purity possible"),
            ("Cost", "Extraction: Depends on plant availability", "Synthesis: Often very expensive"),
            ("Sustainability", "Extraction: Can deplete resources", "Synthesis: Sustainable if efficient"),
            ("Control", "Extraction: Limited control", "Synthesis: Full control of structure"),
        ]

        print()
        for criterion, extraction, synthesis in comparison:
            print(f"  {criterion}:")
            print(f"    • {extraction}")
            print(f"    • {synthesis}")

        print("\n💡 SEMI-SYNTHESIS: Best of Both Worlds")
        print("   • Extract abundant natural precursor")
        print("   • Chemically modify to final drug")
        print("   • Example: Taxol from 10-deacetylbaccatin III")
        print("   • Advantage: Simpler than total synthesis, higher yield than extraction")

        print("\n🧬 BIOSYNTHESIS: The Future")
        print("   • Engineer bacteria/yeast to produce drug")
        print("   • Insert genes for biosynthetic pathway")
        print("   • Example: Artemisinin (antimalarial) from engineered yeast")
        print("   • Advantage: Scalable, sustainable, no plant harvesting")

        # Quiz
        print("\n📝 QUIZ: Natural Products\n")

        questions = [
            ("What percentage of drugs are derived from natural products?", ["50", "half"], "~50% or more"),
            ("What extraction method uses CO2 above its critical point?", ["supercritical", "scco2"], "Supercritical CO2 extraction"),
            ("How is penicillin produced industrially?", ["fermentation", "culture", "microorganism"], "Fermentation of Penicillium fungus"),
            ("What is semi-synthesis?", ["natural precursor", "extract then modify", "partial"], "Extract natural precursor, then chemically modify"),
        ]

        for question, correct_keywords, correct_answer in questions:
            print(f"❓ {question}")
            user_answer = input("Your answer: ").strip().lower()
            self.total_questions += 1

            if any(kw in user_answer for kw in correct_keywords):
                print("✅ Correct!\n")
                self.score += 1
            else:
                print(f"❌ Incorrect. Answer: {correct_answer}\n")

        self.completed_modules.append("natural_products")
        print(f"✅ Module 5 complete! Score: {self.score}/{self.total_questions}")

    def lesson_pharmaceutical_principles(self):
        """Module 6: Pharmaceutical Chemistry Principles"""
        print("\n" + "="*80)
        print("MODULE 6: PHARMACEUTICAL CHEMISTRY PRINCIPLES")
        print("="*80)

        print("\n📖 Making a drug requires more than just chemistry:")

        print("\n1. DRUG DESIGN CONSIDERATIONS")
        print("   • Target selectivity (binds to disease protein, not others)")
        print("   • ADME properties (Absorption, Distribution, Metabolism, Excretion)")
        print("   • Toxicity (minimize off-target effects)")
        print("   • Patent landscape (avoid existing patents)")

        print("\n2. LIPINSKI'S RULE OF FIVE")
        print("   Drug-likeness criteria for oral bioavailability:")
        print("   • Molecular weight < 500 Da")
        print("   • LogP < 5 (lipophilicity)")
        print("   • H-bond donors ≤ 5 (-OH, -NH groups)")
        print("   • H-bond acceptors ≤ 10 (O, N atoms)")
        print("   • Rotatable bonds < 10 (flexibility)")

        print("\n3. STRUCTURE-ACTIVITY RELATIONSHIP (SAR)")
        print("   • Change structure slightly, measure activity")
        print("   • Identify essential functional groups")
        print("   • Optimize potency and selectivity")
        print("   • Example: Modify aspirin → ibuprofen, naproxen")

        print("\n4. PRODRUG STRATEGY")
        print("   • Inactive precursor → activated in body")
        print("   • Improves bioavailability or targets specific tissue")
        print("   • Examples:")
        print("     - Capecitabine → 5-FU (activated in tumor)")
        print("     - Enalapril → Enalaprilat (ACE inhibitor)")
        print("     - Heroin → Morphine (crosses blood-brain barrier faster)")

        print("\n5. BIOISOSTERES")
        print("   • Replace functional group with similar one")
        print("   • Maintains activity, changes properties")
        print("   • Examples:")
        print("     - Carboxylic acid ↔ Sulfonamide (isosteric)")
        print("     - Benzene ↔ Thiophene (aromatic bioisostere)")
        print("     - Ester ↔ Amide (metabolic stability)")

        print("\n6. CHIRALITY & STEREOCHEMISTRY")
        print("   • Many drugs are chiral (non-superimposable mirror images)")
        print("   • Enantiomers can have different activity:")
        print("     - R-enantiomer: Active")
        print("     - S-enantiomer: Inactive or toxic")
        print("   • Examples:")
        print("     - Thalidomide (R: sedative, S: teratogenic)")
        print("     - Ibuprofen (S: active, R: inactive)")
        print("     - Ethambutol (S,S: antibiotic, R,R: causes blindness)")

        print("\n7. PHARMACEUTICAL FORMULATION")
        print("   • Pure drug ≠ final medicine")
        print("   • Need to consider:")
        print("     - Salt form (HCl, Na+, etc. for solubility)")
        print("     - Crystal polymorph (affects dissolution)")
        print("     - Excipients (binders, fillers, coatings)")
        print("     - Delivery method (pill, injection, patch, etc.)")

        print("\n8. REGULATORY REQUIREMENTS")
        print("   • Preclinical testing (cell culture, animal models)")
        print("   • Phase I trials (safety, 20-100 healthy volunteers)")
        print("   • Phase II trials (efficacy, 100-300 patients)")
        print("   • Phase III trials (large-scale, 1000-3000 patients)")
        print("   • FDA approval process: 10-15 years, $1-2 billion")

        # Quiz
        print("\n📝 QUIZ: Pharmaceutical Principles\n")

        questions = [
            ("What is Lipinski's Rule of Five cutoff for molecular weight?", ["500"], "< 500 Da"),
            ("What is a prodrug?", ["inactive", "activated", "precursor"], "Inactive precursor activated in the body"),
            ("What are bioisosteres?", ["similar", "replace", "functional group"], "Replacement functional groups with similar properties"),
            ("Why is chirality important in drugs?", ["enantiomers", "different activity", "mirror"], "Enantiomers can have different activity or toxicity"),
        ]

        for question, correct_keywords, correct_answer in questions:
            print(f"❓ {question}")
            user_answer = input("Your answer: ").strip().lower()
            self.total_questions += 1

            if any(kw in user_answer for kw in correct_keywords):
                print("✅ Correct!\n")
                self.score += 1
            else:
                print(f"❌ Incorrect. Answer: {correct_answer}\n")

        self.completed_modules.append("pharmaceutical_principles")
        print(f"✅ Module 6 complete! Score: {self.score}/{self.total_questions}")

    def final_assessment(self):
        """Comprehensive final exam"""
        print("\n" + "="*80)
        print("FINAL ASSESSMENT - Drug Synthesis & Pharmaceutical Chemistry")
        print("="*80)

        if len(self.completed_modules) < 3:
            print("\n⚠️  Warning: You should complete at least modules 1-3 before the final exam.")
            proceed = input("Continue anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                return

        print("\n📝 10 comprehensive questions covering all modules:\n")

        questions = [
            ("What functional group is both acidic and can form esters?", ["carboxylic acid", "cooh"], "Carboxylic acid (-COOH)"),
            ("Name the reaction that couples two aromatic rings using Pd catalyst.", ["suzuki"], "Suzuki cross-coupling"),
            ("What does the retrosynthetic arrow (⇒) mean?", ["can be made from", "made from"], "'Can be made from'"),
            ("What is the first step in aspirin synthesis from salicylic acid?", ["esterification", "acetylation", "acetic anhydride"], "Esterification with acetic anhydride"),
            ("How is morphine primarily obtained?", ["extraction", "natural", "opium"], "Natural extraction from opium poppy"),
            ("What extraction method uses microorganisms to produce drugs?", ["fermentation"], "Fermentation"),
            ("What does ADME stand for?", ["absorption distribution metabolism excretion"], "Absorption, Distribution, Metabolism, Excretion"),
            ("What is Lipinski's Rule of Five used for?", ["drug-likeness", "bioavailability", "oral"], "Predicting oral bioavailability"),
            ("What is a prodrug?", ["inactive precursor", "activated"], "Inactive precursor activated in the body"),
            ("Why might two enantiomers of a drug have different effects?", ["chirality", "stereochemistry", "3d structure"], "Different 3D shape interacts differently with chiral biological targets"),
        ]

        exam_score = 0
        for i, (question, correct_keywords, correct_answer) in enumerate(questions, 1):
            print(f"{i}. {question}")
            user_answer = input("Your answer: ").strip().lower()
            self.total_questions += 1

            if any(kw in user_answer for kw in correct_keywords):
                print("✅ Correct!\n")
                self.score += 1
                exam_score += 1
            else:
                print(f"❌ Incorrect. Answer: {correct_answer}\n")

        # Final grade
        exam_percent = 100 * exam_score / len(questions)
        print("\n" + "="*80)
        print(f"FINAL EXAM SCORE: {exam_score}/{len(questions)} ({exam_percent:.1f}%)")
        print("="*80)

        if exam_percent >= 90:
            grade = "A - Excellent! You have mastery of drug synthesis concepts."
        elif exam_percent >= 80:
            grade = "B - Very good! You understand most key concepts."
        elif exam_percent >= 70:
            grade = "C - Good! You have a solid foundation."
        elif exam_percent >= 60:
            grade = "D - Passing, but review the material."
        else:
            grade = "F - Please review the modules and retake the exam."

        print(f"\nGrade: {grade}")
        print(f"\nOverall course score: {self.score}/{self.total_questions} ({100*self.score/max(self.total_questions,1):.1f}%)")

    def quick_reference(self):
        """Quick reference guide"""
        print("\n" + "="*80)
        print("QUICK REFERENCE GUIDE - Drug Synthesis Cheat Sheet")
        print("="*80)

        print("\n🔍 FUNCTIONAL GROUPS (9 KEY GROUPS):")
        for key, fg in self.functional_groups.items():
            print(f"  • {fg.name}: {fg.formula}")

        print("\n⚗️  ESSENTIAL REACTIONS (7 CORE REACTIONS):")
        for key, rxn in self.reactions.items():
            print(f"  • {rxn.name}: {' + '.join(rxn.reactants)} → {' + '.join(rxn.products)}")

        print("\n🎯 RETROSYNTHESIS STEPS:")
        print("  1. Identify target molecule")
        print("  2. Find key disconnections")
        print("  3. Work backwards to precursors")
        print("  4. Repeat until commercial materials")
        print("  5. Reverse for forward synthesis")

        print("\n💊 DRUG SYNTHESIS DIFFICULTY:")
        print("  • BEGINNER: Aspirin (1 step), Paracetamol (3 steps)")
        print("  • INTERMEDIATE: Ibuprofen (6 steps), Penicillin (fermentation + semi-synthesis)")
        print("  • ADVANCED: Morphine (22+ steps total synthesis), Taxol (semi-synthesis from natural)")

        print("\n🌿 EXTRACTION METHODS:")
        print("  1. Solvent extraction (most common)")
        print("  2. Steam distillation (volatile compounds)")
        print("  3. Supercritical CO2 (clean, no residue)")
        print("  4. Fermentation (biosynthesis)")

        print("\n📏 LIPINSKI'S RULE OF FIVE:")
        print("  • MW < 500 Da")
        print("  • LogP < 5")
        print("  • H-bond donors ≤ 5")
        print("  • H-bond acceptors ≤ 10")

        print("\n💡 KEY PHARMACEUTICAL CONCEPTS:")
        print("  • SAR: Structure-Activity Relationship")
        print("  • ADME: Absorption, Distribution, Metabolism, Excretion")
        print("  • Prodrug: Inactive → Active in body")
        print("  • Bioisostere: Replacement group with similar properties")
        print("  • Chirality: Mirror image molecules (enantiomers)")

        print("\n📊 DRUG DEVELOPMENT TIMELINE:")
        print("  • Discovery: 2-5 years")
        print("  • Preclinical: 1-2 years")
        print("  • Phase I: 1-2 years (safety)")
        print("  • Phase II: 2-3 years (efficacy)")
        print("  • Phase III: 2-4 years (large trials)")
        print("  • FDA Review: 1-2 years")
        print("  • TOTAL: 10-15 years, $1-2 billion")

        print("\n" + "="*80)


def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Drug Synthesis Training Module")
        print("\nUsage:")
        print("  python drug_synthesis_training.py         # Interactive training")
        print("  python drug_synthesis_training.py --help  # Show this help")
        return

    trainer = DrugSynthesisTrainer()
    trainer.start_course()


if __name__ == "__main__":
    main()
