#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Drug Discovery Assistant - Help users find drugs they don't know about

This tool helps discover compounds based on:
- Cancer type
- Treatment goals
- Mechanisms of action
- Side effect profiles
- Alternative/natural preferences
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from oncology_lab.drug_response import DRUG_DATABASE, DrugClass
from oncology_lab import TumorType, CancerStage
from typing import List, Dict, Set


class DrugDiscoveryAssistant:
    """Interactive assistant to help discover relevant drugs"""

    def __init__(self):
        self.db = DRUG_DATABASE

    def discover_by_cancer_type(self, tumor_type: str) -> List[Dict]:
        """Find drugs commonly used for specific cancer type"""
        results = []

        # Map tumor types to common indications
        tumor_mappings = {
            'breast': ['Breast cancer', 'breast', 'HER2+', 'ER+'],
            'lung': ['NSCLC', 'SCLC', 'Lung cancer', 'lung'],
            'colorectal': ['Colorectal cancer', 'colorectal', 'Gastric cancer'],
            'prostate': ['Prostate cancer', 'prostate', 'castration-resistant'],
            'melanoma': ['Melanoma', 'melanoma', 'BRAF'],
            'pancreatic': ['Pancreatic cancer', 'pancreatic'],
            'glioblastoma': ['Glioblastoma', 'glioblastoma', 'Anaplastic astrocytoma'],
            'ovarian': ['Ovarian cancer', 'ovarian'],
            'leukemia': ['Leukemia', 'AML', 'ALL', 'CML'],
            'lymphoma': ['Lymphoma', 'Hodgkin', 'lymphoma'],
        }

        search_terms = tumor_mappings.get(tumor_type.lower(), [tumor_type])

        for drug_name, drug in self.db.items():
            # Check approved indications
            for indication in drug.approved_indications:
                if any(term.lower() in indication.lower() for term in search_terms):
                    results.append({
                        'name': drug.name,
                        'drug_name': drug_name,
                        'class': drug.drug_class.value,
                        'indication': indication,
                        'fda_approved': drug.fda_approved,
                        'mechanism': drug.mechanism_of_action,
                        'ic50': drug.ic50,
                    })
                    break

        return sorted(results, key=lambda x: (not x['fda_approved'], x['ic50']))

    def discover_by_mechanism(self, mechanism: str) -> List[Dict]:
        """Find drugs with specific mechanism of action"""
        results = []

        for drug_name, drug in self.db.items():
            if mechanism.lower() in drug.mechanism_of_action.lower():
                results.append({
                    'name': drug.name,
                    'drug_name': drug_name,
                    'class': drug.drug_class.value,
                    'mechanism': drug.mechanism_of_action,
                    'targets': drug.target_proteins,
                    'ic50': drug.ic50,
                })

        return sorted(results, key=lambda x: x['ic50'])

    def discover_natural_alternatives(self) -> List[Dict]:
        """Find all natural/supplement compounds"""
        natural_compounds = [
            'vitamin_d3', 'vitamin_c', 'curcumin', 'quercetin', 'resveratrol',
            'egcg', 'artemisinin', 'berberine', 'cbd', 'melatonin',
            'omega3_dha', 'sulforaphane'
        ]

        results = []
        for drug_name in natural_compounds:
            if drug_name in self.db:
                drug = self.db[drug_name]
                results.append({
                    'name': drug.name,
                    'drug_name': drug_name,
                    'mechanism': drug.mechanism_of_action,
                    'source': self._get_natural_source(drug_name),
                    'dose': drug.standard_dose_mg,
                    'route': drug.route,
                })

        return results

    def discover_repurposed_drugs(self) -> List[Dict]:
        """Find FDA-approved drugs used off-label for cancer"""
        repurposed = [
            'metformin', 'ivermectin', 'mebendazole', 'hydroxychloroquine',
            'aspirin', 'dichloroacetate', 'fenbendazole'
        ]

        results = []
        for drug_name in repurposed:
            if drug_name in self.db:
                drug = self.db[drug_name]
                results.append({
                    'name': drug.name,
                    'drug_name': drug_name,
                    'original_use': drug.approved_indications[0] if drug.approved_indications else 'Experimental',
                    'cancer_mechanism': drug.mechanism_of_action,
                    'fda_approved': drug.fda_approved,
                    'approval_year': drug.approval_year,
                })

        return results

    def discover_by_target(self, target: str) -> List[Dict]:
        """Find drugs targeting specific protein/pathway"""
        results = []

        for drug_name, drug in self.db.items():
            # Check target proteins
            if any(target.upper() in protein.upper() for protein in drug.target_proteins):
                results.append({
                    'name': drug.name,
                    'drug_name': drug_name,
                    'class': drug.drug_class.value,
                    'targets': drug.target_proteins,
                    'mechanism': drug.mechanism_of_action,
                    'ic50': drug.ic50,
                })

        return sorted(results, key=lambda x: x['ic50'])

    def discover_low_toxicity(self) -> List[Dict]:
        """Find drugs with minimal side effects"""
        results = []

        for drug_name, drug in self.db.items():
            # Check toxicity scores
            toxicity_score = sum([
                getattr(drug, 'myelosuppression', 0),
                getattr(drug, 'neurotoxicity', 0),
                getattr(drug, 'cardiotoxicity', 0),
                getattr(drug, 'hepatotoxicity', 0),
            ])

            if toxicity_score < 0.5:  # Low toxicity threshold
                results.append({
                    'name': drug.name,
                    'drug_name': drug_name,
                    'class': drug.drug_class.value,
                    'toxicity_score': toxicity_score,
                    'mechanism': drug.mechanism_of_action,
                })

        return sorted(results, key=lambda x: x['toxicity_score'])

    def discover_by_class(self, drug_class: str) -> List[Dict]:
        """Find all drugs in a specific class"""
        results = []

        for drug_name, drug in self.db.items():
            if drug_class.lower() in drug.drug_class.value.lower():
                results.append({
                    'name': drug.name,
                    'drug_name': drug_name,
                    'mechanism': drug.mechanism_of_action,
                    'ic50': drug.ic50,
                    'approval_year': drug.approval_year,
                    'fda_approved': drug.fda_approved,
                })

        return sorted(results, key=lambda x: x['approval_year'] if x['approval_year'] else 9999)

    def discover_novel_combinations(self, current_drugs: List[str]) -> Dict[str, List[str]]:
        """Suggest complementary drugs based on what's already being used"""
        suggestions = {
            'synergistic': [],
            'different_mechanism': [],
            'metabolic_support': [],
        }

        current_mechanisms = set()
        current_classes = set()

        # Analyze current drugs
        for drug_name in current_drugs:
            if drug_name in self.db:
                drug = self.db[drug_name]
                current_mechanisms.add(drug.mechanism_of_action.split(',')[0])
                current_classes.add(drug.drug_class.value)

        # Find complementary drugs
        for drug_name, drug in self.db.items():
            if drug_name in current_drugs:
                continue

            mechanism_first = drug.mechanism_of_action.split(',')[0]

            # Different mechanism (complementary)
            if mechanism_first not in current_mechanisms:
                suggestions['different_mechanism'].append(drug.name)

            # Metabolic support (always useful)
            if drug.drug_class == DrugClass.METABOLIC_INHIBITOR:
                suggestions['metabolic_support'].append(drug.name)

            # Immunotherapy (synergistic with most things)
            if drug.drug_class == DrugClass.IMMUNOTHERAPY:
                suggestions['synergistic'].append(drug.name)

        # Limit to top suggestions
        for key in suggestions:
            suggestions[key] = suggestions[key][:5]

        return suggestions

    def _get_natural_source(self, drug_name: str) -> str:
        """Get natural source description"""
        sources = {
            'vitamin_d3': 'Sunlight, fish oil, fortified foods',
            'vitamin_c': 'Citrus fruits, vegetables',
            'curcumin': 'Turmeric root',
            'quercetin': 'Onions, apples, berries',
            'resveratrol': 'Grapes, red wine, berries',
            'egcg': 'Green tea',
            'artemisinin': 'Sweet wormwood (Artemisia annua)',
            'berberine': 'Goldenseal, barberry',
            'cbd': 'Cannabis/hemp',
            'melatonin': 'Pineal gland (endogenous)',
            'omega3_dha': 'Fish oil, algae',
            'sulforaphane': 'Broccoli sprouts, cruciferous vegetables',
        }
        return sources.get(drug_name, 'Various sources')

    def interactive_discovery(self):
        """Interactive mode for drug discovery"""
        print("\n" + "=" * 80)
        print("  🔍 Drug Discovery Assistant")
        print("=" * 80)
        print("\nHelp you find compounds you might not know about!\n")

        print("Discovery Options:")
        print("  1. By cancer type")
        print("  2. By mechanism of action")
        print("  3. Natural compounds only")
        print("  4. Repurposed drugs")
        print("  5. By molecular target")
        print("  6. Low toxicity options")
        print("  7. By drug class")
        print("  8. Browse all drugs by category")
        print("  9. Suggest combinations")
        print("  0. Exit")

        while True:
            choice = input("\nSelect option (0-9): ").strip()

            if choice == '0':
                print("\nGoodbye!")
                break

            elif choice == '1':
                print("\nCancer types: breast, lung, colorectal, prostate, melanoma,")
                print("              pancreatic, glioblastoma, ovarian, leukemia, lymphoma")
                cancer = input("Enter cancer type: ").strip()
                results = self.discover_by_cancer_type(cancer)
                self._print_cancer_results(results)

            elif choice == '2':
                print("\nMechanism keywords: DNA, kinase, checkpoint, microtubule,")
                print("                    AMPK, apoptosis, angiogenic, etc.")
                mechanism = input("Enter mechanism keyword: ").strip()
                results = self.discover_by_mechanism(mechanism)
                self._print_mechanism_results(results)

            elif choice == '3':
                results = self.discover_natural_alternatives()
                self._print_natural_results(results)

            elif choice == '4':
                results = self.discover_repurposed_drugs()
                self._print_repurposed_results(results)

            elif choice == '5':
                print("\nTargets: EGFR, BRAF, ALK, PD-1, VEGF, ER, AR, etc.")
                target = input("Enter target: ").strip()
                results = self.discover_by_target(target)
                self._print_target_results(results)

            elif choice == '6':
                results = self.discover_low_toxicity()
                self._print_low_tox_results(results)

            elif choice == '7':
                print("\nClasses: chemotherapy, targeted_therapy, immunotherapy,")
                print("         hormone_therapy, metabolic_inhibitor")
                drug_class = input("Enter class: ").strip()
                results = self.discover_by_class(drug_class)
                self._print_class_results(results)

            elif choice == '8':
                self._browse_all_drugs()

            elif choice == '9':
                print("\nEnter drugs you're currently using (comma-separated):")
                current = input("Drugs: ").strip().split(',')
                current = [d.strip() for d in current if d.strip()]
                suggestions = self.discover_novel_combinations(current)
                self._print_combination_suggestions(suggestions)

            else:
                print("Invalid choice. Try again.")

    def _print_cancer_results(self, results):
        print(f"\n✓ Found {len(results)} drugs:\n")
        for r in results:
            fda = "✓ FDA" if r['fda_approved'] else "✗ Experimental"
            print(f"  • {r['name']:25s} {fda:15s} {r['class']}")
            print(f"    Indication: {r['indication']}")
            print(f"    Mechanism: {r['mechanism'][:60]}...")
            print()

    def _print_mechanism_results(self, results):
        print(f"\n✓ Found {len(results)} drugs:\n")
        for r in results:
            print(f"  • {r['name']:25s} ({r['class']})")
            print(f"    {r['mechanism']}")
            print(f"    Targets: {', '.join(r['targets'][:3])}")
            print()

    def _print_natural_results(self, results):
        print(f"\n✓ Found {len(results)} natural compounds:\n")
        for r in results:
            print(f"  • {r['name']:25s}")
            print(f"    Source: {r['source']}")
            print(f"    Mechanism: {r['mechanism'][:60]}...")
            print(f"    Dose: {r['dose']} mg {r['route']}")
            print()

    def _print_repurposed_results(self, results):
        print(f"\n✓ Found {len(results)} repurposed drugs:\n")
        for r in results:
            fda = "✓ FDA" if r['fda_approved'] else "✗ Experimental"
            print(f"  • {r['name']:25s} {fda}")
            print(f"    Original use: {r['original_use']}")
            print(f"    Cancer mechanism: {r['cancer_mechanism'][:60]}...")
            print()

    def _print_target_results(self, results):
        print(f"\n✓ Found {len(results)} drugs:\n")
        for r in results:
            print(f"  • {r['name']:25s} ({r['class']})")
            print(f"    Targets: {', '.join(r['targets'])}")
            print(f"    IC50: {r['ic50']} µM")
            print()

    def _print_low_tox_results(self, results):
        print(f"\n✓ Found {len(results)} low-toxicity drugs:\n")
        for r in results:
            print(f"  • {r['name']:25s} (Toxicity: {r['toxicity_score']:.2f})")
            print(f"    Class: {r['class']}")
            print(f"    Mechanism: {r['mechanism'][:60]}...")
            print()

    def _print_class_results(self, results):
        print(f"\n✓ Found {len(results)} drugs:\n")
        for r in results:
            year = r['approval_year'] if r['approval_year'] else 'N/A'
            fda = "✓" if r['fda_approved'] else "✗"
            print(f"  • {r['name']:25s} ({year}, {fda})")
            print(f"    {r['mechanism'][:60]}...")
            print()

    def _browse_all_drugs(self):
        print("\n" + "=" * 80)
        print("ALL DRUGS BY CATEGORY")
        print("=" * 80)

        by_class = {}
        for drug_name, drug in self.db.items():
            cls = drug.drug_class.value
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(drug.name)

        for cls in sorted(by_class.keys()):
            drugs = by_class[cls]
            print(f"\n{cls.upper()} ({len(drugs)} drugs):")
            for drug in sorted(drugs):
                print(f"  • {drug}")

    def _print_combination_suggestions(self, suggestions):
        print("\n" + "=" * 80)
        print("SUGGESTED COMBINATIONS")
        print("=" * 80)

        if suggestions['synergistic']:
            print("\n✓ Synergistic (combine well with most drugs):")
            for drug in suggestions['synergistic']:
                print(f"  • {drug}")

        if suggestions['different_mechanism']:
            print("\n✓ Different mechanisms (complementary):")
            for drug in suggestions['different_mechanism'][:5]:
                print(f"  • {drug}")

        if suggestions['metabolic_support']:
            print("\n✓ Metabolic support (enhance other drugs):")
            for drug in suggestions['metabolic_support'][:5]:
                print(f"  • {drug}")


def main():
    """Main entry point"""
    assistant = DrugDiscoveryAssistant()

    if len(sys.argv) > 1:
        # Command-line mode
        command = sys.argv[1].lower()

        if command == 'cancer' and len(sys.argv) > 2:
            results = assistant.discover_by_cancer_type(sys.argv[2])
            assistant._print_cancer_results(results)

        elif command == 'natural':
            results = assistant.discover_natural_alternatives()
            assistant._print_natural_results(results)

        elif command == 'repurposed':
            results = assistant.discover_repurposed_drugs()
            assistant._print_repurposed_results(results)

        elif command == 'low-tox':
            results = assistant.discover_low_toxicity()
            assistant._print_low_tox_results(results)

        elif command == 'all':
            assistant._browse_all_drugs()

        else:
            print("Usage:")
            print("  python drug_discovery_assistant.py                  # Interactive mode")
            print("  python drug_discovery_assistant.py cancer <type>    # By cancer type")
            print("  python drug_discovery_assistant.py natural          # Natural compounds")
            print("  python drug_discovery_assistant.py repurposed       # Repurposed drugs")
            print("  python drug_discovery_assistant.py low-tox          # Low toxicity")
            print("  python drug_discovery_assistant.py all              # Browse all")

    else:
        # Interactive mode
        assistant.interactive_discovery()


if __name__ == "__main__":
    main()
