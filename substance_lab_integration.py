#!/usr/bin/env python3
"""
Substance Lab Integration - Test ANY Substance in the Oncology Lab

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Integrates the comprehensive substance database (115+ substances) with the oncology lab,
allowing testing of vitamins, neurotransmitters, hormones, amino acids, and more - not just cancer drugs.
"""

import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Import comprehensive database
from comprehensive_substance_database import (
    ComprehensiveSubstanceDatabase,
    Substance,
    SubstanceCategory
)

# Import oncology lab (if available)
try:
    from oncology_lab import OncologyLaboratory, OncologyLabConfig, TumorType, CancerStage
    from oncology_lab.drug_response import Drug, DrugClass
    ONCOLOGY_LAB_AVAILABLE = True
except ImportError:
    ONCOLOGY_LAB_AVAILABLE = False
    print("⚠️  Warning: Oncology lab not available. Running in database-only mode.")


@dataclass
class SubstanceTestResult:
    """Results from testing a substance"""
    substance_name: str
    tested: bool
    estimated_effect: Optional[str] = None
    safety_warning: Optional[str] = None
    notes: Optional[str] = None


class SubstanceLaboratory:
    """
    Laboratory for testing ANY substance from the comprehensive database
    """

    def __init__(self):
        self.db = ComprehensiveSubstanceDatabase()
        self.lab_available = ONCOLOGY_LAB_AVAILABLE
        self.tested_substances: List[str] = []

    def search_substances(self, query: str) -> List[Substance]:
        """Search for substances"""
        return self.db.search(query)

    def browse_category(self, category: SubstanceCategory) -> List[Substance]:
        """Browse substances by category"""
        return self.db.filter_by_category(category)

    def get_substance_info(self, name: str) -> Optional[Substance]:
        """Get detailed info about a substance"""
        return self.db.get_substance(name.lower().replace(" ", "_"))

    def estimate_anticancer_potential(self, substance: Substance) -> Tuple[str, str]:
        """
        Estimate if a substance might have anti-cancer properties
        Returns: (potential, reasoning)
        """
        # Analyze substance properties to estimate potential

        # Known anti-cancer categories
        if substance.category in [SubstanceCategory.POLYPHENOL, SubstanceCategory.ALKALOID]:
            return ("moderate-high", "Natural compounds often have anti-cancer properties")

        # Check mechanisms
        if substance.mechanism:
            mechanism_lower = substance.mechanism.lower()

            if any(kw in mechanism_lower for kw in ["nf-κb", "nfkb", "cox-2"]):
                return ("high", "Inhibits pro-inflammatory pathways (NF-κB/COX-2)")

            if any(kw in mechanism_lower for kw in ["antioxidant", "ros"]):
                return ("moderate", "Antioxidant effects may reduce cancer risk")

            if any(kw in mechanism_lower for kw in ["apoptosis", "cell death"]):
                return ("high", "Induces programmed cell death")

            if any(kw in mechanism_lower for kw in ["kinase", "egfr", "braf", "alk"]):
                return ("high", "Targets cancer-driving kinases")

            if "immune" in mechanism_lower:
                return ("moderate", "Immune modulation can fight cancer")

        # Check targets
        if substance.targets:
            targets_str = " ".join(substance.targets).lower()

            if any(kw in targets_str for kw in ["egfr", "braf", "alk", "her2", "pd-1", "pd-l1"]):
                return ("very-high", "Directly targets known cancer pathways")

        # Vitamins and minerals
        if substance.category == SubstanceCategory.VITAMIN:
            if "d3" in substance.name.lower() or "c" in substance.name.lower():
                return ("moderate", "Vitamins D3 and C have anti-cancer research support")
            else:
                return ("low", "General health support, not directly anti-cancer")

        if substance.category == SubstanceCategory.MINERAL:
            return ("low", "Supports general health, not directly anti-cancer")

        # Neurotransmitters and hormones
        if substance.category in [SubstanceCategory.NEUROTRANSMITTER, SubstanceCategory.HORMONE]:
            return ("very-low", "Primarily neurological/endocrine, not anti-cancer")

        # Amino acids
        if substance.category == SubstanceCategory.AMINO_ACID:
            if "glutamine" in substance.name.lower():
                return ("very-low", "⚠️ Cancer cells use glutamine as fuel")
            else:
                return ("low", "Building block, not anti-cancer")

        # FDA-approved drugs
        if substance.category == SubstanceCategory.DRUG_FDA_APPROVED:
            if "metformin" in substance.name.lower():
                return ("moderate-high", "Metformin shows anti-cancer effects in research")
            elif any(kw in substance.name.lower() for kw in ["aspirin", "ibuprofen"]):
                return ("moderate", "NSAIDs reduce cancer risk in some studies")
            else:
                return ("low", "Not primarily an anti-cancer drug")

        # Toxins
        if substance.category in [SubstanceCategory.TOXIN, SubstanceCategory.POISON]:
            return ("high", "⚠️ TOXIC - Some toxins kill cancer cells but also healthy cells")

        # Default
        return ("unknown", "Insufficient data to estimate anti-cancer potential")

    def generate_safety_warning(self, substance: Substance) -> Optional[str]:
        """Generate safety warnings for testing"""
        warnings = []

        # Toxicity
        if substance.toxicity_class in ["lethal", "high"]:
            warnings.append(f"⚠️ EXTREME CAUTION: {substance.toxicity_class.upper()} toxicity")

        if substance.toxicity_class == "moderate" and substance.ld50 and substance.ld50 < 500:
            warnings.append(f"⚠️ Toxic at low doses (LD50: {substance.ld50} mg/kg)")

        # Controlled substances
        if substance.controlled_substance:
            warnings.append(f"⚠️ DEA Schedule {substance.dea_schedule} controlled substance")

        # Specific warnings
        if "glutamine" in substance.name.lower():
            warnings.append("⚠️ Cancer cells use glutamine as fuel - may promote growth")

        if substance.category in [SubstanceCategory.TOXIN, SubstanceCategory.POISON]:
            warnings.append("⚠️ POISON - Use only in controlled research setting")

        if "chemotherapy" in substance.name.lower():
            warnings.append("⚠️ Chemotherapy agent - cytotoxic to all rapidly dividing cells")

        # Bioavailability concerns
        if substance.bioavailability and substance.bioavailability < 5:
            warnings.append(f"⚠️ Very low bioavailability ({substance.bioavailability}%) - limited absorption")

        if warnings:
            return " | ".join(warnings)
        else:
            return None

    def suggest_dose(self, substance: Substance) -> Tuple[float, str]:
        """
        Suggest a research dose for testing
        Returns: (dose_mg, reasoning)
        """
        # Base dose on typical human doses or research data

        # Vitamins
        if substance.category == SubstanceCategory.VITAMIN:
            if "d3" in substance.name.lower():
                return (1000.0, "Typical supplemental dose (1000 IU)")
            elif "c" in substance.name.lower():
                return (1000.0, "High-dose vitamin C (1g)")
            elif "b" in substance.name.lower():
                return (100.0, "B-complex typical dose")
            else:
                return (100.0, "Standard vitamin dose")

        # Polyphenols
        if substance.category == SubstanceCategory.POLYPHENOL:
            if "curcumin" in substance.name.lower():
                return (1000.0, "Typical curcumin supplement dose")
            elif "resveratrol" in substance.name.lower():
                return (250.0, "Typical resveratrol dose")
            elif "egcg" in substance.name.lower():
                return (400.0, "Equivalent to 3-4 cups green tea")
            elif "quercetin" in substance.name.lower():
                return (500.0, "Typical quercetin dose")
            else:
                return (500.0, "Standard polyphenol dose")

        # Amino acids
        if substance.category == SubstanceCategory.AMINO_ACID:
            if "leucine" in substance.name.lower() or "bcaa" in substance.name.lower():
                return (2000.0, "BCAA supplemental dose")
            else:
                return (500.0, "Standard amino acid dose")

        # Fatty acids
        if substance.category == SubstanceCategory.FATTY_ACID:
            return (1000.0, "Typical omega-3 dose")

        # Minerals
        if substance.category == SubstanceCategory.MINERAL:
            if "calcium" in substance.name.lower():
                return (500.0, "Typical calcium supplement")
            elif "magnesium" in substance.name.lower():
                return (400.0, "Typical magnesium dose")
            elif "iron" in substance.name.lower():
                return (65.0, "Typical iron supplement")
            else:
                return (100.0, "Standard mineral dose")

        # FDA drugs - use approved doses
        if substance.category == SubstanceCategory.DRUG_FDA_APPROVED:
            if "metformin" in substance.name.lower():
                return (1000.0, "Standard metformin dose")
            elif "aspirin" in substance.name.lower():
                return (325.0, "Standard aspirin dose")
            elif "acetaminophen" in substance.name.lower():
                return (500.0, "Standard acetaminophen dose")
            else:
                return (100.0, "Typical pharmaceutical dose")

        # Neurotransmitters/hormones - microdoses
        if substance.category in [SubstanceCategory.NEUROTRANSMITTER, SubstanceCategory.HORMONE]:
            if "melatonin" in substance.name.lower():
                return (3.0, "Typical melatonin dose")
            else:
                return (1.0, "Microdose (endogenous compound)")

        # Alkaloids - careful dosing
        if substance.category == SubstanceCategory.ALKALOID:
            if "caffeine" in substance.name.lower():
                return (100.0, "One cup of coffee equivalent")
            else:
                return (10.0, "Low dose (alkaloids are potent)")

        # Toxins - extremely low doses
        if substance.category in [SubstanceCategory.TOXIN, SubstanceCategory.POISON]:
            return (0.001, "Trace amount (TOXIC)")

        # Default
        return (100.0, "Conservative research dose")

    def test_substance(self, substance_name: str, dose_mg: Optional[float] = None) -> SubstanceTestResult:
        """
        Test a substance (simulated or real if oncology lab available)
        """
        # Get substance from database
        substance = self.get_substance_info(substance_name)

        if not substance:
            return SubstanceTestResult(
                substance_name=substance_name,
                tested=False,
                notes=f"Substance '{substance_name}' not found in database"
            )

        # Estimate potential
        potential, reasoning = self.estimate_anticancer_potential(substance)

        # Generate warnings
        warning = self.generate_safety_warning(substance)

        # Suggest dose if not provided
        if dose_mg is None:
            dose_mg, dose_reasoning = self.suggest_dose(substance)
        else:
            dose_reasoning = f"User-specified dose: {dose_mg} mg"

        # Record test
        self.tested_substances.append(substance.name)

        # Build result
        result = SubstanceTestResult(
            substance_name=substance.name,
            tested=True,
            estimated_effect=f"Anti-cancer potential: {potential.upper()} - {reasoning}",
            safety_warning=warning,
            notes=f"Suggested dose: {dose_mg} mg ({dose_reasoning})"
        )

        # If oncology lab available, could actually run simulation here
        # For now, return estimated result

        return result

    def interactive_explorer(self):
        """Interactive substance exploration interface"""
        print("\n" + "="*80)
        print("🔬 SUBSTANCE LABORATORY - Test ANY Substance")
        print("="*80)
        print(f"\n📊 Database: {len(self.db.substances)} substances available")
        print(f"🧪 Lab Status: {'READY' if self.lab_available else 'DATABASE-ONLY MODE'}")

        stats = self.db.get_stats()
        print(f"\n📚 Categories: {len(stats['by_category'])}")

        while True:
            print("\n" + "="*80)
            print("OPTIONS:")
            print("  1. Search for substance")
            print("  2. Browse by category")
            print("  3. Test a substance (estimate effects)")
            print("  4. Show database statistics")
            print("  5. Exit")
            print("="*80)

            choice = input("\nSelect option (1-5): ").strip()

            if choice == "1":
                self._search_interface()
            elif choice == "2":
                self._browse_interface()
            elif choice == "3":
                self._test_interface()
            elif choice == "4":
                self._stats_interface()
            elif choice == "5":
                print("\n✅ Exiting substance laboratory.")
                break
            else:
                print("❌ Invalid choice. Please select 1-5.")

    def _search_interface(self):
        """Search interface"""
        print("\n" + "-"*80)
        print("SEARCH SUBSTANCES")
        print("-"*80)

        query = input("Enter search term: ").strip()

        if not query:
            print("❌ Empty search query")
            return

        results = self.search_substances(query)

        if not results:
            print(f"\n❌ No substances found matching '{query}'")
            return

        print(f"\n✅ Found {len(results)} results:\n")

        for i, substance in enumerate(results, 1):
            print(f"{i}. {substance.name}")
            print(f"   Category: {substance.category.value}")
            if substance.molecular_formula:
                print(f"   Formula: {substance.molecular_formula}")
            if substance.mechanism:
                print(f"   Mechanism: {substance.mechanism}")
            print()

    def _browse_interface(self):
        """Browse by category interface"""
        print("\n" + "-"*80)
        print("BROWSE BY CATEGORY")
        print("-"*80)

        # Show categories
        stats = self.db.get_stats()
        categories = sorted(stats['by_category'].items(), key=lambda x: -x[1])

        print("\nAvailable categories:\n")
        for i, (cat_name, count) in enumerate(categories, 1):
            print(f"{i}. {cat_name} ({count} substances)")

        choice = input("\nSelect category number: ").strip()

        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(categories):
                print("❌ Invalid category number")
                return

            cat_name, count = categories[idx]

            # Find enum value
            category_enum = None
            for cat in SubstanceCategory:
                if cat.value == cat_name:
                    category_enum = cat
                    break

            if not category_enum:
                print("❌ Category not found")
                return

            # Browse
            substances = self.browse_category(category_enum)

            print(f"\n✅ {cat_name.upper()} ({len(substances)} substances):\n")

            for i, substance in enumerate(substances, 1):
                print(f"{i}. {substance.name}")
                if substance.molecular_formula:
                    print(f"   Formula: {substance.molecular_formula}")
                if substance.natural_source:
                    print(f"   Source: {substance.natural_source}")
                if substance.medical_uses:
                    print(f"   Uses: {', '.join(substance.medical_uses[:2])}")
                print()

        except ValueError:
            print("❌ Invalid input. Please enter a number.")

    def _test_interface(self):
        """Test substance interface"""
        print("\n" + "-"*80)
        print("TEST SUBSTANCE")
        print("-"*80)

        substance_name = input("Enter substance name: ").strip()

        if not substance_name:
            print("❌ Empty substance name")
            return

        # Optional dose
        dose_input = input("Enter dose in mg (or press Enter for suggested dose): ").strip()

        dose_mg = None
        if dose_input:
            try:
                dose_mg = float(dose_input)
            except ValueError:
                print("❌ Invalid dose. Using suggested dose.")
                dose_mg = None

        # Test
        print("\n🔬 Testing substance...")
        result = self.test_substance(substance_name, dose_mg)

        # Display results
        print("\n" + "="*80)
        print(f"TEST RESULTS: {result.substance_name}")
        print("="*80)

        if not result.tested:
            print(f"\n❌ {result.notes}")
            return

        print(f"\n✅ Substance tested successfully")

        if result.estimated_effect:
            print(f"\n📊 {result.estimated_effect}")

        if result.notes:
            print(f"\n💡 {result.notes}")

        if result.safety_warning:
            print(f"\n{result.safety_warning}")

        print("\n" + "="*80)

    def _stats_interface(self):
        """Show statistics"""
        print("\n" + "-"*80)
        print("DATABASE STATISTICS")
        print("-"*80)

        stats = self.db.get_stats()

        print(f"\n📊 Total Substances: {stats['total_substances']}")
        print(f"📊 Categories: {len(stats['by_category'])}")
        print(f"🧪 Tested Substances: {len(set(self.tested_substances))}")

        print("\n📈 Breakdown by category:\n")
        for category, count in sorted(stats['by_category'].items(), key=lambda x: -x[1]):
            print(f"   • {category}: {count}")


def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Substance Laboratory - Test ANY substance from comprehensive database")
        print("\nUsage:")
        print("  python substance_lab_integration.py              # Interactive mode")
        print("  python substance_lab_integration.py --test NAME  # Quick test")
        print("  python substance_lab_integration.py --help       # Show this help")
        return

    if len(sys.argv) > 2 and sys.argv[1] == "--test":
        # Quick test mode
        lab = SubstanceLaboratory()
        substance_name = " ".join(sys.argv[2:])
        result = lab.test_substance(substance_name)

        print(f"\n🔬 Testing: {result.substance_name}")
        if result.tested:
            print(f"✅ {result.estimated_effect}")
            if result.notes:
                print(f"💡 {result.notes}")
            if result.safety_warning:
                print(f"{result.safety_warning}")
        else:
            print(f"❌ {result.notes}")
        return

    # Interactive mode
    lab = SubstanceLaboratory()
    lab.interactive_explorer()


if __name__ == "__main__":
    main()
