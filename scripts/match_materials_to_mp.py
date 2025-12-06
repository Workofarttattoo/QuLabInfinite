#!/usr/bin/env python3
"""
Match Lab Materials to Materials Project IDs

This script matches materials in the lab database with Materials Project entries
by searching for chemical formulas and comparing properties.

Usage:
    export MP_API_KEY='your-api-key-here'
    python3 match_materials_to_mp.py [--dry-run] [--limit N]

Options:
    --dry-run    Show what would be matched without saving
    --limit N    Only process first N materials (for testing)
    --force      Re-match materials that already have MP IDs
"""

import os
import sys
import json
import re
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from materials_lab.materials_project_client import MaterialsProjectClient, MPMaterialData
from materials_lab.materials_database import MaterialsDatabase, MaterialProperties


@dataclass
class MatchResult:
    """Result of matching a material to MP"""
    material_name: str
    formula: Optional[str]
    mp_id: Optional[str]
    confidence: float  # 0-1 score
    match_method: str  # 'formula_exact', 'formula_search', 'property_match', 'manual', 'no_match'
    notes: str = ""


class MaterialMatcher:
    """Matches lab materials to Materials Project entries"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize matcher with MP API client"""
        self.mp_client = MaterialsProjectClient(api_key=api_key)
        self.cache: Dict[str, List[MPMaterialData]] = {}
        self.match_results: List[MatchResult] = []

        # Manual mappings for common materials that are hard to parse
        self.manual_mappings = {
            # Aluminum alloys -> Pure Al
            "Al 2024-T3": "mp-134",  # Al
            "Al 6061-T6": "mp-134",
            "Al 7075-T6": "mp-134",

            # Stainless steels -> Fe (as approximation)
            "SS 304": "mp-13",  # Fe
            "SS 316": "mp-13",
            "SS 17-4 PH": "mp-13",

            # Titanium alloys -> Pure Ti
            "Ti-6Al-4V": "mp-72",  # Ti
            "Ti Grade 2": "mp-72",

            # Copper
            "Cu C11000": "mp-30",  # Cu
            "Brass C26000": "mp-30",  # Cu (approximation)

            # Nickel
            "Ni 200": "mp-23",  # Ni

            # Common ceramics
            "Alumina 99.5%": "mp-1143",  # Al2O3
            "Silicon Carbide": "mp-8062",  # SiC
            "Zirconia 3Y-TZP": "mp-2858",  # ZrO2

            # Nanomaterials
            "Graphene": "mp-48",  # C (graphite)
            "SWCNT": "mp-48",  # C (graphite)
        }

    def extract_formula(self, material_name: str) -> Optional[str]:
        """
        Extract chemical formula from material name

        Examples:
            'Silicon Carbide' -> 'SiC'
            'Alumina 99.5%' -> 'Al2O3'
            'Al 2024-T3' -> 'Al'
            'SS 304' -> None (alloy, no simple formula)
        """
        name = material_name.strip()

        # Check for common chemical names
        chemical_formulas = {
            # Oxides
            'alumina': 'Al2O3',
            'silica': 'SiO2',
            'titania': 'TiO2',
            'zirconia': 'ZrO2',
            'magnesia': 'MgO',
            'zinc oxide': 'ZnO',
            'copper oxide': 'CuO',
            'iron oxide': 'Fe2O3',
            'hematite': 'Fe2O3',
            'magnetite': 'Fe3O4',

            # Carbides
            'silicon carbide': 'SiC',
            'tungsten carbide': 'WC',
            'titanium carbide': 'TiC',
            'boron carbide': 'B4C',

            # Nitrides
            'silicon nitride': 'Si3N4',
            'titanium nitride': 'TiN',
            'aluminum nitride': 'AlN',
            'boron nitride': 'BN',

            # Elements
            'silicon': 'Si',
            'copper': 'Cu',
            'aluminum': 'Al',
            'titanium': 'Ti',
            'nickel': 'Ni',
            'iron': 'Fe',
            'chromium': 'Cr',
            'tungsten': 'W',
            'molybdenum': 'Mo',
            'graphene': 'C',
            'graphite': 'C',
            'diamond': 'C',

            # Common compounds
            'quartz': 'SiO2',
            'sapphire': 'Al2O3',
            'ruby': 'Al2O3',
        }

        name_lower = name.lower()
        for chem_name, formula in chemical_formulas.items():
            if chem_name in name_lower:
                return formula

        # Try to extract element symbol from alloy designation
        # e.g., "Al 6061-T6" -> "Al"
        if name.startswith(('Al ', 'Ti ', 'Cu ', 'Ni ', 'Fe ')):
            return name.split()[0]

        # Check if name itself is a formula (e.g., "SiO2", "Fe2O3")
        formula_pattern = r'^[A-Z][a-z]?(\d*[A-Z][a-z]?\d*)*$'
        if re.match(formula_pattern, name):
            return name

        return None

    def search_by_formula(self, formula: str, limit: int = 10) -> List[MPMaterialData]:
        """Search MP database by chemical formula"""
        # Check cache first
        if formula in self.cache:
            return self.cache[formula]

        print(f"  Searching MP for formula: {formula}")
        results = self.mp_client.search_materials(
            formula=formula,
            is_stable=True,
            limit=limit
        )

        # Cache results
        self.cache[formula] = results

        # Small delay to respect rate limits
        time.sleep(0.25)

        return results

    def find_best_match(self, material: MaterialProperties, mp_materials: List[MPMaterialData]) -> Tuple[Optional[MPMaterialData], float, str]:
        """
        Find best matching MP material by comparing properties

        Returns:
            (best_match, confidence_score, notes)
        """
        if not mp_materials:
            return None, 0.0, "No MP materials to compare"

        if len(mp_materials) == 1:
            return mp_materials[0], 0.9, "Only one match found"

        # Score each candidate
        scores = []
        for mp_mat in mp_materials:
            score = 0.0
            reasons = []

            # Compare density (if available)
            if material.density > 0 and mp_mat.density > 0:
                density_lab = material.density / 1000  # kg/m³ -> g/cm³
                density_diff = abs(density_lab - mp_mat.density) / mp_mat.density
                if density_diff < 0.1:  # Within 10%
                    score += 0.4
                    reasons.append(f"density match ({density_diff*100:.1f}% diff)")
                elif density_diff < 0.3:  # Within 30%
                    score += 0.2
                    reasons.append(f"density close ({density_diff*100:.1f}% diff)")

            # Compare band gap (if available)
            if hasattr(material, 'band_gap_ev') and material.band_gap_ev > 0:
                gap_diff = abs(material.band_gap_ev - mp_mat.band_gap)
                if gap_diff < 0.5:
                    score += 0.3
                    reasons.append(f"band gap match ({gap_diff:.2f} eV diff)")

            # Prefer stable materials
            if mp_mat.is_stable:
                score += 0.2
                reasons.append("stable phase")

            # Prefer materials with experimental data
            if not mp_mat.theoretical:
                score += 0.1
                reasons.append("experimental data")

            scores.append((mp_mat, score, "; ".join(reasons) if reasons else "formula match only"))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        best_match, best_score, notes = scores[0]

        return best_match, best_score, notes

    def match_material(self, material: MaterialProperties, force: bool = False) -> MatchResult:
        """
        Match a single material to Materials Project

        Args:
            material: Material to match
            force: Re-match even if MP ID already exists
        """
        name = material.name

        # Check if already has MP ID (if we add that field)
        # if hasattr(material, 'mp_id') and material.mp_id and not force:
        #     return MatchResult(name, None, material.mp_id, 1.0, 'existing', 'Already matched')

        # Check manual mappings first
        if name in self.manual_mappings:
            mp_id = self.manual_mappings[name]
            return MatchResult(
                name, None, mp_id, 0.95, 'manual',
                'Manual mapping for alloy/commercial material'
            )

        # Try to extract formula
        formula = self.extract_formula(name)

        if not formula:
            # Can't extract formula - common for alloys and commercial materials
            return MatchResult(
                name, None, None, 0.0, 'no_match',
                'Could not extract chemical formula from name'
            )

        # Search MP by formula
        mp_materials = self.search_by_formula(formula)

        if not mp_materials:
            return MatchResult(
                name, formula, None, 0.0, 'no_match',
                f'No MP entries found for formula {formula}'
            )

        # Find best match
        best_match, confidence, notes = self.find_best_match(material, mp_materials)

        if best_match:
            return MatchResult(
                name, formula, best_match.mp_id, confidence,
                'formula_search' if len(mp_materials) > 1 else 'formula_exact',
                notes
            )
        else:
            return MatchResult(
                name, formula, None, 0.0, 'no_match',
                'Search returned no usable results'
            )

    def match_all_materials(self, db: MaterialsDatabase, limit: Optional[int] = None, force: bool = False) -> List[MatchResult]:
        """
        Match all materials in database

        Args:
            db: Materials database
            limit: Maximum number of materials to process (for testing)
            force: Re-match materials that already have MP IDs
        """
        results = []
        materials = list(db.materials.values())

        if limit:
            materials = materials[:limit]

        total = len(materials)
        print(f"\nMatching {total} materials to Materials Project...")
        print("=" * 70)

        matched = 0
        for i, material in enumerate(materials, 1):
            print(f"\n[{i}/{total}] {material.name}")

            result = self.match_material(material, force=force)
            results.append(result)

            if result.mp_id:
                matched += 1
                print(f"  ✓ Matched: {result.mp_id} (confidence: {result.confidence:.2f})")
                if result.notes:
                    print(f"    {result.notes}")
            else:
                print(f"  ✗ No match: {result.notes}")

            # Progress update every 50 materials
            if i % 50 == 0:
                print(f"\n--- Progress: {i}/{total} ({matched} matched) ---")

        print("\n" + "=" * 70)
        print(f"Matching complete: {matched}/{total} materials matched ({matched/total*100:.1f}%)")

        return results

    def save_results(self, results: List[MatchResult], output_path: str):
        """Save matching results to JSON file"""
        data = []
        for r in results:
            data.append({
                'material_name': r.material_name,
                'formula': r.formula,
                'mp_id': r.mp_id,
                'confidence': r.confidence,
                'match_method': r.match_method,
                'notes': r.notes
            })

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    def update_database(self, db: MaterialsDatabase, results: List[MatchResult]) -> int:
        """
        Update materials database with MP IDs

        Returns:
            Number of materials updated
        """
        updated = 0

        for result in results:
            if result.mp_id:
                material = db.materials.get(result.material_name)
                if material:
                    # Add MP metadata to notes or provenance
                    mp_info = f"Materials Project ID: {result.mp_id}"
                    if material.notes:
                        if "Materials Project ID:" not in material.notes:
                            material.notes += f" | {mp_info}"
                    else:
                        material.notes = mp_info

                    # Add to provenance if available
                    if material.provenance is None:
                        material.provenance = {}
                    material.provenance['mp_id'] = result.mp_id
                    material.provenance['mp_formula'] = result.formula
                    material.provenance['mp_match_confidence'] = result.confidence
                    material.provenance['mp_match_method'] = result.match_method

                    updated += 1

        return updated


def main():
    parser = argparse.ArgumentParser(description='Match lab materials to Materials Project IDs')
    parser.add_argument('--dry-run', action='store_true', help='Show matches without saving')
    parser.add_argument('--limit', type=int, help='Only process first N materials')
    parser.add_argument('--force', action='store_true', help='Re-match materials with existing MP IDs')
    parser.add_argument('--output', type=str, default='mp_matching_results.json', help='Output file for results')

    args = parser.parse_args()

    # Check for API key
    if not os.environ.get('MP_API_KEY'):
        print("\n❌ Error: MP_API_KEY environment variable not set!")
        print("\nTo get a free API key:")
        print("  1. Visit https://materialsproject.org/api")
        print("  2. Sign up for a free account")
        print("  3. Copy your API key")
        print("  4. Set it: export MP_API_KEY='your-key-here'")
        return 1

    try:
        # Load materials database
        print("Loading materials database...")
        db = MaterialsDatabase()
        print(f"✓ Loaded {db.get_count()} materials")

        # Initialize matcher
        print("\nInitializing Materials Project client...")
        matcher = MaterialMatcher()
        print("✓ Connected to Materials Project API")

        # Match materials
        results = matcher.match_all_materials(db, limit=args.limit, force=args.force)

        # Save results
        matcher.save_results(results, args.output)

        # Print summary
        print("\n" + "=" * 70)
        print("MATCHING SUMMARY")
        print("=" * 70)

        by_method = {}
        for r in results:
            by_method[r.match_method] = by_method.get(r.match_method, 0) + 1

        print("\nMatches by method:")
        for method, count in sorted(by_method.items()):
            print(f"  {method}: {count}")

        matched = sum(1 for r in results if r.mp_id)
        total = len(results)
        print(f"\nTotal matched: {matched}/{total} ({matched/total*100:.1f}%)")

        # Update database unless dry-run
        if not args.dry_run:
            print("\nUpdating materials database...")
            updated = matcher.update_database(db, results)
            print(f"✓ Updated {updated} materials with MP IDs")

            db.save()
            print("✓ Database saved")
        else:
            print("\n⚠️  Dry run - database not updated")

        print("\n✓ Matching complete!")
        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
