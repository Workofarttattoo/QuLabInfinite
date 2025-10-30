#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

QuLabInfinite Materials Test List
Comprehensive list of materials for testing across all lab departments
"""

from typing import List, Dict
from materials_lab.materials_database import MaterialsDatabase


class MaterialsTestList:
    """Curated materials list for QuLabInfinite testing"""

    def __init__(self):
        self.db = MaterialsDatabase()

        # Comprehensive test material categories
        self.test_materials = {
            # Structural materials for mechanical testing
            'structural': [
                '304 Stainless Steel',
                'Al 6061-T6',
                'Ti-6Al-4V',
                'Carbon Fiber Epoxy',
                'Al 7075-T6',
            ],

            # Aerogels for thermal and environmental testing
            'aerogels': [
                'Airloy X103 Strong Aerogel',
                'Silica Aerogel',
                'Graphene Aerogel',
            ],

            # Energy materials for electrochemical testing
            'energy': [
                'NMC_111',
                'NMC_622',
                'NMC_811',
                'NMC811 Cathode Powder',
                'Graphite_Anode',
                'Silicon_Anode',
            ],

            # Optical materials for photonics testing
            'optical': [
                'BBO',
                'LBO',
                'KTP',
                'ZnSe_Window',
                'Sapphire substrate',
            ],

            # Superconductors for quantum testing
            'superconductors': [
                'YBCO',
                'Nb3Sn',
                'NbN',
                'BSCCO_2223',
            ],

            # 2D materials for nanoelectronics testing
            '2d_materials': [
                'MoSe2',
                'Ti3C2Tx',
                'Bi2Te3',
                'WTe2',
                'Phosphorene',
                'Graphene',
            ],

            # Ceramics for high-temperature testing
            'ceramics': [
                'Silicon Carbide',
                'Tungsten Carbide (WC-Co)',
                'Silicon Nitride Si3N4',
                'HfC',
                'TaC',
            ],

            # Magnetic materials for EM testing
            'magnetic': [
                'Neodymium',
                'Permalloy_80',
                'Supermalloy',
            ],

            # Thermal materials for heat transfer testing
            'thermal': [
                'Gallium Arsenide',
                'Graphene CVD',
            ],

            # Piezoelectric materials for sensor testing
            'piezoelectric': [
                'PZT_4',
                'PZT_5A',
                'PZT_5H',
            ],

            # Biomaterials for medical testing
            'biomaterials': [
                'Hydroxyapatite',
                'PEEK',
                'PCL',
            ],
        }

    def get_all_test_materials(self) -> List[str]:
        """Get flat list of all test materials"""
        all_materials = []
        for category_materials in self.test_materials.values():
            all_materials.extend(category_materials)
        return all_materials

    def get_by_category(self, category: str) -> List[str]:
        """Get test materials by category"""
        return self.test_materials.get(category, [])

    def verify_materials(self) -> Dict[str, bool]:
        """Verify all test materials exist in database"""
        verification = {}
        for mat_name in self.get_all_test_materials():
            verification[mat_name] = mat_name in self.db.materials
        return verification

    def get_available_materials(self) -> List[str]:
        """Get list of test materials available in database"""
        return [name for name in self.get_all_test_materials()
                if name in self.db.materials]

    def get_missing_materials(self) -> List[str]:
        """Get list of test materials missing from database"""
        return [name for name in self.get_all_test_materials()
                if name not in self.db.materials]

    def generate_test_report(self) -> str:
        """Generate comprehensive test materials report"""
        lines = []
        lines.append("=" * 80)
        lines.append("QuLabInfinite Materials Test List Report")
        lines.append("=" * 80)
        lines.append("")

        all_materials = self.get_all_test_materials()
        available = self.get_available_materials()
        missing = self.get_missing_materials()

        lines.append(f"Total test materials: {len(all_materials)}")
        lines.append(f"Available: {len(available)} ({len(available)/len(all_materials)*100:.1f}%)")
        lines.append(f"Missing: {len(missing)} ({len(missing)/len(all_materials)*100:.1f}%)")
        lines.append("")

        # By category
        lines.append("Materials by Category:")
        lines.append("-" * 80)
        for category, materials in self.test_materials.items():
            available_in_cat = sum(1 for m in materials if m in self.db.materials)
            lines.append(f"\n{category.upper()} ({available_in_cat}/{len(materials)} available):")
            for mat_name in materials:
                status = "✓" if mat_name in self.db.materials else "✗"
                lines.append(f"  {status} {mat_name}")

        if missing:
            lines.append("")
            lines.append("=" * 80)
            lines.append("Missing Materials (need to be added):")
            lines.append("-" * 80)
            for mat_name in missing:
                lines.append(f"  - {mat_name}")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def export_for_ingest(self, output_path: str = None):
        """Export materials list for ingest pipeline"""
        import json

        if output_path is None:
            output_path = "/Users/noone/QuLabInfinite/materials_test_manifest.json"

        manifest = {
            "manifest_version": "1.0",
            "description": "QuLabInfinite materials test list",
            "total_materials": len(self.get_all_test_materials()),
            "categories": {}
        }

        for category, materials in self.test_materials.items():
            manifest["categories"][category] = {
                "count": len(materials),
                "materials": materials,
                "available": [m for m in materials if m in self.db.materials],
                "missing": [m for m in materials if m not in self.db.materials]
            }

        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"[info] Exported materials test manifest to: {output_path}")
        return output_path


def main():
    """Generate and display materials test list"""
    test_list = MaterialsTestList()

    # Generate report
    print(test_list.generate_test_report())

    # Export manifest
    test_list.export_for_ingest()

    # Summary
    available = test_list.get_available_materials()
    print(f"\n✓ {len(available)} materials ready for QuLabInfinite testing")


if __name__ == "__main__":
    main()
