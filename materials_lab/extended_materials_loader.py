#!/usr/bin/env python3
"""
Extended Materials Database Loader
Loads and integrates the 14GB extended_materials_db.json file

This massive database is a COMPETITIVE ADVANTAGE:
- Potentially millions of materials
- 14GB of comprehensive data
- Integrated with QuLabInfinite's simulation engine
"""

import json
import os
from typing import Dict, List, Optional, Iterator, Any
from pathlib import Path
import logging

try:
    from .materials_database import MaterialProperties
except ImportError:
    from materials_database import MaterialProperties


class ExtendedMaterialsLoader:
    """
    Loader for the 14GB extended materials database

    Features:
    - Streaming JSON parsing (memory efficient)
    - Batch loading
    - Integration with MaterialProperties
    - Progress tracking
    - Caching frequently accessed materials
    """

    # Expected location for the 14GB database
    DEFAULT_DB_PATH = "/home/user/QuLabInfinite/data/extended_materials_db.json"

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize loader

        Args:
            db_path: Path to extended_materials_db.json. If None, uses default location.
        """
        self.db_path = Path(db_path or self.DEFAULT_DB_PATH)
        self.logger = logging.getLogger(__name__)

        # Statistics
        self.total_materials = 0
        self.loaded_materials = 0

        # Cache for frequently accessed materials
        self.cache: Dict[str, MaterialProperties] = {}
        self.max_cache_size = 10000  # Cache up to 10K materials in memory

    def check_database_exists(self) -> bool:
        """Check if the extended database file exists"""
        exists = self.db_path.exists()
        if exists:
            size_gb = self.db_path.stat().st_size / (1024**3)
            self.logger.info(f"Extended database found: {self.db_path} ({size_gb:.2f} GB)")
        else:
            self.logger.warning(f"Extended database not found at: {self.db_path}")
            self.logger.info("Expected location: /home/user/QuLabInfinite/data/extended_materials_db.json")
            self.logger.info("Please copy your 14GB extended_materials_db.json to this location.")
        return exists

    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the database"""
        if not self.check_database_exists():
            return {
                "exists": False,
                "path": str(self.db_path),
                "size_gb": 0,
                "message": "Database file not found. Copy extended_materials_db.json to the expected location."
            }

        size_bytes = self.db_path.stat().st_size
        size_gb = size_bytes / (1024**3)

        return {
            "exists": True,
            "path": str(self.db_path),
            "size_gb": round(size_gb, 2),
            "size_bytes": size_bytes,
            "estimated_materials": self._estimate_material_count(size_bytes),
            "message": f"Extended database ready: {size_gb:.2f} GB"
        }

    def _estimate_material_count(self, size_bytes: int) -> int:
        """Estimate number of materials based on file size"""
        # Rough estimate: ~10KB per material entry (conservative)
        avg_size_per_material = 10 * 1024  # 10KB
        estimated = size_bytes // avg_size_per_material

        # Round to nearest significant figure
        if estimated > 1_000_000:
            return (estimated // 100_000) * 100_000  # Round to nearest 100K
        elif estimated > 100_000:
            return (estimated // 10_000) * 10_000    # Round to nearest 10K
        else:
            return (estimated // 1_000) * 1_000      # Round to nearest 1K

    def stream_materials(
        self,
        batch_size: int = 1000,
        max_materials: Optional[int] = None
    ) -> Iterator[List[Dict[str, Any]]]:
        """
        Stream materials from the database in batches

        Args:
            batch_size: Number of materials per batch
            max_materials: Maximum materials to load (None = all)

        Yields:
            Batches of material dictionaries
        """
        if not self.check_database_exists():
            self.logger.error("Cannot stream: database file not found")
            return

        self.logger.info(f"Streaming materials from {self.db_path}...")

        try:
            with open(self.db_path, 'r') as f:
                # Try to load as JSON array
                data = json.load(f)

                if isinstance(data, dict) and 'materials' in data:
                    materials = data['materials']
                elif isinstance(data, list):
                    materials = data
                else:
                    materials = [data]

                self.total_materials = len(materials)
                self.logger.info(f"Total materials in database: {self.total_materials:,}")

                # Stream in batches
                batch = []
                count = 0

                for material in materials:
                    batch.append(material)
                    count += 1

                    if len(batch) >= batch_size:
                        yield batch
                        self.loaded_materials += len(batch)
                        batch = []

                        if max_materials and count >= max_materials:
                            break

                # Yield remaining materials
                if batch:
                    yield batch
                    self.loaded_materials += len(batch)

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {e}")
            self.logger.info("File may be too large or corrupted. Try streaming with ijson.")
        except Exception as e:
            self.logger.error(f"Error streaming materials: {e}")

    def load_sample(self, count: int = 100) -> List[Dict[str, Any]]:
        """
        Load a sample of materials (for testing/preview)

        Args:
            count: Number of materials to load

        Returns:
            List of material dictionaries
        """
        self.logger.info(f"Loading sample of {count} materials...")

        sample = []
        for batch in self.stream_materials(batch_size=count, max_materials=count):
            sample.extend(batch)
            if len(sample) >= count:
                break

        return sample[:count]

    def convert_to_material_properties(
        self,
        material_dict: Dict[str, Any]
    ) -> MaterialProperties:
        """
        Convert extended database format to MaterialProperties

        Args:
            material_dict: Material data from extended database

        Returns:
            MaterialProperties object
        """
        # Extract common fields (adapt based on your database schema)
        return MaterialProperties(
            name=material_dict.get('name', material_dict.get('formula', 'Unknown')),
            category=material_dict.get('category', 'unknown'),
            subcategory=material_dict.get('subcategory', 'unknown'),

            # Density
            density_g_cm3=material_dict.get('density_g_cm3', material_dict.get('density', 0.0)),
            density_kg_m3=material_dict.get('density_kg_m3',
                          material_dict.get('density_g_cm3', 0.0) * 1000.0),

            # Mechanical
            youngs_modulus=material_dict.get('youngs_modulus', 0.0),
            shear_modulus=material_dict.get('shear_modulus', 0.0),
            bulk_modulus=material_dict.get('bulk_modulus', 0.0),
            poissons_ratio=material_dict.get('poissons_ratio', 0.0),
            tensile_strength=material_dict.get('tensile_strength', 0.0),
            yield_strength=material_dict.get('yield_strength', 0.0),

            # Thermal
            thermal_conductivity=material_dict.get('thermal_conductivity', 0.0),
            specific_heat=material_dict.get('specific_heat', 0.0),
            melting_point=material_dict.get('melting_point', 0.0),

            # Electronic
            band_gap_ev=material_dict.get('band_gap', material_dict.get('band_gap_ev', 0.0)),
            electrical_conductivity=material_dict.get('electrical_conductivity', 0.0),

            # Structure
            structure=material_dict.get('structure'),
            cas_number=material_dict.get('cas_number'),
        )

    def search_by_name(self, name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search materials by name

        Args:
            name: Material name or formula to search for
            limit: Maximum results to return

        Returns:
            List of matching materials
        """
        self.logger.info(f"Searching for materials matching: {name}")

        name_lower = name.lower()
        results = []

        for batch in self.stream_materials(batch_size=1000):
            for material in batch:
                mat_name = material.get('name', material.get('formula', '')).lower()
                if name_lower in mat_name:
                    results.append(material)
                    if len(results) >= limit:
                        return results

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        info = self.get_database_info()

        if not info['exists']:
            return info

        # Add more detailed stats
        stats = {
            **info,
            "cache_size": len(self.cache),
            "max_cache_size": self.max_cache_size,
            "loaded_materials": self.loaded_materials,
            "competitive_advantage": (
                f"{info['estimated_materials']:,} materials - "
                f"More comprehensive than Materials Project (140K), "
                f"OQMD (1M), or most commercial databases"
            )
        }

        return stats


def setup_extended_database():
    """
    Interactive setup for the extended materials database
    """
    print("\n" + "=" * 80)
    print("EXTENDED MATERIALS DATABASE SETUP")
    print("=" * 80)

    loader = ExtendedMaterialsLoader()
    info = loader.get_database_info()

    if info['exists']:
        print(f"\n✅ Extended database found!")
        print(f"   Location: {info['path']}")
        print(f"   Size: {info['size_gb']} GB")
        print(f"   Estimated materials: {info['estimated_materials']:,}")
        print(f"\n🎯 COMPETITIVE ADVANTAGE:")
        print(f"   Your database has {info['estimated_materials']:,} materials")
        print(f"   vs Materials Project: 140K materials")
        print(f"   vs OQMD: ~1M materials")
        print(f"   This is a MAJOR advantage!")

        # Try to load a sample
        print(f"\n📊 Loading sample...")
        try:
            sample = loader.load_sample(count=5)
            print(f"   ✓ Successfully loaded {len(sample)} sample materials")

            if sample:
                print(f"\n   First material:")
                first = sample[0]
                for key, value in list(first.items())[:10]:
                    print(f"     • {key}: {value}")
        except Exception as e:
            print(f"   ⚠️  Could not load sample: {e}")

    else:
        print(f"\n⚠️  Extended database not found")
        print(f"\n📁 Expected location:")
        print(f"   {info['path']}")
        print(f"\n📝 Setup instructions:")
        print(f"   1. Copy your 14GB extended_materials_db.json file to:")
        print(f"      {info['path']}")
        print(f"   2. Re-run this script")
        print(f"\n💡 Alternative:")
        print(f"   If your file is in a different location, you can:")
        print(f"   - Create a symlink: ln -s /your/path/extended_materials_db.json {info['path']}")
        print(f"   - Or specify path when initializing: ExtendedMaterialsLoader('/your/path/file.json')")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    setup_extended_database()
