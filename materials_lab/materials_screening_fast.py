#!/usr/bin/env python3
"""
Fast Materials Screening with Polars
10-100x faster than Pandas for large datasets

Usage:
    from materials_lab.materials_screening_fast import FastMaterialsScreener

    screener = FastMaterialsScreener()
    results = screener.screen_lightweight_strong(density_max=500, strength_min=50)
"""

import polars as pl
from pathlib import Path
from typing import Dict, List, Optional
import json


class FastMaterialsScreener:
    """Ultra-fast materials screening using Polars (Rust-based DataFrame)"""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize screener with materials database

        Args:
            db_path: Path to materials JSON database
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent / "data" / "extended_materials_db.json"

        self.db_path = Path(db_path)
        self._df: Optional[pl.DataFrame] = None

    def load_database(self) -> pl.DataFrame:
        """
        Load materials database into Polars DataFrame

        Returns:
            Polars DataFrame with all materials
        """
        if self._df is not None:
            return self._df

        if not self.db_path.exists():
            print(f"⚠️  Extended database not found at {self.db_path}")
            print("   Using curated materials library instead...")
            return self._load_curated_library()

        print(f"📊 Loading materials database from {self.db_path}...")
        print(f"   File size: {self.db_path.stat().st_size / (1024**3):.2f} GB")

        # Polars can read JSON directly with lazy evaluation
        self._df = pl.read_json(self.db_path)

        print(f"✅ Loaded {len(self._df):,} materials")
        return self._df

    def _load_curated_library(self) -> pl.DataFrame:
        """Fallback: load sample materials for testing"""
        print("   Creating sample materials dataset for testing...")

        # Create sample materials for demonstration
        materials = [
            {'id': 'Al', 'name': 'Aluminum', 'category': 'metal', 'density_kg_m3': 2700, 'tensile_strength': 90, 'youngs_modulus': 70, 'thermal_conductivity': 237, 'melting_point': 933},
            {'id': 'Ti', 'name': 'Titanium', 'category': 'metal', 'density_kg_m3': 4500, 'tensile_strength': 450, 'youngs_modulus': 116, 'thermal_conductivity': 21.9, 'melting_point': 1941},
            {'id': 'Steel', 'name': 'Steel', 'category': 'metal', 'density_kg_m3': 7850, 'tensile_strength': 400, 'youngs_modulus': 200, 'thermal_conductivity': 50, 'melting_point': 1673},
            {'id': 'CFRP', 'name': 'Carbon Fiber', 'category': 'composite', 'density_kg_m3': 1600, 'tensile_strength': 600, 'youngs_modulus': 150, 'thermal_conductivity': 7, 'melting_point': 3800},
            {'id': 'Aerogel', 'name': 'Silica Aerogel', 'category': 'ceramic', 'density_kg_m3': 100, 'tensile_strength': 0.02, 'youngs_modulus': 0.01, 'thermal_conductivity': 0.015, 'melting_point': 1473},
            {'id': 'Cu', 'name': 'Copper', 'category': 'metal', 'density_kg_m3': 8960, 'tensile_strength': 220, 'youngs_modulus': 130, 'thermal_conductivity': 401, 'melting_point': 1358},
            {'id': 'Al2O3', 'name': 'Alumina', 'category': 'ceramic', 'density_kg_m3': 3950, 'tensile_strength': 300, 'youngs_modulus': 370, 'thermal_conductivity': 30, 'melting_point': 2345},
            {'id': 'Mg', 'name': 'Magnesium', 'category': 'metal', 'density_kg_m3': 1740, 'tensile_strength': 125, 'youngs_modulus': 45, 'thermal_conductivity': 156, 'melting_point': 923},
        ]

        self._df = pl.DataFrame(materials)
        print(f"✅ Loaded {len(self._df):,} sample materials for testing")
        print(f"   📝 Note: Copy 14GB extended_materials_db.json to {self.db_path} for full database")
        return self._df

    def screen_lightweight_strong(
        self,
        density_max: float = 500,
        strength_min: float = 50,
        limit: int = 100
    ) -> pl.DataFrame:
        """
        Find lightweight materials with high strength

        Args:
            density_max: Maximum density in kg/m³
            strength_min: Minimum tensile strength in MPa
            limit: Maximum results to return

        Returns:
            Polars DataFrame with matching materials
        """
        df = self.load_database()

        # Fast parallel filtering with Polars
        results = df.filter(
            (pl.col("density_kg_m3") < density_max) &
            (pl.col("tensile_strength") > strength_min)
        ).sort(
            "tensile_strength",
            descending=True
        ).head(limit)

        return results

    def screen_high_thermal_conductivity(
        self,
        conductivity_min: float = 100,
        limit: int = 100
    ) -> pl.DataFrame:
        """
        Find materials with high thermal conductivity

        Args:
            conductivity_min: Minimum thermal conductivity in W/(m·K)
            limit: Maximum results

        Returns:
            Matching materials sorted by conductivity
        """
        df = self.load_database()

        results = df.filter(
            pl.col("thermal_conductivity") > conductivity_min
        ).sort(
            "thermal_conductivity",
            descending=True
        ).head(limit)

        return results

    def screen_custom(self, filters: Dict, limit: int = 100) -> pl.DataFrame:
        """
        Custom materials screening with multiple criteria

        Args:
            filters: Dictionary of property filters
                Example: {
                    'density_max': 500,
                    'strength_min': 50,
                    'thermal_conductivity_min': 10,
                    'melting_point_min': 1000
                }
            limit: Maximum results

        Returns:
            Matching materials
        """
        df = self.load_database()

        # Build filter expression
        conditions = []

        if 'density_max' in filters:
            conditions.append(pl.col("density_kg_m3") < filters['density_max'])
        if 'density_min' in filters:
            conditions.append(pl.col("density_kg_m3") > filters['density_min'])

        if 'strength_min' in filters:
            conditions.append(pl.col("tensile_strength") > filters['strength_min'])
        if 'strength_max' in filters:
            conditions.append(pl.col("tensile_strength") < filters['strength_max'])

        if 'thermal_conductivity_min' in filters:
            conditions.append(pl.col("thermal_conductivity") > filters['thermal_conductivity_min'])

        if 'melting_point_min' in filters:
            conditions.append(pl.col("melting_point") > filters['melting_point_min'])
        if 'melting_point_max' in filters:
            conditions.append(pl.col("melting_point") < filters['melting_point_max'])

        # Combine all conditions with AND
        if not conditions:
            return df.head(limit)

        combined = conditions[0]
        for condition in conditions[1:]:
            combined = combined & condition

        results = df.filter(combined).head(limit)
        return results

    def get_statistics(self) -> Dict:
        """
        Get statistics about the materials database

        Returns:
            Dictionary with database statistics
        """
        df = self.load_database()

        stats = {
            'total_materials': len(df),
            'density': {
                'min': df['density_kg_m3'].min(),
                'max': df['density_kg_m3'].max(),
                'mean': df['density_kg_m3'].mean(),
            },
            'tensile_strength': {
                'min': df['tensile_strength'].min(),
                'max': df['tensile_strength'].max(),
                'mean': df['tensile_strength'].mean(),
            }
        }

        return stats

    def benchmark(self) -> Dict:
        """
        Benchmark screening performance

        Returns:
            Timing statistics
        """
        import time

        df = self.load_database()

        # Benchmark filtering
        start = time.time()
        results = df.filter(
            (pl.col("density_kg_m3") < 500) &
            (pl.col("tensile_strength") > 50)
        ).head(100)
        filter_time = time.time() - start

        # Benchmark sorting
        start = time.time()
        sorted_df = df.sort("tensile_strength", descending=True).head(1000)
        sort_time = time.time() - start

        # Benchmark aggregation
        start = time.time()
        stats = df.select([
            pl.col("density_kg_m3").mean().alias("avg_density"),
            pl.col("tensile_strength").mean().alias("avg_strength"),
        ])
        agg_time = time.time() - start

        return {
            'filter_time_ms': filter_time * 1000,
            'sort_time_ms': sort_time * 1000,
            'aggregation_time_ms': agg_time * 1000,
            'materials_count': len(df),
        }

    def export_csv(self, results: pl.DataFrame, filename: str) -> str:
        """
        Export screening results to CSV

        Args:
            results: DataFrame to export
            filename: Output CSV filename

        Returns:
            Path to exported file
        """
        output_path = Path(filename)
        results.write_csv(output_path)
        print(f"✅ Exported {len(results)} materials to {output_path}")
        return str(output_path)

    def export_json(self, results: pl.DataFrame, filename: str) -> str:
        """
        Export screening results to JSON

        Args:
            results: DataFrame to export
            filename: Output JSON filename

        Returns:
            Path to exported file
        """
        output_path = Path(filename)
        results.write_json(output_path)
        print(f"✅ Exported {len(results)} materials to {output_path}")
        return str(output_path)

    def to_dict(self, results: pl.DataFrame) -> List[Dict]:
        """
        Convert results to list of dictionaries

        Args:
            results: DataFrame to convert

        Returns:
            List of material dictionaries
        """
        return results.to_dicts()

    def search_by_composition(self, element: str, limit: int = 100) -> pl.DataFrame:
        """
        Search materials containing a specific element

        Args:
            element: Chemical element symbol (e.g., 'Fe', 'Al', 'Ti')
            limit: Maximum results

        Returns:
            Materials containing the element

        Example:
            screener.search_by_composition('Fe')  # All iron-containing materials
        """
        df = self.load_database()

        # Search in id, name, or formula fields
        results = df.filter(
            pl.col("id").str.contains(element, literal=False) |
            pl.col("name").str.contains(element, literal=False)
        ).head(limit)

        return results

    def search_by_name(self, pattern: str, fuzzy: bool = False, limit: int = 100) -> pl.DataFrame:
        """
        Search materials by name (case-insensitive)

        Args:
            pattern: Name pattern to search
            fuzzy: If True, use fuzzy matching
            limit: Maximum results

        Returns:
            Matching materials

        Example:
            screener.search_by_name('aluminum')  # Case-insensitive
            screener.search_by_name('steel', fuzzy=True)  # Fuzzy matching
        """
        df = self.load_database()

        if fuzzy:
            # Fuzzy matching using lowercase contains
            results = df.filter(
                pl.col("name").str.to_lowercase().str.contains(pattern.lower())
            ).head(limit)
        else:
            # Exact case-insensitive match
            results = df.filter(
                pl.col("name").str.to_lowercase() == pattern.lower()
            ).head(limit)

        return results

    def search_by_category(self, category: str, limit: int = 1000) -> pl.DataFrame:
        """
        Search materials by category

        Args:
            category: Material category (e.g., 'metal', 'ceramic', 'polymer')
            limit: Maximum results

        Returns:
            All materials in category
        """
        df = self.load_database()

        results = df.filter(
            pl.col("category") == category
        ).head(limit)

        return results


def main():
    """Demo: Fast materials screening"""

    print("=" * 80)
    print("FAST MATERIALS SCREENING - Powered by Polars")
    print("=" * 80)

    screener = FastMaterialsScreener()

    # Benchmark
    print("\n📊 Running performance benchmark...")
    benchmark = screener.benchmark()
    print(f"\n   Materials: {benchmark['materials_count']:,}")
    print(f"   Filter time: {benchmark['filter_time_ms']:.2f} ms")
    print(f"   Sort time: {benchmark['sort_time_ms']:.2f} ms")
    print(f"   Aggregation time: {benchmark['aggregation_time_ms']:.2f} ms")

    # Screen lightweight strong materials
    print("\n" + "=" * 80)
    print("🔍 Screening: Lightweight + High Strength")
    print("   Criteria: density < 500 kg/m³, strength > 50 MPa")
    print("=" * 80)

    results = screener.screen_lightweight_strong(
        density_max=500,
        strength_min=50,
        limit=10
    )

    print(f"\n✅ Found {len(results)} materials")
    print("\nTop 10 Results:")
    print(results.select(['name', 'density_kg_m3', 'tensile_strength']))

    # Statistics
    print("\n" + "=" * 80)
    print("📈 Database Statistics")
    print("=" * 80)

    stats = screener.get_statistics()
    print(f"\n   Total materials: {stats['total_materials']:,}")
    print(f"\n   Density (kg/m³):")
    print(f"      Min: {stats['density']['min']:.2f}")
    print(f"      Max: {stats['density']['max']:.2f}")
    print(f"      Mean: {stats['density']['mean']:.2f}")
    print(f"\n   Tensile Strength (MPa):")
    print(f"      Min: {stats['tensile_strength']['min']:.2f}")
    print(f"      Max: {stats['tensile_strength']['max']:.2f}")
    print(f"      Mean: {stats['tensile_strength']['mean']:.2f}")

    print("\n" + "=" * 80)
    print("🚀 Polars is 10-100x faster than Pandas for this workload!")
    print("=" * 80)


if __name__ == "__main__":
    main()
