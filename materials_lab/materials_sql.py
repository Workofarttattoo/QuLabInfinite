#!/usr/bin/env python3
"""
SQL Query Interface for Materials Database using DuckDB

Enables SQL queries on 14GB+ materials database without loading into memory

Usage:
    from materials_lab.materials_sql import MaterialsSQL

    db = MaterialsSQL()
    results = db.query("SELECT * FROM materials WHERE density < 500 LIMIT 10")
"""

import duckdb
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd
import json


class MaterialsSQL:
    """SQL query interface for materials database using DuckDB"""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize SQL interface

        Args:
            db_path: Path to materials JSON database
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent / "data" / "extended_materials_db.json"

        self.db_path = Path(db_path)
        self.con = duckdb.connect(database=':memory:')

        # Register materials view if database exists
        if self.db_path.exists():
            self._register_database()
        else:
            print(f"⚠️  Extended database not found at {self.db_path}")
            print("   SQL queries will use curated materials library")
            self._register_curated_library()

    def _register_database(self):
        """Register materials database as SQL view"""
        print(f"📊 Registering materials database for SQL queries...")

        # Create view from JSON file (DuckDB reads directly without loading to memory)
        self.con.execute(f"""
            CREATE VIEW materials AS
            SELECT * FROM read_json_auto('{self.db_path}')
        """)

        # Get count
        count = self.con.execute("SELECT COUNT(*) FROM materials").fetchone()[0]
        print(f"✅ Registered {count:,} materials for SQL queries")

    def _register_curated_library(self):
        """Fallback: register sample materials for testing"""
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

        # Create table from Python list (DuckDB can scan Python objects directly)
        import pandas as pd
        df = pd.DataFrame(materials)
        self.con.execute("CREATE TABLE materials AS SELECT * FROM df")
        print(f"✅ Registered {len(materials):,} sample materials for testing")
        print(f"   📝 Note: Copy 14GB extended_materials_db.json to {self.db_path} for full database")

    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute SQL query on materials database

        Args:
            sql: SQL query string (can reference 'materials' table)

        Returns:
            Pandas DataFrame with results

        Examples:
            # Simple select
            db.query("SELECT * FROM materials WHERE density_kg_m3 < 500 LIMIT 10")

            # Aggregations
            db.query("SELECT category, AVG(density_kg_m3) as avg_density FROM materials GROUP BY category")

            # Complex filters
            db.query('''
                SELECT name, density_kg_m3, tensile_strength
                FROM materials
                WHERE density_kg_m3 < 500
                  AND tensile_strength > 50
                ORDER BY tensile_strength DESC
                LIMIT 100
            ''')
        """
        try:
            result = self.con.execute(sql).df()
            return result
        except Exception as e:
            print(f"❌ Query error: {e}")
            raise

    def lightweight_strong(self, density_max: float = 500, strength_min: float = 50) -> pd.DataFrame:
        """
        Pre-built query: Find lightweight materials with high strength

        Args:
            density_max: Maximum density in kg/m³
            strength_min: Minimum tensile strength in MPa

        Returns:
            DataFrame with matching materials
        """
        return self.query(f"""
            SELECT
                name,
                density_kg_m3,
                tensile_strength,
                youngs_modulus,
                thermal_conductivity
            FROM materials
            WHERE density_kg_m3 < {density_max}
              AND tensile_strength > {strength_min}
            ORDER BY tensile_strength DESC
            LIMIT 100
        """)

    def high_temperature(self, melting_point_min: float = 2000) -> pd.DataFrame:
        """
        Pre-built query: Find high-temperature materials

        Args:
            melting_point_min: Minimum melting point in K

        Returns:
            Materials sorted by melting point
        """
        return self.query(f"""
            SELECT
                name,
                melting_point,
                thermal_conductivity,
                density_kg_m3
            FROM materials
            WHERE melting_point > {melting_point_min}
            ORDER BY melting_point DESC
            LIMIT 100
        """)

    def by_category(self, category: str) -> pd.DataFrame:
        """
        Pre-built query: Find materials by category

        Args:
            category: Material category (e.g., 'metal', 'ceramic', 'polymer')

        Returns:
            All materials in category
        """
        return self.query(f"""
            SELECT *
            FROM materials
            WHERE category = '{category}'
            LIMIT 1000
        """)

    def statistics_by_category(self) -> pd.DataFrame:
        """
        Pre-built query: Get statistics grouped by category

        Returns:
            Aggregated statistics per category
        """
        return self.query("""
            SELECT
                category,
                COUNT(*) as material_count,
                AVG(density_kg_m3) as avg_density,
                AVG(tensile_strength) as avg_strength,
                AVG(thermal_conductivity) as avg_conductivity,
                MIN(density_kg_m3) as min_density,
                MAX(density_kg_m3) as max_density
            FROM materials
            GROUP BY category
            ORDER BY material_count DESC
        """)

    def property_ranges(self) -> Dict:
        """
        Get min/max ranges for all properties

        Returns:
            Dictionary with property ranges
        """
        result = self.query("""
            SELECT
                MIN(density_kg_m3) as min_density,
                MAX(density_kg_m3) as max_density,
                MIN(tensile_strength) as min_strength,
                MAX(tensile_strength) as max_strength,
                MIN(thermal_conductivity) as min_conductivity,
                MAX(thermal_conductivity) as max_conductivity,
                MIN(melting_point) as min_melting_point,
                MAX(melting_point) as max_melting_point
            FROM materials
        """)

        return result.to_dict('records')[0]

    def top_n_by_property(self, property: str, n: int = 10, ascending: bool = False) -> pd.DataFrame:
        """
        Get top N materials by any property

        Args:
            property: Property column name (e.g., 'density_kg_m3', 'tensile_strength')
            n: Number of results
            ascending: Sort order (False = highest first)

        Returns:
            Top N materials
        """
        order = "ASC" if ascending else "DESC"
        return self.query(f"""
            SELECT name, {property}
            FROM materials
            WHERE {property} IS NOT NULL
            ORDER BY {property} {order}
            LIMIT {n}
        """)

    def search_by_name(self, name_pattern: str) -> pd.DataFrame:
        """
        Search materials by name (case-insensitive)

        Args:
            name_pattern: Name pattern (supports SQL LIKE wildcards)

        Returns:
            Matching materials

        Example:
            db.search_by_name('%steel%')  # All materials with "steel" in name
            db.search_by_name('Carbon%')  # Materials starting with "Carbon"
        """
        return self.query(f"""
            SELECT *
            FROM materials
            WHERE LOWER(name) LIKE LOWER('{name_pattern}')
            LIMIT 100
        """)

    def benchmark(self) -> Dict:
        """
        Benchmark SQL query performance

        Returns:
            Timing statistics
        """
        import time

        # Count query
        start = time.time()
        count = self.query("SELECT COUNT(*) FROM materials")
        count_time = time.time() - start

        # Filter query
        start = time.time()
        filtered = self.query("""
            SELECT * FROM materials
            WHERE density_kg_m3 < 500 AND tensile_strength > 50
            LIMIT 100
        """)
        filter_time = time.time() - start

        # Aggregation query
        start = time.time()
        stats = self.query("""
            SELECT
                AVG(density_kg_m3) as avg_density,
                AVG(tensile_strength) as avg_strength
            FROM materials
        """)
        agg_time = time.time() - start

        # Group by query
        start = time.time()
        grouped = self.query("""
            SELECT category, COUNT(*) as count
            FROM materials
            GROUP BY category
        """)
        group_time = time.time() - start

        return {
            'count_time_ms': count_time * 1000,
            'filter_time_ms': filter_time * 1000,
            'aggregation_time_ms': agg_time * 1000,
            'groupby_time_ms': group_time * 1000,
            'total_materials': int(count.iloc[0, 0]),
        }


def main():
    """Demo: SQL queries on materials database"""

    print("=" * 80)
    print("SQL QUERY INTERFACE - Powered by DuckDB")
    print("=" * 80)

    db = MaterialsSQL()

    # Benchmark
    print("\n📊 Running performance benchmark...")
    benchmark = db.benchmark()
    print(f"\n   Total materials: {benchmark['total_materials']:,}")
    print(f"   Count query: {benchmark['count_time_ms']:.2f} ms")
    print(f"   Filter query: {benchmark['filter_time_ms']:.2f} ms")
    print(f"   Aggregation: {benchmark['aggregation_time_ms']:.2f} ms")
    print(f"   Group by: {benchmark['groupby_time_ms']:.2f} ms")

    # Example 1: Lightweight strong materials
    print("\n" + "=" * 80)
    print("🔍 SQL Query Example 1: Lightweight + High Strength")
    print("=" * 80)

    results = db.lightweight_strong(density_max=500, strength_min=50)
    print(f"\n✅ Found {len(results)} materials")
    print("\nTop 10 Results:")
    print(results.head(10))

    # Example 2: Statistics by category
    print("\n" + "=" * 80)
    print("📊 SQL Query Example 2: Statistics by Category")
    print("=" * 80)

    stats = db.statistics_by_category()
    print("\n", stats)

    # Example 3: Custom SQL
    print("\n" + "=" * 80)
    print("🔧 SQL Query Example 3: Custom SQL")
    print("=" * 80)

    custom = db.query("""
        SELECT
            name,
            density_kg_m3,
            tensile_strength,
            (tensile_strength / density_kg_m3) as strength_to_weight_ratio
        FROM materials
        WHERE density_kg_m3 > 0
        ORDER BY strength_to_weight_ratio DESC
        LIMIT 10
    """)

    print("\nTop 10 materials by strength-to-weight ratio:")
    print(custom)

    # Example 4: Property ranges
    print("\n" + "=" * 80)
    print("📈 Property Ranges")
    print("=" * 80)

    ranges = db.property_ranges()
    print(f"\n   Density: {ranges['min_density']:.2f} - {ranges['max_density']:.2f} kg/m³")
    print(f"   Strength: {ranges['min_strength']:.2f} - {ranges['max_strength']:.2f} MPa")
    print(f"   Thermal Conductivity: {ranges['min_conductivity']:.2f} - {ranges['max_conductivity']:.2f} W/(m·K)")

    print("\n" + "=" * 80)
    print("🚀 DuckDB enables SQL queries on 14GB+ databases instantly!")
    print("=" * 80)


if __name__ == "__main__":
    main()
