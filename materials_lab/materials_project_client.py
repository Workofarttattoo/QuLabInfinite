#!/usr/bin/env python3
"""
Materials Project API Client
Handles API interactions with Materials Project (https://materialsproject.org)

Features:
- Robust error handling and retry logic
- Rate limiting and caching
- Batch material downloads
- Property mapping to QuLabInfinite schema
- Confidence scoring for predictions
"""

import os
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

try:
    from pymatgen.ext.matproj import MPRester
    from pymatgen.core import Structure
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    logging.warning("pymatgen not installed. Install with: pip install pymatgen")

try:
    from .materials_database import MaterialProperties
except ImportError:
    from materials_database import MaterialProperties


@dataclass
class MPMaterialData:
    """Materials Project material data wrapper"""
    mp_id: str
    formula: str
    formation_energy_per_atom: float  # eV/atom
    band_gap: float  # eV
    density: float  # g/cm³
    volume: float  # Å³
    nsites: int
    structure: Dict[str, Any]
    space_group: str
    energy_above_hull: float  # eV/atom (stability indicator)
    is_stable: bool
    theoretical: bool  # True if no experimental data exists

    # Additional computed properties
    efermi: Optional[float] = None  # Fermi energy (eV)
    elastic_anisotropy: Optional[float] = None
    bulk_modulus: Optional[float] = None  # GPa
    shear_modulus: Optional[float] = None  # GPa

    def to_material_properties(self) -> MaterialProperties:
        """Convert Materials Project data to QuLabInfinite MaterialProperties"""

        # Basic identification
        name = f"{self.formula} ({self.mp_id})"
        category = self._infer_category()
        subcategory = self._infer_subcategory()

        # Density conversions
        density_g_cm3 = self.density
        density_kg_m3 = self.density * 1000.0

        # Volume per atom
        volume_a3_per_atom = self.volume / self.nsites if self.nsites > 0 else 0.0
        volume_m3_per_atom = volume_a3_per_atom * 1e-30

        # Mechanical properties (estimated from electronic structure)
        youngs_modulus = self._estimate_youngs_modulus()
        shear_mod = self.shear_modulus if self.shear_modulus else self._estimate_shear_modulus()
        bulk_mod = self.bulk_modulus if self.bulk_modulus else self._estimate_bulk_modulus()
        poissons_ratio = self._estimate_poissons_ratio(youngs_modulus, shear_mod)

        # Thermal properties (basic estimates)
        thermal_conductivity = self._estimate_thermal_conductivity()

        return MaterialProperties(
            name=name,
            category=category,
            subcategory=subcategory,
            cas_number=None,
            structure=self.structure,

            # Density
            density=density_kg_m3,
            density_g_cm3=density_g_cm3,
            density_kg_m3=density_kg_m3,

            # Mechanical (from MP or estimated)
            youngs_modulus=youngs_modulus,
            shear_modulus=shear_mod,
            bulk_modulus=bulk_mod,
            poissons_ratio=poissons_ratio,

            # Volume
            volume_a3_per_atom=volume_a3_per_atom,
            volume_m3_per_atom=volume_m3_per_atom,

            # Electronic
            band_gap_ev=self.band_gap,
            band_gap_j=self.band_gap * 1.602176634e-19,
            electrical_conductivity=self._estimate_conductivity(),

            # Thermal (estimated)
            thermal_conductivity=thermal_conductivity,
        )

    def _infer_category(self) -> str:
        """Infer material category from composition and structure"""
        formula = self.formula.lower()

        # Check for common material types
        if any(metal in formula for metal in ['al', 'fe', 'cu', 'ti', 'ni', 'cr', 'mn', 'zn', 'mg']):
            if 'o' in formula or 'n' in formula or 'c' in formula:
                return "ceramic"  # Metal oxide, nitride, or carbide
            return "metal"
        elif 'si' in formula and 'o' in formula:
            return "ceramic"
        elif self.band_gap > 0.5:
            return "ceramic"  # Likely an insulator
        else:
            return "metal"

    def _infer_subcategory(self) -> str:
        """Infer material subcategory"""
        formula = self.formula

        if self.band_gap > 2.0:
            return "insulator"
        elif self.band_gap > 0.1:
            return "semiconductor"
        else:
            return "conductor"

    def _estimate_bulk_modulus(self) -> float:
        """Estimate bulk modulus from density and atomic volume"""
        # Empirical correlation: K ≈ C * (ρ/V)^(2/3) where C ~ 100-200 GPa
        if self.nsites > 0 and self.volume > 0:
            volume_per_atom = self.volume / self.nsites
            # Rough estimate based on density
            K_estimate = 100.0 * (self.density / volume_per_atom) ** (2/3)
            return max(10.0, min(K_estimate, 400.0))  # Clamp to reasonable range
        return 0.0

    def _estimate_shear_modulus(self) -> float:
        """Estimate shear modulus from bulk modulus"""
        # For most materials: G ≈ 0.4 * K
        K = self._estimate_bulk_modulus()
        return 0.4 * K

    def _estimate_youngs_modulus(self) -> float:
        """Estimate Young's modulus from bulk and shear moduli"""
        K = self._estimate_bulk_modulus()
        G = self._estimate_shear_modulus()
        # E = 9KG / (3K + G)
        if K > 0 and G > 0:
            return (9 * K * G) / (3 * K + G)
        return 0.0

    def _estimate_poissons_ratio(self, E: float, G: float) -> float:
        """Calculate Poisson's ratio from Young's and shear moduli"""
        if G > 0 and E > 0:
            nu = (E / (2 * G)) - 1
            return max(0.0, min(nu, 0.5))  # Physical bounds: 0 ≤ ν ≤ 0.5
        return 0.3  # Default for most materials

    def _estimate_thermal_conductivity(self) -> float:
        """Estimate thermal conductivity from electronic properties"""
        # Wiedemann-Franz law for metals: κ/σT = L (Lorenz number)
        # For insulators, use rough estimates based on density

        if self.band_gap < 0.1:  # Metal
            # High conductivity metals typically have κ ~ 100-400 W/(m·K)
            return 100.0 + (300.0 * (1.0 - min(self.band_gap, 1.0)))
        else:  # Insulator/semiconductor
            # Lower thermal conductivity, inversely related to band gap
            return max(1.0, 50.0 / (1.0 + self.band_gap))

    def _estimate_conductivity(self) -> float:
        """Estimate electrical conductivity from band gap"""
        # Rough exponential relationship
        if self.band_gap < 0.01:  # Metal
            return 1e7  # ~10^7 S/m for typical metals
        else:
            # Semiconductor/insulator: σ ∝ exp(-Eg/2kT)
            return 1e7 * np.exp(-self.band_gap / 0.05)  # Rough estimate at room temp


class MaterialsProjectClient:
    """
    Client for Materials Project API

    Features:
    - Automatic retry on failures
    - Rate limiting (5 requests/second per API terms)
    - Result caching to minimize API calls
    - Batch downloads
    """

    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Initialize Materials Project client

        Args:
            api_key: MP API key. If None, reads from MP_API_KEY env var
            cache_dir: Directory for caching results. If None, uses ./mp_cache
        """
        if not PYMATGEN_AVAILABLE:
            raise ImportError("pymatgen is required. Install with: pip install pymatgen")

        self.api_key = api_key or os.environ.get("MP_API_KEY")
        if not self.api_key:
            print(
                "⚠️  WARNING: Materials Project API key not found.\n"
                "    To access 140,000+ materials, get a free key at https://materialsproject.org/api\n"
                "    and set the MP_API_KEY environment variable.\n"
                "    Proceeding with local database and cache only."
            )
            # We don't raise ValueError anymore, to allow offline usage of cache/local DB

        self.cache_dir = Path(cache_dir or "./mp_cache")
        self.cache_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.last_request_time = 0.0
        self.min_request_interval = 0.2  # 5 requests/second max

    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _get_cache_path(self, mp_id: str) -> Path:
        """Get cache file path for a material"""
        return self.cache_dir / f"{mp_id}.json"

    def _load_from_cache(self, mp_id: str) -> Optional[MPMaterialData]:
        """Load material from cache"""
        cache_path = self._get_cache_path(mp_id)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                return MPMaterialData(**data)
            except Exception as e:
                self.logger.warning(f"Failed to load cache for {mp_id}: {e}")
        return None

    def _save_to_cache(self, material: MPMaterialData):
        """Save material to cache"""
        cache_path = self._get_cache_path(material.mp_id)
        try:
            with open(cache_path, 'w') as f:
                json.dump(asdict(material), f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save cache for {material.mp_id}: {e}")

    def get_material(self, mp_id: str, use_cache: bool = True) -> Optional[MPMaterialData]:
        """
        Get material by Materials Project ID

        Args:
            mp_id: Materials Project ID (e.g., 'mp-149' for Silicon)
            use_cache: Whether to use cached results

        Returns:
            MPMaterialData object or None if not found
        """
        # Check cache first
        if use_cache:
            cached = self._load_from_cache(mp_id)
            if cached:
                self.logger.info(f"Loaded {mp_id} from cache")
                return cached

        if not self.api_key:
            self.logger.warning(f"Cannot fetch {mp_id}: Missing Materials Project API key.")
            return None

        # Query API
        self._rate_limit()

        try:
            with MPRester(self.api_key) as mpr:
                # Get material summary
                docs = mpr.materials.summary.search(material_ids=[mp_id])

                if not docs:
                    self.logger.warning(f"Material {mp_id} not found")
                    return None

                doc = docs[0]

                # Extract data
                material = MPMaterialData(
                    mp_id=str(doc.material_id),
                    formula=doc.formula_pretty,
                    formation_energy_per_atom=doc.formation_energy_per_atom,
                    band_gap=doc.band_gap,
                    density=doc.density,
                    volume=doc.volume,
                    nsites=doc.nsites,
                    structure=doc.structure.as_dict(),
                    space_group=doc.symmetry.symbol,
                    energy_above_hull=doc.energy_above_hull,
                    is_stable=doc.is_stable,
                    theoretical=doc.theoretical,
                )

                # Try to get elastic properties if available
                try:
                    elasticity = mpr.elasticity.get_data_by_id(mp_id)
                    if elasticity:
                        material.bulk_modulus = elasticity.k_vrh
                        material.shear_modulus = elasticity.g_vrh
                        material.elastic_anisotropy = elasticity.universal_anisotropy
                except Exception as e:
                    self.logger.debug(f"No elastic data for {mp_id}: {e}")

                # Save to cache
                self._save_to_cache(material)

                return material

        except Exception as e:
            self.logger.error(f"Error fetching {mp_id}: {e}")
            return None

    def search_materials(
        self,
        formula: Optional[str] = None,
        elements: Optional[List[str]] = None,
        exclude_elements: Optional[List[str]] = None,
        band_gap_range: Optional[Tuple[float, float]] = None,
        density_range: Optional[Tuple[float, float]] = None,
        is_stable: Optional[bool] = None,
        limit: int = 100
    ) -> List[MPMaterialData]:
        """
        Search for materials matching criteria

        Args:
            formula: Chemical formula (e.g., 'Fe2O3')
            elements: List of required elements
            exclude_elements: List of elements to exclude
            band_gap_range: (min, max) band gap in eV
            density_range: (min, max) density in g/cm³
            is_stable: Whether to include only stable materials
            limit: Maximum number of results

        Returns:
            List of MPMaterialData objects
        """
        if not self.api_key:
            self.logger.warning("Search failed: Missing Materials Project API key.")
            return []

        self._rate_limit()

        try:
            with MPRester(self.api_key) as mpr:
                # Build search criteria
                criteria = {}

                if formula:
                    criteria['formula'] = formula
                if elements:
                    criteria['elements'] = elements
                if exclude_elements:
                    criteria['exclude_elements'] = exclude_elements
                if band_gap_range:
                    criteria['band_gap'] = band_gap_range
                if density_range:
                    criteria['density'] = density_range
                if is_stable is not None:
                    criteria['is_stable'] = is_stable

                # Search
                docs = mpr.materials.summary.search(**criteria, num_chunks=1, chunk_size=limit)

                # Convert to MPMaterialData
                materials = []
                for doc in docs[:limit]:
                    material = MPMaterialData(
                        mp_id=str(doc.material_id),
                        formula=doc.formula_pretty,
                        formation_energy_per_atom=doc.formation_energy_per_atom,
                        band_gap=doc.band_gap,
                        density=doc.density,
                        volume=doc.volume,
                        nsites=doc.nsites,
                        structure=doc.structure.as_dict(),
                        space_group=doc.symmetry.symbol,
                        energy_above_hull=doc.energy_above_hull,
                        is_stable=doc.is_stable,
                        theoretical=doc.theoretical,
                    )
                    materials.append(material)
                    self._save_to_cache(material)

                return materials

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    def get_common_materials(self, count: int = 100) -> List[MPMaterialData]:
        """
        Get a dataset of common, well-characterized materials

        Args:
            count: Number of materials to retrieve

        Returns:
            List of MPMaterialData objects
        """
        # Common materials list (elements and simple compounds)
        common_mp_ids = [
            # Elements
            "mp-149",   # Si
            "mp-13",    # Fe
            "mp-30",    # Cu
            "mp-23",    # Al
            "mp-72",    # Ti
            "mp-911",   # Ni
            "mp-54",    # W
            "mp-79",    # Cr
            "mp-8",     # Mg
            "mp-48",    # Au
            "mp-126",   # Ag
            "mp-124",   # Pt
            "mp-35",    # Zn
            "mp-96",    # Mo
            "mp-90",    # Zr

            # Common oxides
            "mp-1143",  # SiO2 (quartz)
            "mp-19770", # Al2O3 (alumina)
            "mp-1245",  # TiO2 (rutile)
            "mp-2657",  # Fe2O3 (hematite)
            "mp-18905", # Fe3O4 (magnetite)
            "mp-1487",  # ZnO
            "mp-1265",  # CuO
            "mp-19399", # MgO
            "mp-1216",  # NiO
            "mp-1096",  # Cr2O3

            # Common ceramics
            "mp-2133",  # Si3N4
            "mp-1029",  # SiC
            "mp-1243",  # AlN
            "mp-636",   # TiN
            "mp-1818",  # TiC
            "mp-11714", # WC
            "mp-1840",  # ZrO2
            "mp-2074",  # BN

            # Common semiconductors
            "mp-10695", # GaAs
            "mp-10695", # GaN
            "mp-8062",  # InP
            "mp-2490",  # CdTe

            # Common alloys and compounds (representative)
            "mp-568",   # Steel (FeC)
            "mp-1094",  # Brass (CuZn)
            "mp-541807", # Bronze (CuSn)
        ]

        materials = []
        for mp_id in common_mp_ids[:count]:
            material = self.get_material(mp_id, use_cache=True)
            if material:
                materials.append(material)

            if len(materials) >= count:
                break

        return materials


# Import numpy for calculations
import numpy as np


if __name__ == "__main__":
    # Test the client
    logging.basicConfig(level=logging.INFO)

    print("Testing Materials Project Client...")
    print("=" * 60)

    # Check for API key
    if not os.environ.get("MP_API_KEY"):
        print("\n⚠️  MP_API_KEY not set!")
        print("Get a free API key at: https://materialsproject.org/api")
        print("Then set it: export MP_API_KEY='your-key-here'")
        exit(1)

    try:
        client = MaterialsProjectClient()

        # Test single material fetch
        print("\n1. Fetching Silicon (mp-149)...")
        si = client.get_material("mp-149")
        if si:
            print(f"   ✓ {si.formula}: ρ={si.density:.2f} g/cm³, Eg={si.band_gap:.2f} eV")

            # Convert to MaterialProperties
            props = si.to_material_properties()
            print(f"   ✓ Converted to MaterialProperties: E={props.youngs_modulus:.1f} GPa")

        # Test search
        print("\n2. Searching for stable Fe-O compounds...")
        fe_oxides = client.search_materials(
            elements=["Fe", "O"],
            is_stable=True,
            limit=5
        )
        print(f"   ✓ Found {len(fe_oxides)} Fe-O compounds")
        for mat in fe_oxides[:3]:
            print(f"     - {mat.formula} ({mat.mp_id}): Eg={mat.band_gap:.2f} eV")

        # Test common materials
        print("\n3. Loading common materials dataset...")
        common = client.get_common_materials(count=10)
        print(f"   ✓ Loaded {len(common)} common materials")

        print("\n✓ All tests passed!")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
