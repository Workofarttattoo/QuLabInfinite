# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Metabolomics Laboratory Module
Metabolic pathway analysis, flux balance analysis, biomarker discovery, disease metabolism
"""

__all__ = []

try:
    from .metabolomics_engine import MetabolomicsEngine
    __all__.append("MetabolomicsEngine")
except ImportError:
    MetabolomicsEngine = None

try:
    from .pathway_analyzer import PathwayAnalyzer
    __all__.append("PathwayAnalyzer")
except ImportError:
    PathwayAnalyzer = None

try:
    from .flux_balance import FluxBalanceAnalyzer
    __all__.append("FluxBalanceAnalyzer")
except ImportError:
    FluxBalanceAnalyzer = None

try:
    from .biomarker_discovery import BiomarkerDiscovery
    __all__.append("BiomarkerDiscovery")
except ImportError:
    BiomarkerDiscovery = None

__version__ = '1.0.0'
