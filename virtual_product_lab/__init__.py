"""
Virtual Product Development Laboratory
======================================

A comprehensive virtual product development and lifecycle management system
inspired by enterprise PLM platforms like ENOVIA/3DEXPERIENCE.

This module provides:
- Multi-discipline design integration (mechanical, electrical, systems, software)
- Product configuration and variant management
- Bill of Materials (BOM) management
- Value network collaboration tools
- Design cycle optimization
- Virtual product definitions

Copyright (c) Joshua Hendricks Cole (DBA: Corporation of Light)
PATENT PENDING - All Rights Reserved
"""

from .vpd_lab import VirtualProductLab
from .product_definition import (
    ProductDefinition,
    DesignDiscipline,
    ComponentType,
    DesignVariant
)
from .bom_manager import BOMManager, BOMItem
from .collaboration import CollaborationHub, Stakeholder
from .design_optimizer import DesignOptimizer

__all__ = [
    'VirtualProductLab',
    'ProductDefinition',
    'DesignDiscipline',
    'ComponentType',
    'DesignVariant',
    'BOMManager',
    'BOMItem',
    'CollaborationHub',
    'Stakeholder',
    'DesignOptimizer'
]

__version__ = '1.0.0'
