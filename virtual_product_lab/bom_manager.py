"""
Bill of Materials (BOM) Manager
===============================

Comprehensive BOM management for virtual product development.
Supports multi-level BOMs, costing, sourcing, and variant configurations.

Copyright (c) Joshua Hendricks Cole (DBA: Corporation of Light)
PATENT PENDING - All Rights Reserved
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum, auto
import json
import csv
from io import StringIO

from .product_definition import (
    ProductDefinition, Component, DesignVariant,
    ComponentType, DesignDiscipline
)


class BOMType(Enum):
    """Types of Bill of Materials."""
    ENGINEERING = auto()  # eBOM - as designed
    MANUFACTURING = auto()  # mBOM - as built
    SERVICE = auto()  # sBOM - as maintained
    SALES = auto()  # sales BOM - as sold


class SourceType(Enum):
    """Component sourcing types."""
    MAKE = auto()
    BUY = auto()
    CONFIGURE = auto()
    PHANTOM = auto()  # Not a real part, placeholder


@dataclass
class BOMItem:
    """An item in the Bill of Materials."""
    id: str
    component_id: str
    part_number: str
    name: str
    description: str = ""

    # Hierarchy
    level: int = 0
    parent_id: Optional[str] = None

    # Quantities
    quantity: float = 1.0
    unit_of_measure: str = "EA"  # EA, KG, M, L, etc.

    # Sourcing
    source_type: SourceType = SourceType.BUY
    supplier: str = ""
    supplier_part_number: str = ""
    lead_time_days: int = 0
    minimum_order_qty: float = 1.0

    # Cost
    unit_cost: float = 0.0
    currency: str = "USD"
    extended_cost: float = 0.0

    # Status
    is_optional: bool = False
    is_configurable: bool = False
    effectivity_start: Optional[datetime] = None
    effectivity_end: Optional[datetime] = None
    substitute_items: List[str] = field(default_factory=list)

    # Reference designators (for electronics)
    reference_designators: List[str] = field(default_factory=list)

    # Notes
    notes: str = ""

    def calculate_extended_cost(self):
        """Calculate extended cost based on quantity."""
        self.extended_cost = self.quantity * self.unit_cost

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'component_id': self.component_id,
            'part_number': self.part_number,
            'name': self.name,
            'description': self.description,
            'level': self.level,
            'parent_id': self.parent_id,
            'quantity': self.quantity,
            'unit_of_measure': self.unit_of_measure,
            'source_type': self.source_type.name,
            'supplier': self.supplier,
            'supplier_part_number': self.supplier_part_number,
            'lead_time_days': self.lead_time_days,
            'minimum_order_qty': self.minimum_order_qty,
            'unit_cost': self.unit_cost,
            'currency': self.currency,
            'extended_cost': self.extended_cost,
            'is_optional': self.is_optional,
            'is_configurable': self.is_configurable,
            'effectivity_start': self.effectivity_start.isoformat() if self.effectivity_start else None,
            'effectivity_end': self.effectivity_end.isoformat() if self.effectivity_end else None,
            'substitute_items': self.substitute_items,
            'reference_designators': self.reference_designators,
            'notes': self.notes
        }


@dataclass
class CostRollup:
    """Cost rollup summary for a BOM."""
    total_material_cost: float = 0.0
    total_labor_cost: float = 0.0
    total_overhead_cost: float = 0.0
    total_cost: float = 0.0

    # Breakdown by source
    make_cost: float = 0.0
    buy_cost: float = 0.0

    # Breakdown by discipline
    cost_by_discipline: Dict[str, float] = field(default_factory=dict)

    # Statistics
    item_count: int = 0
    unique_part_count: int = 0
    supplier_count: int = 0
    critical_path_days: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_material_cost': self.total_material_cost,
            'total_labor_cost': self.total_labor_cost,
            'total_overhead_cost': self.total_overhead_cost,
            'total_cost': self.total_cost,
            'make_cost': self.make_cost,
            'buy_cost': self.buy_cost,
            'cost_by_discipline': self.cost_by_discipline,
            'item_count': self.item_count,
            'unique_part_count': self.unique_part_count,
            'supplier_count': self.supplier_count,
            'critical_path_days': self.critical_path_days
        }


class BOMManager:
    """
    Manages Bill of Materials for virtual products.

    Features:
    - Multi-level BOM generation
    - Variant-specific BOMs
    - Cost rollup calculations
    - Sourcing analysis
    - Where-used analysis
    - BOM comparison
    """

    def __init__(self, product: Optional[ProductDefinition] = None):
        """Initialize with optional product definition."""
        self.product = product
        self.bom_items: Dict[str, BOMItem] = {}
        self.bom_type = BOMType.ENGINEERING

        # Labor and overhead rates
        self.labor_rate_per_hour = 75.0
        self.overhead_rate = 0.25  # 25% overhead

    def set_product(self, product: ProductDefinition):
        """Set or change the product definition."""
        self.product = product
        self.bom_items.clear()

    def generate_bom(self, variant_id: Optional[str] = None,
                     bom_type: BOMType = BOMType.ENGINEERING) -> List[BOMItem]:
        """
        Generate a Bill of Materials from the product definition.

        Args:
            variant_id: Optional variant to generate BOM for
            bom_type: Type of BOM to generate

        Returns:
            List of BOM items
        """
        if not self.product:
            raise ValueError("No product definition set")

        self.bom_type = bom_type
        self.bom_items.clear()

        # Get variant configuration if specified
        variant = None
        if variant_id and variant_id in self.product.variants:
            variant = self.product.variants[variant_id]

        # Process root components first
        item_id = 0
        for component in self.product.get_root_components():
            item_id = self._process_component(
                component, level=0, parent_id=None,
                item_id=item_id, variant=variant
            )

        # Calculate extended costs
        for item in self.bom_items.values():
            item.calculate_extended_cost()

        return list(self.bom_items.values())

    def _process_component(self, component: Component, level: int,
                           parent_id: Optional[str], item_id: int,
                           variant: Optional[DesignVariant] = None) -> int:
        """Recursively process a component into BOM items."""
        # Check if component is included in variant
        if variant:
            if component.id in variant.excluded_components:
                return item_id
            if variant.included_components and component.id not in variant.included_components:
                return item_id

        # Create BOM item
        bom_id = f"BOM-{item_id:04d}"

        # Determine source type
        source_map = {
            'make': SourceType.MAKE,
            'buy': SourceType.BUY,
            'configure': SourceType.CONFIGURE
        }
        source_type = source_map.get(component.make_or_buy.lower(), SourceType.BUY)

        bom_item = BOMItem(
            id=bom_id,
            component_id=component.id,
            part_number=component.part_number or f"PN-{component.id}",
            name=component.name,
            description=component.description,
            level=level,
            parent_id=parent_id,
            quantity=1.0,
            source_type=source_type,
            supplier=component.supplier,
            lead_time_days=component.lead_time_days,
            unit_cost=component.unit_cost,
            currency=component.currency
        )

        self.bom_items[bom_id] = bom_item
        item_id += 1

        # Process children
        for child_id in component.children_ids:
            if child_id in self.product.components:
                child = self.product.components[child_id]
                item_id = self._process_component(
                    child, level=level + 1, parent_id=bom_id,
                    item_id=item_id, variant=variant
                )

        return item_id

    def calculate_cost_rollup(self) -> CostRollup:
        """Calculate cost rollup for the current BOM."""
        rollup = CostRollup()

        suppliers = set()
        part_numbers = set()
        max_lead_time = 0

        for item in self.bom_items.values():
            # Material cost
            rollup.total_material_cost += item.extended_cost

            # Source breakdown
            if item.source_type == SourceType.MAKE:
                rollup.make_cost += item.extended_cost
            else:
                rollup.buy_cost += item.extended_cost

            # Statistics
            if item.supplier:
                suppliers.add(item.supplier)
            part_numbers.add(item.part_number)
            max_lead_time = max(max_lead_time, item.lead_time_days)

        # Calculate overhead and labor (simplified)
        rollup.total_labor_cost = rollup.make_cost * 0.3  # 30% labor for make items
        rollup.total_overhead_cost = rollup.total_material_cost * self.overhead_rate

        rollup.total_cost = (
            rollup.total_material_cost +
            rollup.total_labor_cost +
            rollup.total_overhead_cost
        )

        rollup.item_count = len(self.bom_items)
        rollup.unique_part_count = len(part_numbers)
        rollup.supplier_count = len(suppliers)
        rollup.critical_path_days = max_lead_time

        return rollup

    def get_flat_bom(self) -> List[Dict[str, Any]]:
        """Get BOM as a flat list (summarized by part number)."""
        flat: Dict[str, Dict[str, Any]] = {}

        for item in self.bom_items.values():
            pn = item.part_number
            if pn in flat:
                flat[pn]['quantity'] += item.quantity
                flat[pn]['extended_cost'] += item.extended_cost
            else:
                flat[pn] = {
                    'part_number': pn,
                    'name': item.name,
                    'description': item.description,
                    'quantity': item.quantity,
                    'unit_cost': item.unit_cost,
                    'extended_cost': item.extended_cost,
                    'source_type': item.source_type.name,
                    'supplier': item.supplier
                }

        return list(flat.values())

    def get_indented_bom(self) -> List[Dict[str, Any]]:
        """Get BOM as indented list showing hierarchy."""
        result = []
        for item in sorted(self.bom_items.values(), key=lambda x: x.id):
            entry = item.to_dict()
            entry['indent'] = '  ' * item.level
            entry['display_name'] = entry['indent'] + item.name
            result.append(entry)
        return result

    def where_used(self, part_number: str) -> List[Dict[str, Any]]:
        """Find where a part is used in the product."""
        usages = []
        for item in self.bom_items.values():
            if item.part_number == part_number:
                parent_info = None
                if item.parent_id and item.parent_id in self.bom_items:
                    parent = self.bom_items[item.parent_id]
                    parent_info = {
                        'part_number': parent.part_number,
                        'name': parent.name
                    }
                usages.append({
                    'bom_item_id': item.id,
                    'level': item.level,
                    'quantity': item.quantity,
                    'parent': parent_info
                })
        return usages

    def compare_boms(self, other_bom: Dict[str, BOMItem]) -> Dict[str, Any]:
        """Compare current BOM with another BOM."""
        current_parts = {item.part_number for item in self.bom_items.values()}
        other_parts = {item.part_number for item in other_bom.values()}

        added = current_parts - other_parts
        removed = other_parts - current_parts
        common = current_parts & other_parts

        # Check for quantity changes
        quantity_changes = []
        for pn in common:
            curr_qty = sum(i.quantity for i in self.bom_items.values() if i.part_number == pn)
            other_qty = sum(i.quantity for i in other_bom.values() if i.part_number == pn)
            if curr_qty != other_qty:
                quantity_changes.append({
                    'part_number': pn,
                    'old_quantity': other_qty,
                    'new_quantity': curr_qty
                })

        return {
            'added_parts': list(added),
            'removed_parts': list(removed),
            'common_parts': len(common),
            'quantity_changes': quantity_changes
        }

    def get_long_lead_items(self, threshold_days: int = 30) -> List[BOMItem]:
        """Get items with lead times exceeding threshold."""
        return [
            item for item in self.bom_items.values()
            if item.lead_time_days >= threshold_days
        ]

    def get_make_items(self) -> List[BOMItem]:
        """Get all make (manufactured) items."""
        return [
            item for item in self.bom_items.values()
            if item.source_type == SourceType.MAKE
        ]

    def get_buy_items(self) -> List[BOMItem]:
        """Get all buy (purchased) items."""
        return [
            item for item in self.bom_items.values()
            if item.source_type == SourceType.BUY
        ]

    def export_csv(self) -> str:
        """Export BOM to CSV format."""
        output = StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            'Level', 'Part Number', 'Name', 'Description',
            'Quantity', 'UOM', 'Unit Cost', 'Extended Cost',
            'Source', 'Supplier', 'Lead Time (Days)'
        ])

        # Data
        for item in sorted(self.bom_items.values(), key=lambda x: x.id):
            writer.writerow([
                item.level,
                item.part_number,
                item.name,
                item.description,
                item.quantity,
                item.unit_of_measure,
                item.unit_cost,
                item.extended_cost,
                item.source_type.name,
                item.supplier,
                item.lead_time_days
            ])

        return output.getvalue()

    def export_json(self) -> str:
        """Export BOM to JSON format."""
        data = {
            'bom_type': self.bom_type.name,
            'product_id': self.product.id if self.product else None,
            'product_name': self.product.name if self.product else None,
            'generated_date': datetime.now().isoformat(),
            'items': [item.to_dict() for item in self.bom_items.values()],
            'cost_rollup': self.calculate_cost_rollup().to_dict()
        }
        return json.dumps(data, indent=2)

    def get_summary(self) -> Dict[str, Any]:
        """Get BOM summary statistics."""
        rollup = self.calculate_cost_rollup()

        # Count by level
        level_counts = {}
        for item in self.bom_items.values():
            level_counts[item.level] = level_counts.get(item.level, 0) + 1

        # Cost distribution
        cost_dist = {}
        for item in self.bom_items.values():
            if item.extended_cost > 0:
                pct = (item.extended_cost / rollup.total_material_cost * 100
                       if rollup.total_material_cost > 0 else 0)
                if pct >= 1:  # Only items >= 1% of cost
                    cost_dist[item.part_number] = {
                        'name': item.name,
                        'cost': item.extended_cost,
                        'percentage': round(pct, 2)
                    }

        return {
            'bom_type': self.bom_type.name,
            'total_items': len(self.bom_items),
            'level_distribution': level_counts,
            'cost_rollup': rollup.to_dict(),
            'major_cost_drivers': dict(sorted(
                cost_dist.items(),
                key=lambda x: x[1]['cost'],
                reverse=True
            )[:10])
        }
