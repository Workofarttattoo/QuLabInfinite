"""
Product Definition Module
=========================

Core data structures for defining virtual products across multiple disciplines.
Supports mechanical, electrical, systems, and software design integration.

Copyright (c) Joshua Hendricks Cole (DBA: Corporation of Light)
PATENT PENDING - All Rights Reserved
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import uuid
import json
import hashlib


class DesignDiscipline(Enum):
    """Design disciplines supported by the VPD system."""
    MECHANICAL = auto()
    ELECTRICAL = auto()
    ELECTRONIC = auto()
    SOFTWARE = auto()
    SYSTEMS = auto()
    THERMAL = auto()
    FLUID = auto()
    STRUCTURAL = auto()
    OPTICAL = auto()
    ELECTROMAGNETIC = auto()
    CHEMICAL = auto()
    BIOLOGICAL = auto()
    QUANTUM = auto()


class ComponentType(Enum):
    """Types of components in a product definition."""
    ASSEMBLY = auto()
    PART = auto()
    SUBASSEMBLY = auto()
    FASTENER = auto()
    WIRE = auto()
    PCB = auto()
    IC = auto()
    CONNECTOR = auto()
    SENSOR = auto()
    ACTUATOR = auto()
    HOUSING = auto()
    INTERFACE = auto()
    SOFTWARE_MODULE = auto()
    FIRMWARE = auto()
    DOCUMENTATION = auto()
    SPECIFICATION = auto()


class LifecycleState(Enum):
    """Product lifecycle states."""
    CONCEPT = auto()
    DESIGN = auto()
    PROTOTYPE = auto()
    VALIDATION = auto()
    PRODUCTION = auto()
    MAINTENANCE = auto()
    END_OF_LIFE = auto()


class ApprovalStatus(Enum):
    """Design approval status."""
    DRAFT = auto()
    IN_REVIEW = auto()
    APPROVED = auto()
    REJECTED = auto()
    RELEASED = auto()
    OBSOLETE = auto()


@dataclass
class DesignParameter:
    """A design parameter with constraints and metadata."""
    name: str
    value: Any
    unit: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    tolerance: Optional[float] = None
    description: str = ""
    locked: bool = False
    discipline: Optional[DesignDiscipline] = None

    def validate(self) -> bool:
        """Validate the parameter is within constraints."""
        if self.locked:
            return True
        if isinstance(self.value, (int, float)):
            if self.min_value is not None and self.value < self.min_value:
                return False
            if self.max_value is not None and self.value > self.max_value:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'tolerance': self.tolerance,
            'description': self.description,
            'locked': self.locked,
            'discipline': self.discipline.name if self.discipline else None
        }


@dataclass
class DesignInterface:
    """Interface definition between components/systems."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    interface_type: str = ""  # mechanical, electrical, data, thermal, etc.
    source_component: str = ""
    target_component: str = ""
    protocol: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    validated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'interface_type': self.interface_type,
            'source_component': self.source_component,
            'target_component': self.target_component,
            'protocol': self.protocol,
            'parameters': self.parameters,
            'constraints': self.constraints,
            'validated': self.validated
        }


@dataclass
class Component:
    """A component in the product definition."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    part_number: str = ""
    component_type: ComponentType = ComponentType.PART
    discipline: DesignDiscipline = DesignDiscipline.MECHANICAL
    description: str = ""
    version: str = "1.0"
    revision: str = "A"

    # Physical properties
    mass: float = 0.0  # kg
    volume: float = 0.0  # m³
    material: str = ""
    dimensions: Dict[str, float] = field(default_factory=dict)  # length, width, height

    # Cost & sourcing
    unit_cost: float = 0.0
    currency: str = "USD"
    lead_time_days: int = 0
    supplier: str = ""
    make_or_buy: str = "buy"  # make, buy, or configure

    # Design data
    parameters: Dict[str, DesignParameter] = field(default_factory=dict)
    interfaces: List[DesignInterface] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)

    # Lifecycle
    status: ApprovalStatus = ApprovalStatus.DRAFT
    created_date: datetime = field(default_factory=datetime.now)
    modified_date: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    modified_by: str = ""

    # Hierarchy
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)

    # Files and references
    cad_files: List[str] = field(default_factory=list)
    documents: List[str] = field(default_factory=list)
    simulations: List[str] = field(default_factory=list)

    def add_parameter(self, param: DesignParameter):
        """Add a design parameter."""
        self.parameters[param.name] = param
        self.modified_date = datetime.now()

    def add_interface(self, interface: DesignInterface):
        """Add an interface."""
        interface.source_component = self.id
        self.interfaces.append(interface)
        self.modified_date = datetime.now()

    def calculate_hash(self) -> str:
        """Calculate a hash of the component definition."""
        data = f"{self.part_number}:{self.version}:{self.revision}:{json.dumps(self.parameters, default=str)}"
        return hashlib.md5(data.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'part_number': self.part_number,
            'component_type': self.component_type.name,
            'discipline': self.discipline.name,
            'description': self.description,
            'version': self.version,
            'revision': self.revision,
            'mass': self.mass,
            'volume': self.volume,
            'material': self.material,
            'dimensions': self.dimensions,
            'unit_cost': self.unit_cost,
            'currency': self.currency,
            'lead_time_days': self.lead_time_days,
            'supplier': self.supplier,
            'make_or_buy': self.make_or_buy,
            'parameters': {k: v.to_dict() for k, v in self.parameters.items()},
            'interfaces': [i.to_dict() for i in self.interfaces],
            'requirements': self.requirements,
            'status': self.status.name,
            'created_date': self.created_date.isoformat(),
            'modified_date': self.modified_date.isoformat(),
            'created_by': self.created_by,
            'modified_by': self.modified_by,
            'parent_id': self.parent_id,
            'children_ids': self.children_ids,
            'cad_files': self.cad_files,
            'documents': self.documents,
            'simulations': self.simulations
        }


@dataclass
class DesignVariant:
    """A variant of a product design with specific configurations."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    base_product_id: str = ""
    description: str = ""

    # Variant configuration
    configuration_options: Dict[str, Any] = field(default_factory=dict)
    included_components: Set[str] = field(default_factory=set)
    excluded_components: Set[str] = field(default_factory=set)
    parameter_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Market/customer
    target_market: str = ""
    customer_segment: str = ""
    region: str = ""

    # Cost analysis
    estimated_cost: float = 0.0
    target_price: float = 0.0
    margin_percent: float = 0.0

    # Status
    status: ApprovalStatus = ApprovalStatus.DRAFT
    lifecycle_state: LifecycleState = LifecycleState.CONCEPT

    def apply_override(self, component_id: str, parameter_name: str, new_value: Any):
        """Apply a parameter override for a component."""
        if component_id not in self.parameter_overrides:
            self.parameter_overrides[component_id] = {}
        self.parameter_overrides[component_id][parameter_name] = new_value

    def calculate_variant_cost(self, components: Dict[str, Component]) -> float:
        """Calculate the total cost for this variant."""
        total = 0.0
        for comp_id in self.included_components:
            if comp_id in components:
                total += components[comp_id].unit_cost
        return total

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'base_product_id': self.base_product_id,
            'description': self.description,
            'configuration_options': self.configuration_options,
            'included_components': list(self.included_components),
            'excluded_components': list(self.excluded_components),
            'parameter_overrides': self.parameter_overrides,
            'target_market': self.target_market,
            'customer_segment': self.customer_segment,
            'region': self.region,
            'estimated_cost': self.estimated_cost,
            'target_price': self.target_price,
            'margin_percent': self.margin_percent,
            'status': self.status.name,
            'lifecycle_state': self.lifecycle_state.name
        }


@dataclass
class ProductDefinition:
    """
    Complete virtual product definition integrating all design disciplines.

    This is the core data structure for representing a holistic product
    across mechanical, electrical, systems, and software domains.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    product_family: str = ""
    description: str = ""
    version: str = "1.0"
    revision: str = "A"

    # Components organized by discipline
    components: Dict[str, Component] = field(default_factory=dict)

    # System interfaces
    interfaces: List[DesignInterface] = field(default_factory=list)

    # Variants
    variants: Dict[str, DesignVariant] = field(default_factory=dict)

    # Requirements traceability
    requirements: Dict[str, List[str]] = field(default_factory=dict)  # req_id -> component_ids

    # Design rules and constraints
    design_rules: List[Dict[str, Any]] = field(default_factory=list)

    # Lifecycle
    lifecycle_state: LifecycleState = LifecycleState.CONCEPT
    status: ApprovalStatus = ApprovalStatus.DRAFT
    created_date: datetime = field(default_factory=datetime.now)
    modified_date: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    modified_by: str = ""

    # Metadata
    tags: List[str] = field(default_factory=list)
    custom_attributes: Dict[str, Any] = field(default_factory=dict)

    def add_component(self, component: Component, parent_id: Optional[str] = None):
        """Add a component to the product definition."""
        if parent_id:
            component.parent_id = parent_id
            if parent_id in self.components:
                self.components[parent_id].children_ids.append(component.id)

        self.components[component.id] = component
        self.modified_date = datetime.now()

    def remove_component(self, component_id: str):
        """Remove a component and update hierarchy."""
        if component_id in self.components:
            component = self.components[component_id]

            # Remove from parent
            if component.parent_id and component.parent_id in self.components:
                parent = self.components[component.parent_id]
                if component_id in parent.children_ids:
                    parent.children_ids.remove(component_id)

            # Remove children recursively
            for child_id in component.children_ids[:]:
                self.remove_component(child_id)

            del self.components[component_id]
            self.modified_date = datetime.now()

    def add_interface(self, interface: DesignInterface):
        """Add a system interface."""
        self.interfaces.append(interface)
        self.modified_date = datetime.now()

    def create_variant(self, name: str, description: str = "") -> DesignVariant:
        """Create a new variant based on the current product."""
        variant = DesignVariant(
            name=name,
            base_product_id=self.id,
            description=description,
            included_components=set(self.components.keys())
        )
        self.variants[variant.id] = variant
        return variant

    def get_components_by_discipline(self, discipline: DesignDiscipline) -> List[Component]:
        """Get all components for a specific discipline."""
        return [c for c in self.components.values() if c.discipline == discipline]

    def get_root_components(self) -> List[Component]:
        """Get top-level components (no parent)."""
        return [c for c in self.components.values() if c.parent_id is None]

    def get_component_tree(self, component_id: str) -> Dict[str, Any]:
        """Get a component and its children as a tree structure."""
        if component_id not in self.components:
            return {}

        component = self.components[component_id]
        tree = component.to_dict()
        tree['children'] = [
            self.get_component_tree(child_id)
            for child_id in component.children_ids
            if child_id in self.components
        ]
        return tree

    def calculate_total_mass(self) -> float:
        """Calculate total product mass."""
        return sum(c.mass for c in self.components.values())

    def calculate_total_cost(self) -> float:
        """Calculate total product cost."""
        return sum(c.unit_cost for c in self.components.values())

    def validate_interfaces(self) -> List[Dict[str, str]]:
        """Validate all interfaces and return issues."""
        issues = []
        for interface in self.interfaces:
            if interface.source_component not in self.components:
                issues.append({
                    'interface': interface.id,
                    'issue': f"Source component '{interface.source_component}' not found"
                })
            if interface.target_component not in self.components:
                issues.append({
                    'interface': interface.id,
                    'issue': f"Target component '{interface.target_component}' not found"
                })
        return issues

    def get_discipline_summary(self) -> Dict[str, int]:
        """Get count of components by discipline."""
        summary = {}
        for component in self.components.values():
            disc_name = component.discipline.name
            summary[disc_name] = summary.get(disc_name, 0) + 1
        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'product_family': self.product_family,
            'description': self.description,
            'version': self.version,
            'revision': self.revision,
            'components': {k: v.to_dict() for k, v in self.components.items()},
            'interfaces': [i.to_dict() for i in self.interfaces],
            'variants': {k: v.to_dict() for k, v in self.variants.items()},
            'requirements': self.requirements,
            'design_rules': self.design_rules,
            'lifecycle_state': self.lifecycle_state.name,
            'status': self.status.name,
            'created_date': self.created_date.isoformat(),
            'modified_date': self.modified_date.isoformat(),
            'created_by': self.created_by,
            'modified_by': self.modified_by,
            'tags': self.tags,
            'custom_attributes': self.custom_attributes,
            'summary': {
                'total_components': len(self.components),
                'total_interfaces': len(self.interfaces),
                'total_variants': len(self.variants),
                'disciplines': self.get_discipline_summary(),
                'total_mass': self.calculate_total_mass(),
                'total_cost': self.calculate_total_cost()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProductDefinition':
        """Create from dictionary."""
        product = cls(
            id=data.get('id', str(uuid.uuid4())[:8]),
            name=data.get('name', ''),
            product_family=data.get('product_family', ''),
            description=data.get('description', ''),
            version=data.get('version', '1.0'),
            revision=data.get('revision', 'A')
        )

        # Load components
        for comp_id, comp_data in data.get('components', {}).items():
            comp = Component(
                id=comp_data['id'],
                name=comp_data['name'],
                part_number=comp_data.get('part_number', ''),
                component_type=ComponentType[comp_data.get('component_type', 'PART')],
                discipline=DesignDiscipline[comp_data.get('discipline', 'MECHANICAL')],
                description=comp_data.get('description', ''),
                version=comp_data.get('version', '1.0'),
                revision=comp_data.get('revision', 'A'),
                mass=comp_data.get('mass', 0.0),
                unit_cost=comp_data.get('unit_cost', 0.0)
            )
            product.components[comp_id] = comp

        return product

    def export_json(self, filepath: str):
        """Export product definition to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def import_json(cls, filepath: str) -> 'ProductDefinition':
        """Import product definition from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
