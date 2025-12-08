"""
Virtual Product Development Laboratory
======================================

Main laboratory class for comprehensive virtual product development.
Integrates multi-discipline design, BOM management, collaboration, and optimization.

Inspired by enterprise PLM platforms like ENOVIA/3DEXPERIENCE, this lab provides
a holistic approach to product development that connects mechanical, electrical,
systems, and software designs into unified product definitions.

Copyright (c) Joshua Hendricks Cole (DBA: Corporation of Light)
PATENT PENDING - All Rights Reserved
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import uuid

from core.base_lab import BaseLab
from .product_definition import (
    ProductDefinition, Component, DesignVariant, DesignParameter,
    DesignInterface, DesignDiscipline, ComponentType, LifecycleState,
    ApprovalStatus
)
from .bom_manager import BOMManager, BOMType, CostRollup
from .collaboration import (
    CollaborationHub, Stakeholder, StakeholderRole,
    CollaborationTask, TaskPriority, DesignReview, ChangeRequest, ChangeType
)
from .design_optimizer import (
    DesignOptimizer, DesignVariable, OptimizationConstraint,
    OptimizationObjective, OptimizationMethod
)


class VirtualProductLab(BaseLab):
    """
    Virtual Product Development Laboratory.

    A comprehensive simulation laboratory for virtual product development
    that enables multi-discipline design integration, real-time collaboration,
    and design optimization across the entire value network.

    Key Capabilities:
    - Multi-discipline design integration (mechanical, electrical, systems, software)
    - Product configuration and variant management
    - Bill of Materials (BOM) management with cost rollup
    - Value network collaboration and change management
    - Multi-objective design optimization
    - Trade-off studies and sensitivity analysis
    - Design space exploration

    This lab transforms the innovation process by providing a unified platform
    for connecting all stakeholders and design domains, reducing design cycle
    times and revealing collaboration opportunities early in development.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Virtual Product Development Laboratory.

        Args:
            config: Configuration dictionary with optional settings:
                - project_name: Name of the project
                - organization: Organization name
                - enable_collaboration: Enable collaboration features
                - enable_optimization: Enable optimization features
        """
        super().__init__(config)

        self.config = config or {}
        self.project_name = self.config.get('project_name', 'New Product')
        self.organization = self.config.get('organization', 'QuLab')

        # Core components
        self.products: Dict[str, ProductDefinition] = {}
        self.active_product_id: Optional[str] = None

        # BOM management
        self.bom_manager = BOMManager()

        # Collaboration
        self.collaboration_hub = CollaborationHub()

        # Optimization
        self.optimizer = DesignOptimizer()

        # Session tracking
        self.session_id = str(uuid.uuid4())[:8]
        self.session_start = datetime.now()

        print(f"[info] VirtualProductLab initialized - Session: {self.session_id}")

    # =========================================================================
    # BaseLab Interface Implementation
    # =========================================================================

    def run_experiment(self, experiment_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a virtual product development experiment.

        Supported experiment types:
        - 'create_product': Create a new product definition
        - 'add_component': Add a component to a product
        - 'create_variant': Create a product variant
        - 'generate_bom': Generate Bill of Materials
        - 'optimize_design': Run design optimization
        - 'trade_off_study': Perform trade-off analysis
        - 'sensitivity_analysis': Analyze parameter sensitivities
        - 'explore_design_space': Explore the design space

        Args:
            experiment_spec: Dictionary with 'type' and type-specific parameters

        Returns:
            Dictionary with experiment results
        """
        exp_type = experiment_spec.get('type', '')

        try:
            if exp_type == 'create_product':
                return self._exp_create_product(experiment_spec)
            elif exp_type == 'add_component':
                return self._exp_add_component(experiment_spec)
            elif exp_type == 'create_variant':
                return self._exp_create_variant(experiment_spec)
            elif exp_type == 'generate_bom':
                return self._exp_generate_bom(experiment_spec)
            elif exp_type == 'optimize_design':
                return self._exp_optimize_design(experiment_spec)
            elif exp_type == 'trade_off_study':
                return self._exp_trade_off_study(experiment_spec)
            elif exp_type == 'sensitivity_analysis':
                return self._exp_sensitivity_analysis(experiment_spec)
            elif exp_type == 'explore_design_space':
                return self._exp_explore_design_space(experiment_spec)
            else:
                return {
                    'success': False,
                    'error': f"Unknown experiment type: {exp_type}",
                    'available_types': [
                        'create_product', 'add_component', 'create_variant',
                        'generate_bom', 'optimize_design', 'trade_off_study',
                        'sensitivity_analysis', 'explore_design_space'
                    ]
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'experiment_type': exp_type
            }

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the laboratory."""
        return {
            'lab_name': 'Virtual Product Development Laboratory',
            'version': '1.0.0',
            'session_id': self.session_id,
            'session_start': self.session_start.isoformat(),
            'project_name': self.project_name,
            'organization': self.organization,
            'products': {
                'count': len(self.products),
                'active': self.active_product_id,
                'list': [
                    {'id': p.id, 'name': p.name, 'components': len(p.components)}
                    for p in self.products.values()
                ]
            },
            'collaboration': self.collaboration_hub.get_project_dashboard(),
            'optimizer': self.optimizer.get_status()
        }

    def get_capabilities(self) -> Dict[str, Any]:
        """Get the capabilities of the laboratory."""
        return {
            'name': 'Virtual Product Development Laboratory',
            'description': (
                'Comprehensive virtual product development platform for '
                'multi-discipline design integration, collaboration, and optimization.'
            ),
            'capabilities': {
                'product_definition': {
                    'multi_discipline_design': True,
                    'disciplines': [d.name for d in DesignDiscipline],
                    'component_types': [t.name for t in ComponentType],
                    'variant_management': True,
                    'requirements_traceability': True
                },
                'bom_management': {
                    'bom_types': [t.name for t in BOMType],
                    'cost_rollup': True,
                    'where_used_analysis': True,
                    'bom_comparison': True,
                    'export_formats': ['csv', 'json']
                },
                'collaboration': {
                    'stakeholder_roles': [r.name for r in StakeholderRole],
                    'task_management': True,
                    'design_reviews': True,
                    'change_management': True,
                    'notifications': True
                },
                'optimization': {
                    'methods': [m.name for m in OptimizationMethod],
                    'multi_objective': True,
                    'constraint_handling': True,
                    'trade_off_studies': True,
                    'sensitivity_analysis': True,
                    'design_space_exploration': True
                }
            },
            'use_cases': [
                'New product development',
                'Product variant configuration',
                'Cost optimization',
                'Design trade-off analysis',
                'Multi-team collaboration',
                'Engineering change management',
                'Supplier integration'
            ]
        }

    # =========================================================================
    # Product Management
    # =========================================================================

    def create_product(self, name: str, description: str = "",
                       product_family: str = "") -> ProductDefinition:
        """Create a new product definition."""
        product = ProductDefinition(
            name=name,
            description=description,
            product_family=product_family,
            created_by=self.organization
        )

        self.products[product.id] = product
        self.active_product_id = product.id
        self.collaboration_hub.product_id = product.id

        print(f"[info] Created product: {name} (ID: {product.id})")
        return product

    def get_product(self, product_id: Optional[str] = None) -> Optional[ProductDefinition]:
        """Get a product by ID or the active product."""
        pid = product_id or self.active_product_id
        return self.products.get(pid) if pid else None

    def add_component(self, product_id: str, name: str,
                      component_type: ComponentType = ComponentType.PART,
                      discipline: DesignDiscipline = DesignDiscipline.MECHANICAL,
                      parent_id: Optional[str] = None,
                      **kwargs) -> Optional[Component]:
        """Add a component to a product."""
        product = self.products.get(product_id)
        if not product:
            return None

        component = Component(
            name=name,
            component_type=component_type,
            discipline=discipline,
            **kwargs
        )

        product.add_component(component, parent_id)
        print(f"[info] Added component: {name} to product {product.name}")
        return component

    def create_variant(self, product_id: str, name: str,
                       description: str = "") -> Optional[DesignVariant]:
        """Create a product variant."""
        product = self.products.get(product_id)
        if not product:
            return None

        variant = product.create_variant(name, description)
        print(f"[info] Created variant: {name} for product {product.name}")
        return variant

    # =========================================================================
    # BOM Management
    # =========================================================================

    def generate_bom(self, product_id: str, variant_id: Optional[str] = None,
                     bom_type: BOMType = BOMType.ENGINEERING) -> Dict[str, Any]:
        """Generate Bill of Materials for a product."""
        product = self.products.get(product_id)
        if not product:
            return {'success': False, 'error': 'Product not found'}

        self.bom_manager.set_product(product)
        items = self.bom_manager.generate_bom(variant_id, bom_type)
        rollup = self.bom_manager.calculate_cost_rollup()

        return {
            'success': True,
            'product_id': product_id,
            'variant_id': variant_id,
            'bom_type': bom_type.name,
            'item_count': len(items),
            'cost_rollup': rollup.to_dict(),
            'summary': self.bom_manager.get_summary()
        }

    def get_bom_csv(self) -> str:
        """Export current BOM as CSV."""
        return self.bom_manager.export_csv()

    def get_bom_json(self) -> str:
        """Export current BOM as JSON."""
        return self.bom_manager.export_json()

    # =========================================================================
    # Collaboration
    # =========================================================================

    def add_stakeholder(self, name: str, email: str,
                        role: StakeholderRole,
                        organization: str = "") -> Stakeholder:
        """Add a stakeholder to the collaboration hub."""
        stakeholder = Stakeholder(
            name=name,
            email=email,
            role=role,
            organization=organization or self.organization
        )
        self.collaboration_hub.add_stakeholder(stakeholder)
        return stakeholder

    def create_task(self, title: str, description: str,
                    assigned_to: List[str],
                    priority: TaskPriority = TaskPriority.MEDIUM) -> CollaborationTask:
        """Create a collaboration task."""
        return self.collaboration_hub.create_task(
            title=title,
            description=description,
            assigned_to=assigned_to,
            priority=priority
        )

    def schedule_review(self, title: str, review_type: str,
                        reviewers: List[str], presenter: str,
                        components: List[str] = None) -> DesignReview:
        """Schedule a design review."""
        return self.collaboration_hub.create_review(
            title=title,
            review_type=review_type,
            reviewers=reviewers,
            presenter=presenter,
            components=components
        )

    def submit_change_request(self, title: str, description: str,
                              justification: str,
                              change_type: ChangeType = ChangeType.ENGINEERING_CHANGE,
                              affected_components: List[str] = None) -> ChangeRequest:
        """Submit an engineering change request."""
        return self.collaboration_hub.submit_change_request(
            title=title,
            change_type=change_type,
            description=description,
            justification=justification,
            requested_by=self.organization,
            affected_components=affected_components
        )

    # =========================================================================
    # Optimization
    # =========================================================================

    def setup_optimization(self, variables: List[Dict[str, Any]],
                           objectives: List[Dict[str, Any]],
                           constraints: List[Dict[str, Any]] = None):
        """
        Setup optimization problem.

        Args:
            variables: List of dicts with name, min, max, current values
            objectives: List of dicts with name, expression, type (minimize/maximize)
            constraints: List of dicts with name, expression, limit, type
        """
        self.optimizer = DesignOptimizer()

        for var in variables:
            self.optimizer.add_variable(DesignVariable(
                name=var['name'],
                current_value=var.get('current', (var['min'] + var['max']) / 2),
                min_value=var['min'],
                max_value=var['max'],
                unit=var.get('unit', ''),
                description=var.get('description', '')
            ))

        for obj in objectives:
            obj_type = OptimizationObjective.MINIMIZE
            if obj.get('type', '').lower() == 'maximize':
                obj_type = OptimizationObjective.MAXIMIZE
            elif obj.get('type', '').lower() == 'target':
                obj_type = OptimizationObjective.TARGET

            self.optimizer.add_objective(
                name=obj['name'],
                expression=obj['expression'],
                objective_type=obj_type,
                target_value=obj.get('target'),
                weight=obj.get('weight', 1.0)
            )

        if constraints:
            for con in constraints:
                self.optimizer.add_constraint(OptimizationConstraint(
                    name=con['name'],
                    expression=con['expression'],
                    limit_value=con['limit'],
                    constraint_type=con.get('type', '<=')
                ))

    def run_optimization(self, method: str = 'genetic') -> Dict[str, Any]:
        """Run optimization with specified method."""
        method_map = {
            'genetic': OptimizationMethod.GENETIC_ALGORITHM,
            'annealing': OptimizationMethod.SIMULATED_ANNEALING,
            'grid': OptimizationMethod.GRID_SEARCH
        }

        opt_method = method_map.get(method.lower(), OptimizationMethod.GENETIC_ALGORITHM)
        result = self.optimizer.optimize(opt_method)

        return {
            'success': result.success,
            'best_values': result.best_values,
            'best_objectives': result.best_objectives,
            'iterations': result.iterations,
            'computation_time': result.computation_time_seconds,
            'method': result.method_used
        }

    def run_trade_off_study(self, objective_x: str, objective_y: str,
                            num_points: int = 20) -> Dict[str, Any]:
        """Run a trade-off study between two objectives."""
        study = self.optimizer.trade_off_study(objective_x, objective_y, num_points)
        return study.to_dict()

    def run_sensitivity_analysis(self, solution: Dict[str, float]) -> Dict[str, Any]:
        """Run sensitivity analysis around a solution."""
        return self.optimizer.sensitivity_analysis(solution)

    # =========================================================================
    # Experiment Implementations
    # =========================================================================

    def _exp_create_product(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Create product experiment."""
        product = self.create_product(
            name=spec.get('name', 'New Product'),
            description=spec.get('description', ''),
            product_family=spec.get('product_family', '')
        )
        return {
            'success': True,
            'product_id': product.id,
            'product': product.to_dict()
        }

    def _exp_add_component(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Add component experiment."""
        product_id = spec.get('product_id') or self.active_product_id
        if not product_id:
            return {'success': False, 'error': 'No product specified'}

        component = self.add_component(
            product_id=product_id,
            name=spec.get('name', 'Component'),
            component_type=ComponentType[spec.get('component_type', 'PART')],
            discipline=DesignDiscipline[spec.get('discipline', 'MECHANICAL')],
            parent_id=spec.get('parent_id'),
            description=spec.get('description', ''),
            mass=spec.get('mass', 0.0),
            unit_cost=spec.get('unit_cost', 0.0)
        )

        if component:
            return {
                'success': True,
                'component_id': component.id,
                'component': component.to_dict()
            }
        return {'success': False, 'error': 'Failed to add component'}

    def _exp_create_variant(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Create variant experiment."""
        product_id = spec.get('product_id') or self.active_product_id
        if not product_id:
            return {'success': False, 'error': 'No product specified'}

        variant = self.create_variant(
            product_id=product_id,
            name=spec.get('name', 'Variant'),
            description=spec.get('description', '')
        )

        if variant:
            return {
                'success': True,
                'variant_id': variant.id,
                'variant': variant.to_dict()
            }
        return {'success': False, 'error': 'Failed to create variant'}

    def _exp_generate_bom(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Generate BOM experiment."""
        product_id = spec.get('product_id') or self.active_product_id
        if not product_id:
            return {'success': False, 'error': 'No product specified'}

        bom_type = BOMType[spec.get('bom_type', 'ENGINEERING')]
        return self.generate_bom(product_id, spec.get('variant_id'), bom_type)

    def _exp_optimize_design(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Design optimization experiment."""
        if 'variables' in spec:
            self.setup_optimization(
                variables=spec['variables'],
                objectives=spec.get('objectives', []),
                constraints=spec.get('constraints', [])
            )

        method = spec.get('method', 'genetic')
        return self.run_optimization(method)

    def _exp_trade_off_study(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Trade-off study experiment."""
        return self.run_trade_off_study(
            objective_x=spec['objective_x'],
            objective_y=spec['objective_y'],
            num_points=spec.get('num_points', 20)
        )

    def _exp_sensitivity_analysis(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Sensitivity analysis experiment."""
        solution = spec.get('solution', {})
        return {
            'success': True,
            'sensitivities': self.run_sensitivity_analysis(solution)
        }

    def _exp_explore_design_space(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Design space exploration experiment."""
        num_samples = spec.get('num_samples', 100)
        results = self.optimizer.design_space_exploration(num_samples)

        return {
            'success': True,
            'num_samples': len(results),
            'feasible_count': len([r for r in results if r['feasible']]),
            'samples': results[:10],  # Return first 10 for preview
            'best_feasible': min(
                [r for r in results if r['feasible']],
                key=lambda x: x['fitness'],
                default=None
            )
        }

    # =========================================================================
    # Demonstration
    # =========================================================================

    def run_demo(self) -> Dict[str, Any]:
        """
        Run a demonstration of the Virtual Product Development Lab.

        Creates a sample product with components, generates BOM,
        sets up collaboration, and runs a simple optimization.
        """
        print("\n" + "=" * 60)
        print("  Virtual Product Development Lab - Demonstration")
        print("=" * 60)

        # Create a sample product
        print("\n[1] Creating sample product: Smart Sensor Device")
        product = self.create_product(
            name="Smart Sensor Device",
            description="IoT environmental sensor with multi-parameter monitoring",
            product_family="SmartSense"
        )

        # Add components across disciplines
        print("\n[2] Adding multi-discipline components...")

        # Mechanical components
        housing = self.add_component(
            product_id=product.id,
            name="Enclosure Assembly",
            component_type=ComponentType.ASSEMBLY,
            discipline=DesignDiscipline.MECHANICAL,
            description="IP67 rated enclosure",
            mass=0.15,
            unit_cost=12.50
        )

        self.add_component(
            product_id=product.id,
            name="Top Cover",
            component_type=ComponentType.PART,
            discipline=DesignDiscipline.MECHANICAL,
            parent_id=housing.id,
            description="Polycarbonate top cover",
            mass=0.05,
            unit_cost=3.20
        )

        self.add_component(
            product_id=product.id,
            name="Base Plate",
            component_type=ComponentType.PART,
            discipline=DesignDiscipline.MECHANICAL,
            parent_id=housing.id,
            description="Aluminum base with mounting holes",
            mass=0.08,
            unit_cost=5.80
        )

        # Electronic components
        pcb = self.add_component(
            product_id=product.id,
            name="Main PCB Assembly",
            component_type=ComponentType.ASSEMBLY,
            discipline=DesignDiscipline.ELECTRONIC,
            description="4-layer PCB with sensor interfaces",
            mass=0.02,
            unit_cost=18.00
        )

        self.add_component(
            product_id=product.id,
            name="Microcontroller",
            component_type=ComponentType.IC,
            discipline=DesignDiscipline.ELECTRONIC,
            parent_id=pcb.id,
            description="ARM Cortex-M4 MCU",
            unit_cost=4.50
        )

        self.add_component(
            product_id=product.id,
            name="Temperature Sensor",
            component_type=ComponentType.SENSOR,
            discipline=DesignDiscipline.ELECTRONIC,
            parent_id=pcb.id,
            description="Digital temperature sensor ±0.1°C",
            unit_cost=2.80
        )

        self.add_component(
            product_id=product.id,
            name="Humidity Sensor",
            component_type=ComponentType.SENSOR,
            discipline=DesignDiscipline.ELECTRONIC,
            parent_id=pcb.id,
            description="Capacitive humidity sensor",
            unit_cost=3.50
        )

        # Software component
        self.add_component(
            product_id=product.id,
            name="Firmware Module",
            component_type=ComponentType.FIRMWARE,
            discipline=DesignDiscipline.SOFTWARE,
            description="Embedded firmware for sensor control"
        )

        # Generate BOM
        print("\n[3] Generating Bill of Materials...")
        bom_result = self.generate_bom(product.id)
        print(f"    Total items: {bom_result['item_count']}")
        print(f"    Total cost: ${bom_result['cost_rollup']['total_material_cost']:.2f}")

        # Create a variant
        print("\n[4] Creating product variant: Industrial Version")
        variant = self.create_variant(
            product_id=product.id,
            name="Industrial Version",
            description="Extended temperature range for industrial environments"
        )

        # Add stakeholders
        print("\n[5] Setting up collaboration team...")
        pm = self.add_stakeholder(
            name="Sarah Chen",
            email="sarah.chen@company.com",
            role=StakeholderRole.PROGRAM_MANAGER
        )

        me = self.add_stakeholder(
            name="Mike Johnson",
            email="mike.j@company.com",
            role=StakeholderRole.MECHANICAL_ENGINEER
        )

        ee = self.add_stakeholder(
            name="Lisa Park",
            email="lisa.p@company.com",
            role=StakeholderRole.ELECTRICAL_ENGINEER
        )

        # Create a task
        print("\n[6] Creating design task...")
        task = self.create_task(
            title="Complete thermal analysis",
            description="Perform thermal simulation for industrial variant",
            assigned_to=[me.id],
            priority=TaskPriority.HIGH
        )

        # Setup optimization
        print("\n[7] Setting up design optimization...")
        self.setup_optimization(
            variables=[
                {'name': 'wall_thickness', 'min': 1.5, 'max': 4.0, 'unit': 'mm'},
                {'name': 'pcb_layers', 'min': 2, 'max': 6, 'unit': 'layers'},
                {'name': 'sensor_count', 'min': 2, 'max': 8, 'unit': 'count'}
            ],
            objectives=[
                {'name': 'cost', 'expression': 'wall_thickness * 3 + pcb_layers * 5 + sensor_count * 4', 'type': 'minimize'},
                {'name': 'performance', 'expression': 'pcb_layers * 2 + sensor_count * 3', 'type': 'maximize'}
            ],
            constraints=[
                {'name': 'max_cost', 'expression': 'wall_thickness * 3 + pcb_layers * 5 + sensor_count * 4', 'limit': 50, 'type': '<='}
            ]
        )

        # Run optimization
        print("\n[8] Running design optimization...")
        opt_result = self.run_optimization('genetic')
        print(f"    Optimization completed in {opt_result['computation_time']:.2f}s")
        print(f"    Best cost: {opt_result['best_objectives'].get('cost', 'N/A'):.2f}")

        # Get status
        print("\n[9] Final lab status:")
        status = self.get_status()
        print(f"    Products: {status['products']['count']}")
        print(f"    Components: {len(product.components)}")
        print(f"    Stakeholders: {status['collaboration']['stakeholder_count']}")
        print(f"    Tasks: {status['collaboration']['tasks']['total']}")

        print("\n" + "=" * 60)
        print("  Demonstration Complete!")
        print("=" * 60 + "\n")

        return {
            'success': True,
            'product': product.to_dict(),
            'bom': bom_result,
            'optimization': opt_result,
            'status': status
        }


# =============================================================================
# Module Entry Point
# =============================================================================

def main():
    """Main entry point for standalone execution."""
    lab = VirtualProductLab({
        'project_name': 'Demo Project',
        'organization': 'QuLab Demo'
    })

    # Run demonstration
    result = lab.run_demo()

    # Print capabilities
    print("\nLab Capabilities:")
    print(json.dumps(lab.get_capabilities(), indent=2))


if __name__ == "__main__":
    main()
