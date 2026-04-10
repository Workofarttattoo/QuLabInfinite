#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Material Designer - Optimization tools for material design
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from materials_database import MaterialProperties


@dataclass
class DesignResult:
    """Material design result"""
    design_type: str
    optimized_properties: MaterialProperties
    fitness_score: float
    generations: int
    convergence_history: List[float]
    notes: str = ""


class AlloyOptimizer:
    """Genetic algorithm for alloy composition optimization"""

    def __init__(self,
                 base_elements: List[Tuple[str, float, float]],  # (element, min%, max%)
                 target_properties: Dict[str, float],
                 population_size: int = 100,
                 generations: int = 50):
        """
        Initialize alloy optimizer

        Args:
            base_elements: List of (element, min_fraction, max_fraction)
            target_properties: Dict of property names and target values
            population_size: GA population size
            generations: Number of generations
        """
        self.base_elements = base_elements
        self.target_properties = target_properties
        self.population_size = population_size
        self.generations = generations

    def optimize(self) -> DesignResult:
        """
        Optimize alloy composition using genetic algorithm
        """
        # Initialize population
        population = self._initialize_population()

        convergence = []
        best_individual = None
        best_fitness = float('-inf')

        for gen in range(self.generations):
            # Evaluate fitness
            fitness = np.array([self._evaluate_fitness(ind) for ind in population])

            # Track best
            gen_best_idx = np.argmax(fitness)
            if fitness[gen_best_idx] > best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_individual = population[gen_best_idx].copy()

            convergence.append(best_fitness)

            # Selection (tournament)
            parents = self._tournament_selection(population, fitness, k=3)

            # Crossover
            offspring = self._crossover(parents)

            # Mutation
            offspring = self._mutation(offspring, mutation_rate=0.1)

            # Elitism - keep best 10%
            elite_count = int(0.1 * self.population_size)
            elite_indices = np.argsort(fitness)[-elite_count:]
            elite = [population[i] for i in elite_indices]

            # New population
            population = elite + offspring[:self.population_size - elite_count]

        # Create MaterialProperties from best composition
        optimized_props = self._composition_to_material(best_individual)

        return DesignResult(
            design_type="alloy_optimization",
            optimized_properties=optimized_props,
            fitness_score=best_fitness,
            generations=self.generations,
            convergence_history=convergence,
            notes=f"Optimized alloy composition: {self._format_composition(best_individual)}"
        )

    def _initialize_population(self) -> List[np.ndarray]:
        """Initialize random population"""
        population = []
        for _ in range(self.population_size):
            composition = []
            for element, min_frac, max_frac in self.base_elements:
                composition.append(np.random.uniform(min_frac, max_frac))
            # Normalize to sum to 1.0
            composition = np.array(composition)
            composition /= composition.sum()
            population.append(composition)
        return population

    def _evaluate_fitness(self, composition: np.ndarray) -> float:
        """Evaluate fitness of composition"""
        # Predict properties from composition (simplified mixing rules)
        predicted = self._predict_properties(composition)

        # Calculate fitness as negative weighted error
        fitness = 0
        for prop_name, target_value in self.target_properties.items():
            if prop_name in predicted:
                error = abs(predicted[prop_name] - target_value) / target_value
                fitness -= error

        return fitness

    def _predict_properties(self, composition: np.ndarray) -> Dict[str, float]:
        """Predict alloy properties from composition (simplified)"""
        # Base property values for each element (example values)
        base_properties = {
            "Fe": {"density": 7870, "strength": 400, "modulus": 210},
            "Cr": {"density": 7190, "strength": 600, "modulus": 279},
            "Ni": {"density": 8908, "strength": 500, "modulus": 207},
            "Mo": {"density": 10280, "strength": 700, "modulus": 329},
            "Al": {"density": 2700, "strength": 300, "modulus": 69},
            "Cu": {"density": 8960, "strength": 220, "modulus": 117},
            "Ti": {"density": 4500, "strength": 500, "modulus": 116}
        }

        # Rule of mixtures (linear interpolation)
        predicted = {}
        for prop in ["density", "strength", "modulus"]:
            value = 0
            for i, (element, _, _) in enumerate(self.base_elements):
                if element in base_properties:
                    value += composition[i] * base_properties[element].get(prop, 0)
            predicted[prop] = value

        return predicted

    def _tournament_selection(self, population: List[np.ndarray],
                              fitness: np.ndarray, k: int = 3) -> List[np.ndarray]:
        """Tournament selection"""
        parents = []
        for _ in range(len(population)):
            # Random tournament
            tournament_idx = np.random.choice(len(population), k, replace=False)
            winner_idx = tournament_idx[np.argmax(fitness[tournament_idx])]
            parents.append(population[winner_idx].copy())
        return parents

    def _crossover(self, parents: List[np.ndarray]) -> List[np.ndarray]:
        """Uniform crossover"""
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1 = parents[i]
                parent2 = parents[i + 1]

                # Uniform crossover
                mask = np.random.rand(len(parent1)) > 0.5
                child1 = np.where(mask, parent1, parent2)
                child2 = np.where(mask, parent2, parent1)

                # Normalize
                child1 /= child1.sum()
                child2 /= child2.sum()

                offspring.extend([child1, child2])
            else:
                offspring.append(parents[i])

        return offspring

    def _mutation(self, population: List[np.ndarray], mutation_rate: float) -> List[np.ndarray]:
        """Gaussian mutation"""
        mutated = []
        for individual in population:
            if np.random.rand() < mutation_rate:
                # Add Gaussian noise
                noise = np.random.normal(0, 0.05, len(individual))
                individual = individual + noise

                # Clip to valid range
                for i, (_, min_frac, max_frac) in enumerate(self.base_elements):
                    individual[i] = np.clip(individual[i], min_frac, max_frac)

                # Normalize
                individual /= individual.sum()

            mutated.append(individual)

        return mutated

    def _composition_to_material(self, composition: np.ndarray) -> MaterialProperties:
        """Convert composition to MaterialProperties"""
        predicted = self._predict_properties(composition)

        name = "Optimized " + "-".join([
            f"{self.base_elements[i][0]}{composition[i]*100:.1f}"
            for i in range(len(composition))
        ])

        return MaterialProperties(
            name=name,
            category="metal",
            subcategory="optimized_alloy",
            density=predicted.get("density", 0),
            youngs_modulus=predicted.get("modulus", 0),
            tensile_strength=predicted.get("strength", 0),
            yield_strength=predicted.get("strength", 0) * 0.8,
            notes="Genetically optimized alloy"
        )

    def _format_composition(self, composition: np.ndarray) -> str:
        """Format composition as string"""
        parts = []
        for i, (element, _, _) in enumerate(self.base_elements):
            parts.append(f"{element}: {composition[i]*100:.1f}%")
        return ", ".join(parts)


class CompositeDesigner:
    """Composite material design (fiber-reinforced)"""

    def __init__(self,
                 fiber_material: MaterialProperties,
                 matrix_material: MaterialProperties):
        self.fiber = fiber_material
        self.matrix = matrix_material

    def design_laminate(self,
                       fiber_volume_fraction: float = 0.60,
                       layup: List[float] = [0, 90, 0, 90]) -> DesignResult:
        """
        Design fiber-reinforced composite laminate

        Args:
            fiber_volume_fraction: Volume fraction of fibers (0-1)
            layup: List of ply angles in degrees
        """
        Vf = fiber_volume_fraction
        Vm = 1 - Vf

        # Rule of mixtures for properties
        # Longitudinal modulus (E1)
        E1 = Vf * self.fiber.youngs_modulus + Vm * self.matrix.youngs_modulus

        # Transverse modulus (E2) - inverse rule
        E2 = 1 / (Vf / self.fiber.youngs_modulus + Vm / self.matrix.youngs_modulus)

        # Shear modulus
        Gf = self.fiber.shear_modulus if self.fiber.shear_modulus > 0 else self.fiber.youngs_modulus / 2.5
        Gm = self.matrix.shear_modulus if self.matrix.shear_modulus > 0 else self.matrix.youngs_modulus / 2.5
        G12 = 1 / (Vf / Gf + Vm / Gm)

        # Density
        density = Vf * self.fiber.density + Vm * self.matrix.density

        # Strength (rule of mixtures with efficiency factor)
        efficiency = 0.85  # Typical fiber efficiency
        strength = efficiency * Vf * self.fiber.tensile_strength + Vm * self.matrix.tensile_strength

        # Effective properties from layup (simple averaging)
        n_plies = len(layup)
        E_effective = sum([E1 * np.cos(np.radians(angle))**2 + E2 * np.sin(np.radians(angle))**2
                          for angle in layup]) / n_plies

        composite = MaterialProperties(
            name=f"{self.fiber.name}/{self.matrix.name} Composite",
            category="composite",
            subcategory="fiber_reinforced",
            density=density,
            youngs_modulus=E_effective,
            shear_modulus=G12,
            tensile_strength=strength,
            yield_strength=strength,
            notes=f"Layup: [{','.join([str(a) for a in layup])}], Vf={Vf:.0%}"
        )

        return DesignResult(
            design_type="composite_laminate",
            optimized_properties=composite,
            fitness_score=strength / density,  # Specific strength
            generations=1,
            convergence_history=[strength / density],
            notes=f"Fiber volume fraction: {Vf:.0%}, Layup: {layup}"
        )


class NanostructureEngineer:
    """Nanostructure engineering for enhanced properties"""

    def __init__(self, base_material: MaterialProperties):
        self.base = base_material

    def add_nanoparticles(self,
                         nanoparticle_type: str = "CNT",
                         loading_percent: float = 1.0) -> DesignResult:
        """
        Add nanoparticles to base material

        Args:
            nanoparticle_type: Type of nanoparticles (CNT, graphene, etc.)
            loading_percent: Weight percent of nanoparticles
        """
        # Nanoparticle properties
        nano_props = {
            "CNT": {"strength_mult": 3.0, "modulus_mult": 2.5, "conductivity_mult": 100},
            "graphene": {"strength_mult": 5.0, "modulus_mult": 3.0, "conductivity_mult": 1000},
            "silica": {"strength_mult": 1.2, "modulus_mult": 1.5, "conductivity_mult": 0.1}
        }

        if nanoparticle_type not in nano_props:
            nanoparticle_type = "CNT"

        props = nano_props[nanoparticle_type]
        loading = loading_percent / 100.0

        # Property enhancement (diminishing returns at high loading)
        efficiency = np.exp(-loading * 5)  # Aggregation at high loading
        strength_enhancement = 1 + (props["strength_mult"] - 1) * loading * efficiency
        modulus_enhancement = 1 + (props["modulus_mult"] - 1) * loading * efficiency
        conductivity_enhancement = props["conductivity_mult"] * loading

        # Create nanocomposite
        nanocomposite = MaterialProperties(
            name=f"{self.base.name} + {loading_percent:.1f}% {nanoparticle_type}",
            category=self.base.category,
            subcategory="nanocomposite",
            density=self.base.density * (1 + loading * 0.1),  # Slight increase
            youngs_modulus=self.base.youngs_modulus * modulus_enhancement,
            tensile_strength=self.base.tensile_strength * strength_enhancement,
            yield_strength=self.base.yield_strength * strength_enhancement,
            electrical_conductivity=self.base.electrical_conductivity + conductivity_enhancement,
            thermal_conductivity=self.base.thermal_conductivity * (1 + loading * 0.5),
            cost_per_kg=self.base.cost_per_kg * (1 + loading * 10),  # Expensive
            notes=f"Nanocomposite with {loading_percent:.1f}% {nanoparticle_type}"
        )

        return DesignResult(
            design_type="nanostructure_engineering",
            optimized_properties=nanocomposite,
            fitness_score=strength_enhancement * modulus_enhancement,
            generations=1,
            convergence_history=[strength_enhancement],
            notes=f"Added {loading_percent:.1f}% {nanoparticle_type} nanoparticles"
        )


class SurfaceTreatment:
    """Surface treatment simulation"""

    def __init__(self, base_material: MaterialProperties):
        self.base = base_material

    def apply_coating(self,
                     coating_type: str = "DLC",  # Diamond-Like Carbon
                     thickness_um: float = 5.0) -> DesignResult:
        """
        Apply surface coating

        Args:
            coating_type: Type of coating (DLC, TiN, CrN, etc.)
            thickness_um: Coating thickness in micrometers
        """
        coatings = {
            "DLC": {"hardness": 3000, "friction": 0.05, "corrosion": "excellent"},
            "TiN": {"hardness": 2500, "friction": 0.4, "corrosion": "excellent"},
            "CrN": {"hardness": 2000, "friction": 0.5, "corrosion": "excellent"},
            "anodizing": {"hardness": 400, "friction": 0.3, "corrosion": "excellent"}
        }

        if coating_type not in coatings:
            coating_type = "DLC"

        coating = coatings[coating_type]

        # Create coated material (properties are mostly bulk)
        coated = MaterialProperties(
            name=f"{self.base.name} + {coating_type} coating",
            category=self.base.category,
            subcategory=self.base.subcategory,
            density=self.base.density,  # Coating is thin
            youngs_modulus=self.base.youngs_modulus,
            tensile_strength=self.base.tensile_strength,
            hardness_vickers=coating["hardness"],  # Surface hardness
            corrosion_resistance=coating["corrosion"],
            cost_per_kg=self.base.cost_per_kg * 1.2,  # Slight cost increase
            notes=f"Surface treated with {thickness_um:.1f} μm {coating_type} coating"
        )

        return DesignResult(
            design_type="surface_treatment",
            optimized_properties=coated,
            fitness_score=coating["hardness"] / 1000,
            generations=1,
            convergence_history=[coating["hardness"]],
            notes=f"Applied {thickness_um:.1f} μm {coating_type} coating"
        )


class AdditiveManufacturing:
    """Additive manufacturing (3D printing) simulation"""

    def __init__(self, base_material: MaterialProperties):
        self.base = base_material

    def design_lattice_structure(self,
                                relative_density: float = 0.30,
                                cell_type: str = "BCC") -> DesignResult:
        """
        Design lattice structure for AM

        Args:
            relative_density: Fraction of solid material (0-1)
            cell_type: Unit cell type (BCC, FCC, octet, etc.)
        """
        # Lattice scaling laws (Gibson-Ashby)
        rho_rel = relative_density

        # Properties scale with relative density
        if cell_type == "BCC":
            # Bending-dominated
            E_scale = rho_rel**2
            strength_scale = rho_rel**1.5
        elif cell_type == "FCC":
            # Stretch-dominated (more efficient)
            E_scale = rho_rel
            strength_scale = rho_rel
        elif cell_type == "octet":
            # Optimal stretch-dominated
            E_scale = rho_rel
            strength_scale = rho_rel
        else:
            E_scale = rho_rel**2
            strength_scale = rho_rel**1.5

        # Create lattice material
        lattice = MaterialProperties(
            name=f"{self.base.name} Lattice ({cell_type})",
            category=self.base.category,
            subcategory="lattice_structure",
            density=self.base.density * rho_rel,
            youngs_modulus=self.base.youngs_modulus * E_scale,
            tensile_strength=self.base.tensile_strength * strength_scale,
            yield_strength=self.base.yield_strength * strength_scale,
            thermal_conductivity=self.base.thermal_conductivity * rho_rel,
            cost_per_kg=self.base.cost_per_kg * 1.5,  # AM premium
            notes=f"Lattice structure: {cell_type}, relative density: {rho_rel:.0%}"
        )

        # Specific properties (per unit mass)
        specific_strength = lattice.tensile_strength / lattice.density
        specific_modulus = lattice.youngs_modulus / lattice.density * 1000

        return DesignResult(
            design_type="additive_manufacturing",
            optimized_properties=lattice,
            fitness_score=specific_strength,
            generations=1,
            convergence_history=[specific_strength],
            notes=f"{cell_type} lattice with {rho_rel:.0%} density, "
                  f"specific strength: {specific_strength:.1f} MPa/(kg/m³)"
        )


if __name__ == "__main__":
    from materials_database import MaterialsDatabase

    db = MaterialsDatabase()

    print("="*70)
    print("MATERIAL DESIGNER EXAMPLES")
    print("="*70)

    # Example 1: Optimize steel alloy
    print("\n1. ALLOY OPTIMIZATION: High-strength low-alloy steel")
    print("-" * 70)

    optimizer = AlloyOptimizer(
        base_elements=[
            ("Fe", 0.90, 0.98),
            ("Cr", 0.005, 0.05),
            ("Ni", 0.005, 0.03),
            ("Mo", 0.0, 0.02)
        ],
        target_properties={
            "density": 7850,
            "strength": 1000,
            "modulus": 210
        },
        population_size=50,
        generations=30
    )

    result = optimizer.optimize()
    print(f"Optimized alloy: {result.optimized_properties.name}")
    print(f"Density: {result.optimized_properties.density:.0f} kg/m³")
    print(f"Strength: {result.optimized_properties.tensile_strength:.0f} MPa")
    print(f"Modulus: {result.optimized_properties.youngs_modulus:.0f} GPa")
    print(f"Fitness score: {result.fitness_score:.4f}")
    print(f"Generations: {result.generations}")

    # Example 2: Carbon fiber composite
    print("\n2. COMPOSITE DESIGN: Carbon fiber/epoxy laminate")
    print("-" * 70)

    carbon = db.get_material("Carbon Fiber Epoxy")
    epoxy = db.get_material("Epoxy Resin")

    # Create designer (note: we're using carbon fiber properties as fiber)
    designer = CompositeDesigner(carbon, epoxy)
    result = designer.design_laminate(
        fiber_volume_fraction=0.60,
        layup=[0, 45, -45, 90, 90, -45, 45, 0]  # Quasi-isotropic
    )

    print(f"Composite: {result.optimized_properties.name}")
    print(f"Density: {result.optimized_properties.density:.0f} kg/m³")
    print(f"Modulus: {result.optimized_properties.youngs_modulus:.0f} GPa")
    print(f"Strength: {result.optimized_properties.tensile_strength:.0f} MPa")
    print(f"Specific strength: {result.fitness_score:.1f}")
    print(result.notes)

    # Example 3: Nanocomposite
    print("\n3. NANOSTRUCTURE: PEEK + Carbon Nanotubes")
    print("-" * 70)

    peek = db.get_material("PEEK")
    nano_eng = NanostructureEngineer(peek)
    result = nano_eng.add_nanoparticles(
        nanoparticle_type="CNT",
        loading_percent=2.0
    )

    print(f"Nanocomposite: {result.optimized_properties.name}")
    print(f"Strength: {result.optimized_properties.tensile_strength:.0f} MPa")
    print(f"  vs base: {peek.tensile_strength:.0f} MPa")
    print(f"  Enhancement: {result.optimized_properties.tensile_strength/peek.tensile_strength:.2f}x")
    print(f"Modulus: {result.optimized_properties.youngs_modulus:.1f} GPa")
    print(f"  vs base: {peek.youngs_modulus:.1f} GPa")
    print(result.notes)

    # Example 4: Lattice structure
    print("\n4. ADDITIVE MANUFACTURING: Titanium lattice")
    print("-" * 70)

    ti = db.get_material("Ti-6Al-4V")
    am = AdditiveManufacturing(ti)
    result = am.design_lattice_structure(
        relative_density=0.25,
        cell_type="octet"
    )

    print(f"Lattice: {result.optimized_properties.name}")
    print(f"Density: {result.optimized_properties.density:.0f} kg/m³")
    print(f"  vs solid: {ti.density:.0f} kg/m³")
    print(f"  Weight saving: {(1-result.optimized_properties.density/ti.density)*100:.0f}%")
    print(f"Strength: {result.optimized_properties.tensile_strength:.0f} MPa")
    print(f"Specific strength: {result.fitness_score:.1f} MPa/(kg/m³)")
    print(result.notes)

    print("\n" + "="*70)
    print("Material Designer ready! ✓")
