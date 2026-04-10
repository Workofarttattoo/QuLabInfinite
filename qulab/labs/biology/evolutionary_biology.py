"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

EVOLUTIONARY BIOLOGY LAB
Free gift to the scientific community from QuLabInfinite.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy import constants

@dataclass
class EvolutionaryBiology:
    population_size: int = 100
    generations: int = 50
    mutation_rate: float = 0.01
    crossover_prob: float = 0.7
    selection_type: str = 'roulette_wheel'
    
    def __post_init__(self):
        self.population = np.random.rand(self.population_size, 4).astype(np.float64)
        self.fitness_scores = np.zeros(self.population_size)
        self.phenotypic_variation = None
    
    def fitness_function(self, chromosome: np.ndarray) -> float:
        return np.sum(chromosome ** 2)
    
    def roulette_wheel_selection(self) -> int:
        cumulative_fitness = np.cumsum(np.clip(self.fitness_scores, 0, None))
        selected_index = np.searchsorted(cumulative_fitness, np.random.rand() * cumulative_fitness[-1])
        return selected_index
    
    def crossover(self, parent_a: np.ndarray, parent_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        point = np.random.randint(1, 3)
        child_a, child_b = np.copy(parent_a), np.copy(parent_b)
        child_a[point:], child_b[point:] = parent_b[point:], parent_a[point:]
        return child_a, child_b
    
    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        for i in range(chromosome.size):
            if np.random.rand() < self.mutation_rate:
                chromosome[i] += constants.e * (np.random.rand() - 0.5)
        return chromosome
    
    def evolve_population(self) -> None:
        for _ in range(self.generations):
            new_generation = []
            for i in range(0, self.population_size, 2):
                parent_a_index = self.roulette_wheel_selection()
                parent_b_index = self.roulette_wheel_selection() if np.random.rand() < self.crossover_prob else parent_a_index
                child_a, child_b = self.crossover(self.population[parent_a_index], self.population[parent_b_index])
                new_generation.append(self.mutate(child_a))
                new_generation.append(self.mutate(child_b) if i + 1 < self.population_size else None)
            self.population = np.array(new_generation[:self.population_size])
    
    def calculate_phenotypic_variation(self) -> float:
        mean_fitness = np.mean(np.apply_along_axis(self.fitness_function, 1, self.population))
        return np.sqrt(np.sum((np.apply_allong_axis(self.fitness_function, 1, self.population) - mean_fitness)**2) / self.population_size)
    
    def run_simulation(self):
        for i in range(self.generations):
            self.calculate_population_fitness()
            if i % 5 == 0:
                print(f"Generation {i+1}/{self.generations}, Avg Fitness: {np.mean(self.fitness_scores):.2f}")
        self.phenotypic_variation = self.calculate_phenotypic_variation()

    def calculate_population_fitness(self):
        self.fitness_scores = np.apply_along_axis(self.fitness_function, 1, self.population)

def run_demo():
    lab = EvolutionaryBiology(population_size=50, generations=20)
    lab.run_simulation()
    print(f"Phenotypic Variation: {lab.phenotypic_variation:.4f}")

if __name__ == '__main__':
    run_demo()