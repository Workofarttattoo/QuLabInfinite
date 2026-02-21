"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

ECOLOGY LAB
Advanced ecological modeling with population dynamics and ecosystem services.
Production-ready implementation for ecological research and conservation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Set
from enum import Enum
import warnings
from scipy import integrate, optimize, interpolate, spatial
from scipy.stats import poisson, lognorm, gamma
import networkx as nx


class EcosystemType(Enum):
    """Major ecosystem classifications"""
    FOREST = "Forest ecosystem"
    GRASSLAND = "Grassland ecosystem"
    DESERT = "Desert ecosystem"
    WETLAND = "Wetland ecosystem"
    MARINE = "Marine ecosystem"
    FRESHWATER = "Freshwater ecosystem"
    TUNDRA = "Tundra ecosystem"
    URBAN = "Urban ecosystem"


@dataclass
class Species:
    """Species characteristics and ecological parameters"""
    name: str
    trophic_level: float  # 1=primary producer, 2=herbivore, 3=carnivore
    body_mass: float  # kg
    metabolic_rate: float  # W/kg
    growth_rate: float  # per day
    carrying_capacity: float  # individuals/km²
    mortality_rate: float  # per day
    dispersal_range: float  # km
    diet: Dict[str, float] = field(default_factory=dict)  # prey: proportion
    habitat_requirements: Dict[str, float] = field(default_factory=dict)


@dataclass
class EcosystemService:
    """Ecosystem service valuation"""
    service_type: str  # provisioning, regulating, cultural, supporting
    value: float  # $/ha/year
    beneficiaries: int  # number of people
    quality_index: float  # 0-1 scale
    sustainability_score: float  # 0-1 scale


class EcologyLab:
    """
    Advanced ecology laboratory.
    Implements population dynamics, food webs, biodiversity metrics, and ecosystem services.
    """

    def __init__(self):
        self.area = 100.0  # km²
        self.temperature = 15.0  # °C
        self.precipitation = 1000.0  # mm/year
        self.elevation = 100.0  # meters
        self.latitude = 45.0  # degrees
        self._initialize_species()
        self._initialize_parameters()

    def _initialize_species(self):
        """Initialize species database"""
        self.species = {
            'grass': Species('grass', 1.0, 0.001, 0, 0.1, 1e6, 0.05, 0.1),
            'tree': Species('tree', 1.0, 1000, 0, 0.01, 100, 0.001, 10),
            'rabbit': Species('rabbit', 2.0, 2.0, 15, 0.5, 1000, 0.1, 2, {'grass': 1.0}),
            'deer': Species('deer', 2.0, 50, 8, 0.2, 50, 0.05, 20, {'grass': 0.7, 'tree': 0.3}),
            'fox': Species('fox', 3.0, 5, 20, 0.1, 20, 0.08, 30, {'rabbit': 0.8, 'bird': 0.2}),
            'wolf': Species('wolf', 3.5, 40, 15, 0.05, 5, 0.04, 100, {'deer': 0.9, 'rabbit': 0.1}),
            'eagle': Species('eagle', 3.5, 4, 25, 0.08, 10, 0.06, 50, {'rabbit': 0.5, 'fish': 0.5}),
            'insect': Species('insect', 2.0, 0.001, 100, 2.0, 1e9, 0.5, 0.1, {'grass': 1.0}),
            'bird': Species('bird', 2.5, 0.05, 40, 0.3, 500, 0.15, 5, {'insect': 0.7, 'seed': 0.3}),
            'fish': Species('fish', 2.5, 1.0, 10, 0.4, 2000, 0.2, 1, {'algae': 0.5, 'insect': 0.5}),
        }

    def _initialize_parameters(self):
        """Initialize ecological parameters"""
        # Metabolic scaling
        self.metabolic_exponent = 0.75  # Kleiber's law

        # Competition coefficients
        self.competition_matrix = np.array([
            [1.0, 0.5, 0.2],  # intraspecific > interspecific
            [0.5, 1.0, 0.3],
            [0.2, 0.3, 1.0]
        ])

        # Ecosystem service values ($/ha/year)
        self.service_values = {
            'carbon_sequestration': 150,
            'water_purification': 200,
            'pollination': 300,
            'nutrient_cycling': 100,
            'erosion_control': 180,
            'recreation': 250,
            'biodiversity': 400,
            'timber': 500,
            'food_production': 600
        }

    def lotka_volterra_dynamics(self,
                               initial_populations: Dict[str, float],
                               interactions: Dict[Tuple[str, str], float],
                               time_years: float = 10,
                               stochastic: bool = False) -> Dict[str, np.ndarray]:
        """
        Generalized Lotka-Volterra population dynamics.
        Can handle predator-prey, competition, and mutualism.
        """
        species_list = list(initial_populations.keys())
        n_species = len(species_list)

        # Build interaction matrix
        alpha = np.zeros((n_species, n_species))
        for i, sp1 in enumerate(species_list):
            for j, sp2 in enumerate(species_list):
                if (sp1, sp2) in interactions:
                    alpha[i, j] = interactions[(sp1, sp2)]
                elif i == j:
                    alpha[i, j] = 1.0  # Self-regulation

        def derivatives(N, t):
            dN_dt = np.zeros(n_species)
            for i, species in enumerate(species_list):
                sp = self.species[species]

                # Logistic growth with interactions
                growth = sp.growth_rate * N[i] * (1 - N[i] / sp.carrying_capacity)

                # Interaction effects
                interaction_sum = 0
                for j in range(n_species):
                    if i != j:
                        interaction_sum += alpha[i, j] * N[j]

                dN_dt[i] = growth - sp.mortality_rate * N[i] + interaction_sum * N[i]

                # Add environmental stochasticity
                if stochastic:
                    noise = np.random.normal(0, 0.01 * abs(dN_dt[i]))
                    dN_dt[i] += noise

            return dN_dt

        # Initial conditions
        N0 = [initial_populations[sp] for sp in species_list]

        # Time points
        t = np.linspace(0, time_years * 365, int(time_years * 52))  # Weekly resolution

        # Solve ODE
        solution = integrate.odeint(derivatives, N0, t)

        # Ensure non-negative populations
        solution = np.maximum(solution, 0)

        result = {'time': t / 365}  # Convert to years
        for i, species in enumerate(species_list):
            result[species] = solution[:, i]

        return result

    def food_web_structure(self, species_list: List[str]) -> Dict[str, any]:
        """
        Analyze food web structure and stability.
        Returns network metrics and trophic relationships.
        """
        # Build directed graph
        G = nx.DiGraph()

        # Add nodes
        for species in species_list:
            if species in self.species:
                sp = self.species[species]
                G.add_node(species, trophic_level=sp.trophic_level, body_mass=sp.body_mass)

        # Add edges (feeding relationships)
        for predator in species_list:
            if predator in self.species:
                sp = self.species[predator]
                for prey, proportion in sp.diet.items():
                    if prey in species_list:
                        G.add_edge(prey, predator, weight=proportion)

        # Calculate network metrics
        n_species = len(G.nodes())
        n_links = len(G.edges())
        connectance = n_links / (n_species ** 2) if n_species > 0 else 0

        # Trophic levels
        trophic_levels = {}
        for node in G.nodes():
            if G.in_degree(node) == 0:  # Primary producer
                trophic_levels[node] = 1.0
            else:
                # Average trophic level of prey + 1
                prey_levels = [trophic_levels.get(prey, 1.0) for prey in G.predecessors(node)]
                trophic_levels[node] = np.mean(prey_levels) + 1 if prey_levels else 2.0

        # Centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)

        # Identify keystone species (high betweenness, low biomass)
        keystone_candidates = []
        for species in G.nodes():
            if species in self.species:
                sp = self.species[species]
                if betweenness[species] > 0.2 and sp.body_mass < 10:
                    keystone_candidates.append(species)

        # Food chain length
        try:
            longest_path = max(nx.all_simple_paths(G,
                                                   source=min(trophic_levels, key=trophic_levels.get),
                                                   target=max(trophic_levels, key=trophic_levels.get)),
                             key=len)
            max_chain_length = len(longest_path)
        except:
            max_chain_length = max(trophic_levels.values()) if trophic_levels else 0

        # Modularity (community structure)
        if n_species > 2:
            communities = list(nx.community.greedy_modularity_communities(G.to_undirected()))
            modularity = nx.community.modularity(G.to_undirected(), communities)
        else:
            communities = []
            modularity = 0

        return {
            'n_species': n_species,
            'n_links': n_links,
            'connectance': connectance,
            'mean_trophic_level': np.mean(list(trophic_levels.values())) if trophic_levels else 0,
            'max_chain_length': max_chain_length,
            'modularity': modularity,
            'n_communities': len(communities),
            'keystone_species': keystone_candidates,
            'degree_centrality': degree_centrality,
            'betweenness_centrality': betweenness,
            'trophic_levels': trophic_levels,
            'graph': G
        }

    def biodiversity_metrics(self,
                           abundances: np.ndarray,
                           spatial_distribution: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive biodiversity metrics.
        Includes alpha, beta, and gamma diversity.
        """
        # Remove zeros
        abundances = abundances[abundances > 0]
        n_species = len(abundances)

        if n_species == 0:
            return {metric: 0 for metric in [
                'species_richness', 'shannon_index', 'simpson_index',
                'evenness', 'berger_parker_dominance', 'margalef_index'
            ]}

        # Species richness
        richness = n_species

        # Shannon diversity index
        proportions = abundances / np.sum(abundances)
        shannon = -np.sum(proportions * np.log(proportions + 1e-10))

        # Simpson diversity index
        simpson = 1 - np.sum(proportions ** 2)

        # Pielou's evenness
        evenness = shannon / np.log(richness) if richness > 1 else 0

        # Berger-Parker dominance
        berger_parker = np.max(proportions)

        # Margalef's richness index
        margalef = (richness - 1) / np.log(np.sum(abundances)) if np.sum(abundances) > 1 else 0

        # Hill numbers (true diversity)
        hill_0 = richness  # Species richness
        hill_1 = np.exp(shannon)  # Exponential of Shannon
        hill_2 = 1 / (1 - simpson) if simpson < 1 else np.inf  # Inverse Simpson

        # Rarity indices
        rare_species = np.sum(abundances < np.percentile(abundances, 25))
        common_species = np.sum(abundances > np.percentile(abundances, 75))

        metrics = {
            'species_richness': richness,
            'shannon_index': shannon,
            'simpson_index': simpson,
            'evenness': evenness,
            'berger_parker_dominance': berger_parker,
            'margalef_index': margalef,
            'hill_number_0': hill_0,
            'hill_number_1': hill_1,
            'hill_number_2': hill_2,
            'rare_species': rare_species,
            'common_species': common_species
        }

        # Beta diversity if spatial data provided
        if spatial_distribution is not None and spatial_distribution.shape[0] > 1:
            # Jaccard dissimilarity between sites
            n_sites = spatial_distribution.shape[0]
            beta_diversity = 0
            for i in range(n_sites):
                for j in range(i + 1, n_sites):
                    shared = np.sum(np.minimum(spatial_distribution[i], spatial_distribution[j]) > 0)
                    total = np.sum(np.maximum(spatial_distribution[i], spatial_distribution[j]) > 0)
                    if total > 0:
                        beta_diversity += 1 - (shared / total)

            beta_diversity /= (n_sites * (n_sites - 1) / 2)
            metrics['beta_diversity'] = beta_diversity

            # Gamma diversity (regional)
            gamma_abundances = np.sum(spatial_distribution, axis=0)
            gamma_proportions = gamma_abundances / np.sum(gamma_abundances)
            gamma_shannon = -np.sum(gamma_proportions * np.log(gamma_proportions + 1e-10))
            metrics['gamma_diversity'] = gamma_shannon

        return metrics

    def species_area_relationship(self,
                                 min_area: float = 0.1,
                                 max_area: float = 10000,
                                 n_samples: int = 20) -> Dict[str, any]:
        """
        Model species-area relationship (SAR).
        S = c * A^z where S is species, A is area.
        """
        # Generate area samples (log scale)
        areas = np.logspace(np.log10(min_area), np.log10(max_area), n_samples)

        # Simulate species richness for different areas
        species_counts = []
        for area in areas:
            # Power law relationship with noise
            c = 10  # Species richness constant
            z = 0.25  # Typical value for contiguous habitats

            expected_species = c * area ** z
            # Add Poisson noise
            observed_species = np.random.poisson(expected_species)
            species_counts.append(observed_species)

        species_counts = np.array(species_counts)

        # Fit power law
        log_areas = np.log10(areas)
        log_species = np.log10(species_counts + 1)  # Add 1 to avoid log(0)

        # Linear regression in log-log space
        coeffs = np.polyfit(log_areas, log_species, 1)
        z_fitted = coeffs[0]
        log_c_fitted = coeffs[1]
        c_fitted = 10 ** log_c_fitted

        # Calculate fit quality
        predicted = c_fitted * areas ** z_fitted
        r_squared = 1 - np.sum((species_counts - predicted) ** 2) / np.sum((species_counts - np.mean(species_counts)) ** 2)

        # Extrapolate to estimate total species
        total_species_est = c_fitted * self.area ** z_fitted

        return {
            'areas': areas,
            'species_counts': species_counts,
            'c_parameter': c_fitted,
            'z_parameter': z_fitted,
            'r_squared': r_squared,
            'equation': f'S = {c_fitted:.2f} * A^{z_fitted:.3f}',
            'estimated_total_species': total_species_est,
            'species_per_unit_area': c_fitted
        }

    def metapopulation_dynamics(self,
                               patches: int,
                               colonization_rate: float,
                               extinction_rate: float,
                               initial_occupied: int,
                               time_years: float = 50,
                               connectivity_matrix: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Levins metapopulation model with spatial structure.
        Tracks patch occupancy over time.
        """
        if connectivity_matrix is None:
            # Assume all patches equally connected
            connectivity_matrix = np.ones((patches, patches)) - np.eye(patches)
            connectivity_matrix /= (patches - 1)

        def derivatives(occupied_vector, t):
            # occupied_vector is binary (0 or 1) for each patch
            p = np.mean(occupied_vector)  # Proportion occupied

            dO_dt = np.zeros(patches)
            for i in range(patches):
                if occupied_vector[i] == 0:  # Empty patch
                    # Colonization depends on nearby occupied patches
                    colonization_pressure = np.sum(connectivity_matrix[i] * occupied_vector)
                    prob_colonization = colonization_rate * colonization_pressure
                    dO_dt[i] = prob_colonization
                else:  # Occupied patch
                    # Extinction
                    dO_dt[i] = -extinction_rate

            return dO_dt

        # Stochastic simulation
        time_points = np.linspace(0, time_years, int(time_years * 12))  # Monthly
        occupancy = np.zeros((len(time_points), patches))

        # Initial conditions
        initial_patches = np.random.choice(patches, initial_occupied, replace=False)
        occupancy[0, initial_patches] = 1

        for t_idx in range(1, len(time_points)):
            dt = time_points[t_idx] - time_points[t_idx - 1]

            for patch in range(patches):
                if occupancy[t_idx - 1, patch] == 0:  # Empty
                    # Colonization probability
                    colonization_pressure = np.sum(connectivity_matrix[patch] * occupancy[t_idx - 1])
                    prob_col = 1 - np.exp(-colonization_rate * colonization_pressure * dt)
                    if np.random.random() < prob_col:
                        occupancy[t_idx, patch] = 1
                    else:
                        occupancy[t_idx, patch] = 0
                else:  # Occupied
                    # Extinction probability
                    prob_ext = 1 - np.exp(-extinction_rate * dt)
                    if np.random.random() < prob_ext:
                        occupancy[t_idx, patch] = 0
                    else:
                        occupancy[t_idx, patch] = 1

        # Calculate summary statistics
        proportion_occupied = np.mean(occupancy, axis=1)
        turnover = np.sum(np.abs(np.diff(occupancy, axis=0)), axis=1)

        # Equilibrium prediction (Levins model)
        equilibrium = 1 - extinction_rate / colonization_rate if colonization_rate > extinction_rate else 0

        return {
            'time': time_points,
            'occupancy_matrix': occupancy,
            'proportion_occupied': proportion_occupied,
            'turnover_rate': turnover,
            'equilibrium_occupancy': equilibrium,
            'mean_persistence_time': 1 / extinction_rate,
            'colonization_events': np.sum(np.diff(occupancy, axis=0) > 0),
            'extinction_events': np.sum(np.diff(occupancy, axis=0) < 0)
        }

    def island_biogeography(self,
                           island_area: float,
                           distance_to_mainland: float,
                           mainland_species: int = 1000) -> Dict[str, float]:
        """
        MacArthur-Wilson island biogeography theory.
        Predicts species richness based on area and isolation.
        """
        # Immigration rate decreases with distance
        max_immigration = 10  # Species/year at distance = 0
        immigration_rate = max_immigration * np.exp(-distance_to_mainland / 100)

        # Extinction rate decreases with area
        base_extinction = 1.0  # Species/year for 1 km²
        extinction_rate = base_extinction / np.sqrt(island_area)

        # Equilibrium species richness
        # At equilibrium: immigration * (1 - S/P) = extinction * S
        # Where S = island species, P = mainland pool
        equilibrium_richness = mainland_species * immigration_rate / (immigration_rate + extinction_rate)

        # Species turnover rate at equilibrium
        turnover_rate = immigration_rate * (1 - equilibrium_richness / mainland_species)

        # Time to reach equilibrium (90%)
        time_to_equilibrium = -np.log(0.1) / (immigration_rate + extinction_rate)

        # Colonization probability for individual species
        colonization_prob = 1 - np.exp(-immigration_rate)

        # Expected persistence time
        persistence_time = 1 / extinction_rate

        return {
            'equilibrium_richness': equilibrium_richness,
            'immigration_rate': immigration_rate,
            'extinction_rate': extinction_rate,
            'turnover_rate': turnover_rate,
            'time_to_equilibrium_years': time_to_equilibrium,
            'colonization_probability': colonization_prob,
            'mean_persistence_years': persistence_time,
            'isolation_effect': np.exp(-distance_to_mainland / 100),
            'area_effect': np.sqrt(island_area)
        }

    def ecological_succession(self,
                            disturbance_type: str = 'fire',
                            time_years: float = 100) -> Dict[str, np.ndarray]:
        """
        Model ecological succession after disturbance.
        Tracks community composition changes over time.
        """
        # Succession stages and their characteristic species
        stages = {
            'pioneer': {'grass': 0.7, 'insect': 0.3},
            'early': {'grass': 0.4, 'rabbit': 0.3, 'bird': 0.3},
            'mid': {'tree': 0.3, 'deer': 0.2, 'bird': 0.3, 'insect': 0.2},
            'late': {'tree': 0.5, 'deer': 0.2, 'wolf': 0.1, 'bird': 0.2},
            'climax': {'tree': 0.6, 'deer': 0.15, 'wolf': 0.05, 'fox': 0.1, 'bird': 0.1}
        }

        # Time for each stage (years)
        stage_durations = {
            'pioneer': 2,
            'early': 5,
            'mid': 15,
            'late': 30,
            'climax': 48
        }

        # Initialize arrays
        time_points = np.linspace(0, time_years, int(time_years * 4))  # Quarterly
        species_list = list(set(sp for stage in stages.values() for sp in stage.keys()))
        composition = np.zeros((len(time_points), len(species_list)))

        # Track succession
        current_time = 0
        for t_idx, t in enumerate(time_points):
            # Determine current stage
            cumulative_time = 0
            current_stage = 'climax'
            for stage, duration in stage_durations.items():
                cumulative_time += duration
                if t < cumulative_time:
                    current_stage = stage
                    break

            # Set species composition for current stage
            stage_composition = stages[current_stage]
            for sp_idx, species in enumerate(species_list):
                if species in stage_composition:
                    # Add stochastic variation
                    base_proportion = stage_composition[species]
                    noise = np.random.normal(0, 0.05 * base_proportion)
                    composition[t_idx, sp_idx] = max(0, base_proportion + noise)

        # Normalize to sum to 1
        row_sums = np.sum(composition, axis=1, keepdims=True)
        composition = np.divide(composition, row_sums, where=row_sums > 0)

        # Calculate diversity through succession
        diversity_timeline = []
        for t_idx in range(len(time_points)):
            abundances = composition[t_idx] * 100  # Convert to pseudo-abundances
            div_metrics = self.biodiversity_metrics(abundances[abundances > 0])
            diversity_timeline.append(div_metrics['shannon_index'])

        return {
            'time': time_points,
            'species': species_list,
            'composition_matrix': composition,
            'diversity_timeline': np.array(diversity_timeline),
            'stages': list(stages.keys()),
            'stage_durations': stage_durations,
            'disturbance_type': disturbance_type
        }

    def ecosystem_services_valuation(self,
                                   land_use: Dict[str, float],
                                   population: int = 10000) -> Dict[str, any]:
        """
        Calculate economic value of ecosystem services.
        Based on land use and ecosystem health.
        """
        total_area = sum(land_use.values())

        service_provision = {}
        total_value = 0

        for service, base_value in self.service_values.items():
            service_value = 0

            if service == 'carbon_sequestration':
                # Forest and grassland contribute
                forest_area = land_use.get('forest', 0)
                grassland_area = land_use.get('grassland', 0)
                # Forests sequester ~10 tC/ha/year, grasslands ~2 tC/ha/year
                carbon_tons = forest_area * 10 + grassland_area * 2
                service_value = carbon_tons * 50  # $50/tCO2

            elif service == 'water_purification':
                # Wetlands and forests
                wetland_area = land_use.get('wetland', 0)
                forest_area = land_use.get('forest', 0)
                service_value = (wetland_area * 500 + forest_area * 200)

            elif service == 'pollination':
                # Agricultural and natural areas
                ag_area = land_use.get('agriculture', 0)
                natural_area = land_use.get('forest', 0) + land_use.get('grassland', 0)
                service_value = ag_area * 300 * min(1, natural_area / (ag_area + 1))

            elif service == 'nutrient_cycling':
                # All natural ecosystems
                natural_area = total_area - land_use.get('urban', 0) - land_use.get('agriculture', 0)
                service_value = natural_area * base_value

            elif service == 'erosion_control':
                # Forest and grassland
                vegetation_area = land_use.get('forest', 0) + land_use.get('grassland', 0)
                service_value = vegetation_area * base_value

            elif service == 'recreation':
                # Natural areas accessible to population
                natural_area = total_area - land_use.get('urban', 0) - land_use.get('agriculture', 0)
                accessibility = min(1, natural_area / 100)  # Diminishing returns
                service_value = natural_area * base_value * accessibility * (population / 10000)

            elif service == 'biodiversity':
                # All natural habitats
                natural_area = total_area - land_use.get('urban', 0) - land_use.get('agriculture', 0)
                # Value increases non-linearly with area
                service_value = base_value * natural_area * np.sqrt(natural_area / 100)

            elif service == 'timber':
                # Forest products
                forest_area = land_use.get('forest', 0)
                sustainable_harvest = forest_area * 0.02  # 2% annual harvest
                service_value = sustainable_harvest * 1000  # $1000/ha harvested

            elif service == 'food_production':
                # Agricultural land
                ag_area = land_use.get('agriculture', 0)
                service_value = ag_area * base_value

            service_provision[service] = {
                'value': service_value,
                'value_per_capita': service_value / population if population > 0 else 0,
                'area_providing': total_area,
                'quality_index': min(1, service_value / (base_value * total_area + 1))
            }

            total_value += service_value

        # Calculate sustainability metrics
        natural_capital = sum(land_use.get(lu, 0) for lu in ['forest', 'wetland', 'grassland'])
        sustainability_index = natural_capital / total_area if total_area > 0 else 0

        return {
            'services': service_provision,
            'total_value': total_value,
            'total_value_per_ha': total_value / total_area if total_area > 0 else 0,
            'total_value_per_capita': total_value / population if population > 0 else 0,
            'natural_capital_ha': natural_capital,
            'sustainability_index': sustainability_index,
            'land_use': land_use,
            'population_served': population
        }

    def habitat_fragmentation_analysis(self,
                                     landscape_matrix: np.ndarray,
                                     cell_size: float = 1.0) -> Dict[str, float]:
        """
        Analyze habitat fragmentation metrics.
        1 = habitat, 0 = non-habitat in matrix.
        """
        # Basic metrics
        total_cells = landscape_matrix.size
        habitat_cells = np.sum(landscape_matrix == 1)
        habitat_proportion = habitat_cells / total_cells if total_cells > 0 else 0

        # Find patches using connected components
        from scipy import ndimage
        labeled_patches, n_patches = ndimage.label(landscape_matrix == 1)

        # Patch metrics
        patch_sizes = []
        patch_perimeters = []

        for patch_id in range(1, n_patches + 1):
            patch_mask = labeled_patches == patch_id
            patch_size = np.sum(patch_mask)
            patch_sizes.append(patch_size)

            # Calculate perimeter (edge cells)
            eroded = ndimage.binary_erosion(patch_mask)
            perimeter = np.sum(patch_mask) - np.sum(eroded)
            patch_perimeters.append(perimeter)

        patch_sizes = np.array(patch_sizes) * cell_size ** 2  # Convert to area
        patch_perimeters = np.array(patch_perimeters) * cell_size

        # Fragmentation metrics
        mean_patch_size = np.mean(patch_sizes) if len(patch_sizes) > 0 else 0
        largest_patch_index = np.max(patch_sizes) / (habitat_cells * cell_size ** 2) if habitat_cells > 0 else 0

        # Edge density
        total_edge = np.sum(patch_perimeters)
        edge_density = total_edge / (total_cells * cell_size ** 2) if total_cells > 0 else 0

        # Mean shape index (circle = 1, complex shape > 1)
        shape_indices = []
        for size, perimeter in zip(patch_sizes, patch_perimeters):
            if size > 0:
                # Shape index = perimeter / (2 * sqrt(pi * area))
                expected_perimeter = 2 * np.sqrt(np.pi * size)
                shape_index = perimeter / expected_perimeter if expected_perimeter > 0 else 1
                shape_indices.append(shape_index)

        mean_shape_index = np.mean(shape_indices) if shape_indices else 1

        # Connectivity (proportion of habitat within distance threshold)
        distance_threshold = 3  # cells
        connectivity = 0
        if n_patches > 1:
            # Calculate distances between patch centroids
            centroids = []
            for patch_id in range(1, n_patches + 1):
                y, x = np.where(labeled_patches == patch_id)
                centroids.append([np.mean(y), np.mean(x)])

            centroids = np.array(centroids)
            distances = spatial.distance_matrix(centroids, centroids)
            connected_pairs = np.sum(distances < distance_threshold) - n_patches
            possible_pairs = n_patches * (n_patches - 1)
            connectivity = connected_pairs / possible_pairs if possible_pairs > 0 else 0

        return {
            'habitat_proportion': habitat_proportion,
            'number_of_patches': n_patches,
            'mean_patch_size': mean_patch_size,
            'largest_patch_index': largest_patch_index,
            'edge_density': edge_density,
            'total_edge_length': total_edge,
            'mean_shape_index': mean_shape_index,
            'connectivity': connectivity,
            'fragmentation_index': 1 - largest_patch_index,  # Higher = more fragmented
            'patch_density': n_patches / (total_cells * cell_size ** 2) if total_cells > 0 else 0
        }

    def carbon_cycle_model(self,
                          vegetation_biomass: float,
                          soil_carbon: float,
                          temperature: float,
                          precipitation: float,
                          time_years: float = 50) -> Dict[str, np.ndarray]:
        """
        Ecosystem carbon cycle model.
        Tracks carbon pools and fluxes.
        """
        def derivatives(carbon_pools, t):
            veg_c, soil_c, atm_c = carbon_pools

            # Temperature and moisture effects
            temp_factor = 2.0 ** ((temperature - 10) / 10)  # Q10 = 2
            moisture_factor = min(1, precipitation / 1000)  # Optimal at 1000mm

            # GPP (Gross Primary Production)
            gpp = 10 * moisture_factor * (1 - np.exp(-0.001 * veg_c))  # tC/ha/year

            # Plant respiration
            resp_plant = 0.5 * gpp * temp_factor

            # NPP (Net Primary Production)
            npp = gpp - resp_plant

            # Litterfall
            litterfall = 0.1 * veg_c

            # Soil respiration
            resp_soil = 0.05 * soil_c * temp_factor * moisture_factor

            # Derivatives
            dveg_dt = npp - litterfall
            dsoil_dt = litterfall - resp_soil
            datm_dt = resp_plant + resp_soil - gpp

            return [dveg_dt, dsoil_dt, datm_dt]

        # Initial conditions (tC/ha)
        initial = [vegetation_biomass, soil_carbon, 0]

        # Time points
        t = np.linspace(0, time_years, int(time_years * 12))  # Monthly

        # Solve ODE
        solution = integrate.odeint(derivatives, initial, t)

        # Calculate additional metrics
        veg_carbon = solution[:, 0]
        soil_carbon_timeline = solution[:, 1]
        atm_exchange = solution[:, 2]

        # Total ecosystem carbon
        total_carbon = veg_carbon + soil_carbon_timeline

        # Net ecosystem exchange (negative = sink)
        nee = np.gradient(atm_exchange, t)

        return {
            'time': t,
            'vegetation_carbon': veg_carbon,
            'soil_carbon': soil_carbon_timeline,
            'atmosphere_exchange': atm_exchange,
            'total_ecosystem_carbon': total_carbon,
            'net_ecosystem_exchange': nee,
            'carbon_sequestration_rate': -np.mean(nee),
            'equilibrium_carbon': total_carbon[-1]
        }

    def demo(self):
        """Demonstrate ecology modeling capabilities"""
        print("=" * 60)
        print("ECOLOGY LAB - Advanced Ecosystem Modeling")
        print("=" * 60)

        # Lotka-Volterra dynamics
        print("\nPredator-Prey Dynamics (Fox-Rabbit):")
        initial = {'rabbit': 100, 'fox': 10}
        interactions = {
            ('rabbit', 'fox'): -0.01,  # Predation
            ('fox', 'rabbit'): 0.005    # Growth from predation
        }
        dynamics = self.lotka_volterra_dynamics(initial, interactions, time_years=5)
        print(f"Initial: Rabbits={initial['rabbit']}, Foxes={initial['fox']}")
        print(f"After 5 years: Rabbits={dynamics['rabbit'][-1]:.0f}, Foxes={dynamics['fox'][-1]:.0f}")

        # Food web analysis
        print("\n" + "=" * 60)
        print("Food Web Structure Analysis")
        print("=" * 60)
        species = ['grass', 'rabbit', 'deer', 'fox', 'wolf']
        food_web = self.food_web_structure(species)
        print(f"Species: {food_web['n_species']}")
        print(f"Links: {food_web['n_links']}")
        print(f"Connectance: {food_web['connectance']:.3f}")
        print(f"Mean trophic level: {food_web['mean_trophic_level']:.2f}")
        print(f"Keystone species: {food_web['keystone_species']}")

        # Biodiversity metrics
        print("\n" + "=" * 60)
        print("Biodiversity Assessment")
        print("=" * 60)
        abundances = np.array([100, 80, 60, 40, 20, 10, 5, 2, 1, 1])
        biodiv = self.biodiversity_metrics(abundances)
        print(f"Species richness: {biodiv['species_richness']}")
        print(f"Shannon index: {biodiv['shannon_index']:.2f}")
        print(f"Simpson index: {biodiv['simpson_index']:.3f}")
        print(f"Evenness: {biodiv['evenness']:.3f}")

        # Species-area relationship
        print("\n" + "=" * 60)
        print("Species-Area Relationship")
        print("=" * 60)
        sar = self.species_area_relationship()
        print(f"Model: {sar['equation']}")
        print(f"R²: {sar['r_squared']:.3f}")
        print(f"Estimated species in {self.area} km²: {sar['estimated_total_species']:.0f}")

        # Island biogeography
        print("\n" + "=" * 60)
        print("Island Biogeography")
        print("=" * 60)
        island = self.island_biogeography(island_area=10, distance_to_mainland=50)
        print(f"Equilibrium species richness: {island['equilibrium_richness']:.0f}")
        print(f"Immigration rate: {island['immigration_rate']:.2f} species/year")
        print(f"Extinction rate: {island['extinction_rate']:.2f} species/year")
        print(f"Time to equilibrium: {island['time_to_equilibrium_years']:.1f} years")

        # Ecosystem services
        print("\n" + "=" * 60)
        print("Ecosystem Services Valuation")
        print("=" * 60)
        land_use = {'forest': 40, 'grassland': 20, 'wetland': 5, 'agriculture': 30, 'urban': 5}
        services = self.ecosystem_services_valuation(land_use, population=50000)
        print(f"Total area: {sum(land_use.values())} ha")
        print(f"Total ecosystem service value: ${services['total_value']:,.0f}/year")
        print(f"Per capita value: ${services['total_value_per_capita']:,.2f}/year")
        print(f"Sustainability index: {services['sustainability_index']:.2f}")

        # Top services
        print("\nTop ecosystem services by value:")
        top_services = sorted(services['services'].items(),
                            key=lambda x: x[1]['value'],
                            reverse=True)[:3]
        for service, details in top_services:
            print(f"  {service}: ${details['value']:,.0f}/year")

        # Habitat fragmentation
        print("\n" + "=" * 60)
        print("Habitat Fragmentation Analysis")
        print("=" * 60)
        # Create fragmented landscape
        landscape = np.random.random((50, 50)) > 0.6  # 40% habitat
        fragmentation = self.habitat_fragmentation_analysis(landscape.astype(int))
        print(f"Habitat coverage: {fragmentation['habitat_proportion']*100:.1f}%")
        print(f"Number of patches: {fragmentation['number_of_patches']}")
        print(f"Mean patch size: {fragmentation['mean_patch_size']:.2f} units²")
        print(f"Fragmentation index: {fragmentation['fragmentation_index']:.3f}")
        print(f"Connectivity: {fragmentation['connectivity']:.3f}")

        print("\n" + "=" * 60)
        print("Ecology Lab Analysis Complete")
        print("Production-ready for conservation planning")
        print("=" * 60)


if __name__ == "__main__":
    lab = EcologyLab()
    lab.demo()