# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Metabolomics Engine - Metabolic pathway analysis and flux balance
Based on KEGG, BiGG Models, and Systems Biology literature
"""

import numpy as np
from scipy.optimize import linprog
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

@dataclass
class MetabolicReaction:
    """Metabolic reaction definition"""
    id: str
    name: str
    substrates: Dict[str, float]  # metabolite: stoichiometry
    products: Dict[str, float]
    reversible: bool
    flux_bounds: Tuple[float, float]  # (min, max) mmol/gDW/h
    gibbs_energy: float  # kJ/mol


class MetabolomicsEngine:
    """
    Production-ready metabolomics analysis engine

    References:
    - Orth et al., Nature Biotechnology 28:245 (2010)
    - Palsson, "Systems Biology" (2015)
    - KEGG pathway database
    - Human Metabolome Database (HMDB)
    """

    # Standard biochemical constants
    R = 8.314  # J/(mol*K) - Gas constant
    T = 310.15  # K - Body temperature (37°C)
    F = 96485.0  # C/mol - Faraday constant

    # Core metabolic pathways
    GLYCOLYSIS_REACTIONS = {
        'HK': MetabolicReaction(
            id='HK',
            name='Hexokinase',
            substrates={'Glucose': 1, 'ATP': 1},
            products={'G6P': 1, 'ADP': 1, 'H+': 1},
            reversible=False,
            flux_bounds=(0, 10.0),
            gibbs_energy=-16.7  # kJ/mol
        ),
        'PGI': MetabolicReaction(
            id='PGI',
            name='Phosphoglucose Isomerase',
            substrates={'G6P': 1},
            products={'F6P': 1},
            reversible=True,
            flux_bounds=(-10.0, 10.0),
            gibbs_energy=1.67
        ),
        'PFK': MetabolicReaction(
            id='PFK',
            name='Phosphofructokinase',
            substrates={'F6P': 1, 'ATP': 1},
            products={'FBP': 1, 'ADP': 1, 'H+': 1},
            reversible=False,
            flux_bounds=(0, 10.0),
            gibbs_energy=-14.2
        ),
        'PK': MetabolicReaction(
            id='PK',
            name='Pyruvate Kinase',
            substrates={'PEP': 1, 'ADP': 1, 'H+': 1},
            products={'Pyruvate': 1, 'ATP': 1},
            reversible=False,
            flux_bounds=(0, 10.0),
            gibbs_energy=-31.4
        )
    }

    # Standard metabolite concentrations (mM)
    METABOLITE_CONCENTRATIONS = {
        'Glucose': 5.0,  # Blood glucose
        'Lactate': 1.5,
        'Pyruvate': 0.1,
        'ATP': 5.0,
        'ADP': 0.5,
        'NAD+': 1.0,
        'NADH': 0.1,
        'CoA': 0.05,
        'Acetyl-CoA': 0.01,
        'O2': 0.2,
        'CO2': 1.3
    }

    def __init__(self):
        """Initialize metabolomics engine"""
        self.reactions = self.GLYCOLYSIS_REACTIONS.copy()
        self.metabolites = list(self.METABOLITE_CONCENTRATIONS.keys())

    def flux_balance_analysis(
        self,
        objective_reaction: str,
        constraints: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict:
        """
        Perform Flux Balance Analysis (FBA)

        Maximizes: v_objective
        Subject to: S * v = 0 (steady-state)
                   lb <= v <= ub

        Where:
        S = stoichiometric matrix
        v = flux vector
        """

        # Build stoichiometric matrix
        all_metabolites = set()
        for rxn in self.reactions.values():
            all_metabolites.update(rxn.substrates.keys())
            all_metabolites.update(rxn.products.keys())

        metabolites_list = sorted(all_metabolites)
        reactions_list = list(self.reactions.keys())

        n_metabolites = len(metabolites_list)
        n_reactions = len(reactions_list)

        S = np.zeros((n_metabolites, n_reactions))

        for j, rxn_id in enumerate(reactions_list):
            rxn = self.reactions[rxn_id]
            for met, coef in rxn.substrates.items():
                i = metabolites_list.index(met)
                S[i, j] -= coef
            for met, coef in rxn.products.items():
                i = metabolites_list.index(met)
                S[i, j] += coef

        # Set up linear programming problem
        # Maximize objective = minimize -objective
        c = np.zeros(n_reactions)
        obj_idx = reactions_list.index(objective_reaction)
        c[obj_idx] = -1.0  # Maximize by minimizing negative

        # Bounds
        bounds = []
        for rxn_id in reactions_list:
            rxn = self.reactions[rxn_id]
            if constraints and rxn_id in constraints:
                bounds.append(constraints[rxn_id])
            else:
                bounds.append(rxn.flux_bounds)

        # Equality constraints (steady-state)
        A_eq = S
        b_eq = np.zeros(n_metabolites)

        # Solve LP
        result = linprog(
            c=c,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method='highs'
        )

        if result.success:
            fluxes = {rxn_id: result.x[i] for i, rxn_id in enumerate(reactions_list)}
            objective_flux = -result.fun  # Convert back from minimization

            return {
                'success': True,
                'objective_reaction': objective_reaction,
                'objective_flux': objective_flux,
                'fluxes': fluxes,
                'stoichiometric_matrix': S.tolist(),
                'metabolites': metabolites_list,
                'reactions': reactions_list
            }
        else:
            return {
                'success': False,
                'message': 'FBA optimization failed',
                'status': result.message
            }

    def calculate_gibbs_free_energy(
        self,
        reaction_id: str,
        concentrations: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Calculate Gibbs free energy change for a reaction

        ΔG = ΔG° + RT*ln(Q)

        Where Q = [products]/[substrates]
        """

        if reaction_id not in self.reactions:
            raise ValueError(f"Unknown reaction: {reaction_id}")

        rxn = self.reactions[reaction_id]

        if concentrations is None:
            concentrations = self.METABOLITE_CONCENTRATIONS

        # Standard Gibbs energy
        delta_g_standard = rxn.gibbs_energy

        # Calculate reaction quotient Q
        Q = 1.0
        for substrate, coef in rxn.substrates.items():
            if substrate in concentrations:
                Q /= concentrations[substrate] ** coef

        for product, coef in rxn.products.items():
            if product in concentrations:
                Q *= concentrations[product] ** coef

        # Actual Gibbs energy
        delta_g = delta_g_standard + (self.R * self.T / 1000.0) * np.log(Q)

        # Equilibrium constant
        K_eq = np.exp(-delta_g_standard * 1000.0 / (self.R * self.T))

        return {
            'reaction_id': reaction_id,
            'reaction_name': rxn.name,
            'delta_g_standard_kJ_per_mol': delta_g_standard,
            'delta_g_actual_kJ_per_mol': delta_g,
            'reaction_quotient': Q,
            'equilibrium_constant': K_eq,
            'spontaneous': delta_g < 0,
            'concentrations_used': concentrations
        }

    def metabolic_flux_ratio_analysis(
        self,
        glucose_uptake_rate: float = 10.0  # mmol/gDW/h
    ) -> Dict:
        """
        Analyze metabolic flux distribution ratios

        Key ratios:
        - Glycolytic flux vs oxidative phosphorylation
        - Lactate production vs CO2 production
        - ATP production efficiency
        """

        # Glycolysis: 1 Glucose -> 2 Pyruvate + 2 ATP
        glycolytic_atp = 2 * glucose_uptake_rate

        # Assume 30% pyruvate -> lactate (anaerobic)
        #         70% pyruvate -> TCA (aerobic)
        anaerobic_fraction = 0.30
        aerobic_fraction = 0.70

        pyruvate_production = 2 * glucose_uptake_rate
        lactate_production = anaerobic_fraction * pyruvate_production
        tca_pyruvate = aerobic_fraction * pyruvate_production

        # TCA cycle: 1 Pyruvate -> 12.5 ATP (via NADH/FADH2)
        oxidative_atp = tca_pyruvate * 12.5

        total_atp = glycolytic_atp + oxidative_atp

        # ATP yield per glucose
        atp_per_glucose = total_atp / glucose_uptake_rate

        return {
            'glucose_uptake_mmol_per_gDW_per_h': glucose_uptake_rate,
            'pyruvate_production': pyruvate_production,
            'lactate_production': lactate_production,
            'lactate_to_pyruvate_ratio': lactate_production / pyruvate_production,
            'glycolytic_atp_production': glycolytic_atp,
            'oxidative_atp_production': oxidative_atp,
            'total_atp_production': total_atp,
            'atp_per_glucose': atp_per_glucose,
            'glycolytic_fraction': glycolytic_atp / total_atp,
            'oxidative_fraction': oxidative_atp / total_atp,
            'warburg_effect_indicator': 'High' if lactate_production / pyruvate_production > 0.5 else 'Normal'
        }

    def biomarker_discovery_analysis(
        self,
        disease_state: str = 'diabetes'
    ) -> Dict:
        """
        Identify potential metabolic biomarkers for disease states

        Based on known metabolic dysregulation patterns
        """

        biomarkers = {}

        if disease_state == 'diabetes':
            biomarkers = {
                'Glucose': {
                    'normal_range_mM': (4.0, 6.0),
                    'disease_range_mM': (7.0, 15.0),
                    'fold_change': 2.0,
                    'specificity': 'High',
                    'sensitivity': 'High'
                },
                'HbA1c': {
                    'normal_range_percent': (4.0, 5.6),
                    'disease_range_percent': (6.5, 12.0),
                    'fold_change': 1.5,
                    'specificity': 'High',
                    'sensitivity': 'High'
                },
                'Fructosamine': {
                    'normal_range_umol_per_L': (200, 285),
                    'disease_range_umol_per_L': (300, 500),
                    'fold_change': 1.4,
                    'specificity': 'Medium',
                    'sensitivity': 'Medium'
                }
            }
        elif disease_state == 'cancer':
            biomarkers = {
                'Lactate': {
                    'normal_range_mM': (0.5, 2.0),
                    'disease_range_mM': (3.0, 8.0),
                    'fold_change': 3.0,
                    'specificity': 'Medium',
                    'sensitivity': 'High',
                    'mechanism': 'Warburg effect'
                },
                'Glutamine': {
                    'normal_range_mM': (0.4, 0.8),
                    'disease_range_mM': (0.1, 0.3),
                    'fold_change': -2.5,
                    'specificity': 'Medium',
                    'sensitivity': 'Medium',
                    'mechanism': 'Glutamine addiction'
                }
            }
        elif disease_state == 'cardiovascular':
            biomarkers = {
                'TMAO': {
                    'normal_range_uM': (1, 5),
                    'disease_range_uM': (10, 50),
                    'fold_change': 5.0,
                    'specificity': 'High',
                    'sensitivity': 'Medium',
                    'mechanism': 'Gut microbiome'
                },
                'Homocysteine': {
                    'normal_range_uM': (5, 15),
                    'disease_range_uM': (20, 100),
                    'fold_change': 3.0,
                    'specificity': 'Medium',
                    'sensitivity': 'High'
                }
            }

        return {
            'disease_state': disease_state,
            'biomarkers': biomarkers,
            'total_biomarkers': len(biomarkers),
            'high_specificity_count': sum(1 for b in biomarkers.values() if b.get('specificity') == 'High'),
            'high_sensitivity_count': sum(1 for b in biomarkers.values() if b.get('sensitivity') == 'High')
        }


def run_metabolomics_demo():
    """Demonstrate metabolomics engine capabilities"""

    results = {}

    print("=" * 60)
    print("METABOLOMICS LABORATORY - Production Demo")
    print("=" * 60)

    engine = MetabolomicsEngine()

    # 1. Flux Balance Analysis
    print("\n1. Performing Flux Balance Analysis...")
    fba_result = engine.flux_balance_analysis(objective_reaction='PK')

    if fba_result['success']:
        print(f"  Objective flux (Pyruvate Kinase): {fba_result['objective_flux']:.3f} mmol/gDW/h")
        print(f"  Flux distribution:")
        for rxn_id, flux in fba_result['fluxes'].items():
            print(f"    {rxn_id}: {flux:.3f} mmol/gDW/h")
    else:
        print(f"  FBA failed: {fba_result['message']}")

    results['flux_balance_analysis'] = {
        'objective_flux': fba_result.get('objective_flux'),
        'fluxes': fba_result.get('fluxes')
    }

    # 2. Gibbs Free Energy
    print("\n2. Calculating Gibbs Free Energy...")
    for rxn_id in ['HK', 'PFK', 'PK']:
        gibbs = engine.calculate_gibbs_free_energy(rxn_id)
        print(f"  {gibbs['reaction_name']}:")
        print(f"    ΔG°: {gibbs['delta_g_standard_kJ_per_mol']:.2f} kJ/mol")
        print(f"    ΔG: {gibbs['delta_g_actual_kJ_per_mol']:.2f} kJ/mol")
        print(f"    Spontaneous: {gibbs['spontaneous']}")

    results['gibbs_energy'] = {
        rxn_id: engine.calculate_gibbs_free_energy(rxn_id)
        for rxn_id in ['HK', 'PFK', 'PK']
    }

    # 3. Metabolic Flux Ratios
    print("\n3. Analyzing Metabolic Flux Distribution...")
    flux_ratios = engine.metabolic_flux_ratio_analysis(glucose_uptake_rate=10.0)
    print(f"  Glucose uptake: {flux_ratios['glucose_uptake_mmol_per_gDW_per_h']:.2f} mmol/gDW/h")
    print(f"  ATP per glucose: {flux_ratios['atp_per_glucose']:.2f}")
    print(f"  Glycolytic fraction: {flux_ratios['glycolytic_fraction']:.1%}")
    print(f"  Oxidative fraction: {flux_ratios['oxidative_fraction']:.1%}")
    print(f"  Warburg effect: {flux_ratios['warburg_effect_indicator']}")

    results['flux_ratios'] = flux_ratios

    # 4. Biomarker Discovery
    print("\n4. Identifying Disease Biomarkers...")
    for disease in ['diabetes', 'cancer', 'cardiovascular']:
        print(f"\n  {disease.upper()}:")
        biomarkers = engine.biomarker_discovery_analysis(disease_state=disease)
        print(f"    Total biomarkers: {biomarkers['total_biomarkers']}")
        print(f"    High specificity: {biomarkers['high_specificity_count']}")
        print(f"    High sensitivity: {biomarkers['high_sensitivity_count']}")

        for marker, data in biomarkers['biomarkers'].items():
            print(f"      {marker}: {data['fold_change']}x change")

    results['biomarker_discovery'] = {
        disease: engine.biomarker_discovery_analysis(disease_state=disease)
        for disease in ['diabetes', 'cancer', 'cardiovascular']
    }

    print("\n" + "=" * 60)
    print("METABOLOMICS LAB DEMO COMPLETE")
    print("=" * 60)

    return results


if __name__ == '__main__':
    results = run_metabolomics_demo()

    # Save results
    with open('/Users/noone/QuLabInfinite/metabolomics_lab_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to: metabolomics_lab_results.json")
