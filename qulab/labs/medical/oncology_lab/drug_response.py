"""
Drug response heuristics for the QuLabInfinite oncology sandbox.

The structures below model basic pharmacokinetic and pharmacodynamic behaviour
using a one-compartment approximation and literature-inspired defaults. Numbers
were collated from publicly cited sources where available and rounded to retain
clarity. These utilities are illustrative and are not guaranteed to reproduce
patient-level outcomes.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
import json


class DrugClass(Enum):
    """Drug classifications"""
    CHEMOTHERAPY = "chemotherapy"
    TARGETED_THERAPY = "targeted_therapy"
    IMMUNOTHERAPY = "immunotherapy"
    HORMONE_THERAPY = "hormone_therapy"
    METABOLIC_INHIBITOR = "metabolic_inhibitor"
    EPIGENETIC = "epigenetic"
    ANTIANGIOGENIC = "antiangiogenic"


@dataclass
class PharmacokineticModel:
    """
    Pharmacokinetic (PK) parameters for a one-compartment model.
    Defaults loosely reflect values reported in drug labels and clinical
    pharmacology references, but should be customised for precise studies.
    """
    # Absorption
    bioavailability: float = 1.0  # 0-1 (IV = 1.0)
    absorption_rate: float = 1.0  # h^-1
    tmax: float = 1.0  # Hours to peak concentration

    # Distribution
    volume_of_distribution: float = 50.0  # L (total body water ~42L)
    protein_binding: float = 0.9  # Fraction bound to plasma proteins
    tissue_penetration: float = 0.5  # 0-1 (tumor penetration)

    # Metabolism
    clearance: float = 10.0  # L/h
    half_life: float = 24.0  # Hours
    elimination_rate: float = 0.693 / 24.0  # k_e = ln(2)/t_half

    # Special considerations
    active_metabolites: List[str] = field(default_factory=list)
    prodrug: bool = False

    def __post_init__(self):
        """
        Align elimination constant with the selected half-life so that
        downstream concentration curves stay consistent with the PK inputs.
        """
        if self.half_life and self.half_life > 0:
            self.elimination_rate = np.log(2) / self.half_life


@dataclass
class Drug:
    """
    Aggregate of PK/PD settings and qualitative annotations for a drug.
    Populate with literature-informed numbers when available; defaults here
    serve as illustrative placeholders rather than authoritative data.
    """
    # Identity
    name: str
    generic_name: str
    drug_class: DrugClass

    # Pharmacokinetics
    pk_model: PharmacokineticModel

    # Pharmacodynamics - "What the drug does to the body"
    mechanism_of_action: str
    target_proteins: List[str]  # e.g., ["EGFR", "HER2"]
    molecular_weight: float  # g/mol - actual molecular weight
    ic50: float  # μM - concentration for 50% inhibition
    ec50: float  # μM - concentration for 50% effect
    emax: float = 1.0  # Maximum effect (0-1, where 1 = 100% kill)
    hill_coefficient: float = 1.0  # Hill coefficient for dose-response curve

    # Cell cycle specificity
    cell_cycle_specific: bool = False
    target_phases: List[str] = field(default_factory=list)  # ["S", "M"]

    # Resistance
    resistance_mutations: List[str] = field(default_factory=list)
    resistance_mechanisms: List[str] = field(default_factory=list)

    # Toxicity
    myelosuppression: float = 0.0  # 0-1 scale
    cardiotoxicity: float = 0.0
    neurotoxicity: float = 0.0
    hepatotoxicity: float = 0.0

    # Clinical dosing
    standard_dose_mg: float = 100.0
    dosing_interval_hours: float = 24.0
    route: str = "IV"  # IV, oral, subcutaneous

    # FDA status
    fda_approved: bool = True
    approval_year: Optional[int] = None
    approved_indications: List[str] = field(default_factory=list)

    def calculate_concentration(self, dose_mg: float, time_hours: float, weight_kg: float = 70.0) -> float:
        """
        Calculate plasma concentration using 1-compartment PK model
        C(t) = (Dose / Vd) * exp(-k_e * t)
        """
        # Convert dose to amount
        dose_amount = dose_mg * self.pk_model.bioavailability

        # Volume of distribution
        vd = self.pk_model.volume_of_distribution

        # Peak concentration
        c_max = dose_amount / vd  # mg/L

        # Concentration at time t
        c_t = c_max * np.exp(-self.pk_model.elimination_rate * time_hours)

        # Convert to μM using actual molecular weight
        c_uM = (c_t / self.molecular_weight) * 1000.0  # Convert mg/L to μM

        return c_uM

    def calculate_tumor_concentration(self, plasma_conc: float) -> float:
        """
        Calculate tumor concentration from plasma concentration
        Accounts for blood-tumor barrier penetration
        """
        return plasma_conc * self.pk_model.tissue_penetration

    def calculate_effect(self, concentration: float) -> float:
        """
        Calculate drug effect using Hill equation (sigmoid dose-response)
        E = Emax * C^n / (EC50^n + C^n)
        """
        if concentration <= 0:
            return 0.0

        n = self.hill_coefficient
        effect = self.emax * (concentration ** n) / ((self.ec50 ** n) + (concentration ** n))
        return effect

    def calculate_cell_kill_probability(self, concentration: float, dt: float) -> float:
        """
        Calculate probability of cell death due to drug
        Based on concentration and exposure time
        """
        effect = self.calculate_effect(concentration)

        # Kill rate increases with effect
        # Typical chemotherapy: 1-5 log kill per cycle
        base_kill_rate = 0.1  # per hour at Emax
        kill_rate = base_kill_rate * effect

        # Probability of death in time dt
        prob_death = 1.0 - np.exp(-kill_rate * dt)

        return prob_death


# ============================================================================
# REAL DRUG DATABASE - FDA-approved and experimental drugs with actual parameters
# ============================================================================

DRUG_DATABASE = {
    # Chemotherapy - DNA damaging agents
    "cisplatin": Drug(
        name="Cisplatin",
        generic_name="cis-diamminedichloroplatinum(II)",
        drug_class=DrugClass.CHEMOTHERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=1.0,  # IV only
            volume_of_distribution=20.0,  # L
            half_life=0.8,  # hours (initial), 24-127h (terminal)
            clearance=15.0,  # L/h
            protein_binding=0.95,
            tissue_penetration=0.3,  # Poor tumor penetration
        ),
        mechanism_of_action="DNA crosslinking, induces apoptosis",
        target_proteins=["DNA"],
        molecular_weight=300.1,  # g/mol - actual cisplatin MW
        ic50=1.5,  # μM
        ec50=2.0,  # μM
        emax=0.95,
        hill_coefficient=2.0,  # Typical for chemotherapy agents
        cell_cycle_specific=False,
        resistance_mutations=["ERCC1_overexpression", "ATP7B_mutation"],
        myelosuppression=0.6,
        neurotoxicity=0.5,
        hepatotoxicity=0.3,
        standard_dose_mg=135,  # converted from 75.0 mg/m² (1.8 m² -> 135.0 mg)

        dosing_interval_hours=21 * 24,  # Every 3 weeks
        route="IV",
        fda_approved=True,
        approval_year=1978,
        approved_indications=["Testicular cancer", "Ovarian cancer", "Bladder cancer", "Lung cancer"],
    ),

    "doxorubicin": Drug(
        name="Doxorubicin",
        generic_name="doxorubicin hydrochloride",
        drug_class=DrugClass.CHEMOTHERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.05,  # Poor oral bioavailability, usually IV
            volume_of_distribution=800.0,  # L - extensive tissue distribution
            half_life=30.0,  # hours
            clearance=45.0,  # L/h
            protein_binding=0.75,
            tissue_penetration=0.6,
        ),
        mechanism_of_action="DNA intercalation, topoisomerase II inhibition",
        target_proteins=["TOP2A", "TOP2B", "DNA"],
        molecular_weight=543.5,  # g/mol - doxorubicin HCl
        ic50=0.5,  # μM
        ec50=0.8,  # μM
        emax=0.9,
        hill_coefficient=2.5,  # Steep dose-response
        cell_cycle_specific=True,
        target_phases=["S", "G2"],
        resistance_mutations=["MDR1_overexpression", "TOP2A_mutation"],
        resistance_mechanisms=["P-glycoprotein efflux"],
        myelosuppression=0.8,
        cardiotoxicity=0.7,  # Cumulative, dose-limiting
        standard_dose_mg=108,  # converted from 60.0 mg/m² (1.8 m² -> 108.0 mg)

        dosing_interval_hours=21 * 24,
        route="IV",
        fda_approved=True,
        approval_year=1974,
        approved_indications=["Breast cancer", "Leukemia", "Lymphoma", "Many solid tumors"],
    ),

    "paclitaxel": Drug(
        name="Paclitaxel",
        generic_name="paclitaxel",
        drug_class=DrugClass.CHEMOTHERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.3,  # Usually IV
            volume_of_distribution=200.0,  # L
            half_life=20.0,  # hours
            clearance=15.0,  # L/h
            protein_binding=0.95,
            tissue_penetration=0.4,
        ),
        mechanism_of_action="Microtubule stabilization, mitotic arrest",
        target_proteins=["TUBB", "TUBA"],  # Tubulin
        molecular_weight=853.9,  # g/mol - paclitaxel
        ic50=0.01,  # μM - very potent
        ec50=0.05,  # μM
        emax=0.95,
        hill_coefficient=3.0,  # Very steep (M-phase specific)
        cell_cycle_specific=True,
        target_phases=["M"],  # M-phase specific
        resistance_mutations=["TUBB3_overexpression", "MDR1_overexpression"],
        myelosuppression=0.7,
        neurotoxicity=0.6,
        standard_dose_mg=315,  # converted from 175.0 mg/m² (1.8 m² -> 315.0 mg)

        dosing_interval_hours=21 * 24,
        route="IV",
        fda_approved=True,
        approval_year=1992,
        approved_indications=["Ovarian cancer", "Breast cancer", "Lung cancer"],
    ),

    # Targeted therapy
    "erlotinib": Drug(
        name="Erlotinib",
        generic_name="erlotinib hydrochloride",
        drug_class=DrugClass.TARGETED_THERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.6,  # Oral
            volume_of_distribution=230.0,  # L
            half_life=36.0,  # hours
            clearance=6.4,  # L/h
            protein_binding=0.93,
            tissue_penetration=0.5,
        ),
        mechanism_of_action="EGFR tyrosine kinase inhibitor",
        target_proteins=["EGFR"],
        molecular_weight=393.4,  # g/mol - erlotinib HCl
        ic50=0.002,  # μM (2 nM) - very specific
        ec50=0.01,  # μM
        emax=0.85,  # High efficacy in EGFR-mutant tumors
        hill_coefficient=2.0,
        cell_cycle_specific=False,
        resistance_mutations=["EGFR_T790M", "MET_amplification"],
        standard_dose_mg=150.0,
        dosing_interval_hours=24.0,
        route="Oral",
        fda_approved=True,
        approval_year=2004,
        approved_indications=["NSCLC (EGFR mutation)", "Pancreatic cancer"],
    ),

    # Metabolic inhibitors
    "metformin": Drug(
        name="Metformin",
        generic_name="metformin hydrochloride",
        drug_class=DrugClass.METABOLIC_INHIBITOR,
        pk_model=PharmacokineticModel(
            bioavailability=0.5,  # Oral
            volume_of_distribution=650.0,  # L
            half_life=5.0,  # hours
            clearance=600.0,  # L/h - rapid clearance
            protein_binding=0.0,  # Negligible
            tissue_penetration=0.3,
        ),
        mechanism_of_action="Complex I inhibitor, AMPK activator, reduces glucose",
        target_proteins=["NDUFS1", "AMPK"],  # Complex I, AMPK
        molecular_weight=129.2,  # g/mol - metformin HCl
        ic50=50.0,  # μM - relatively high (not cytotoxic)
        ec50=100.0,  # μM
        emax=0.3,  # Modest anti-tumor effect
        hill_coefficient=1.5,  # Metabolic modulation
        cell_cycle_specific=False,
        myelosuppression=0.0,
        standard_dose_mg=1000.0,  # mg (for diabetes)
        dosing_interval_hours=12.0,
        route="Oral",
        fda_approved=True,
        approval_year=1994,
        approved_indications=["Type 2 diabetes"],  # Off-label for cancer
    ),

    "dichloroacetate": Drug(
        name="Dichloroacetate",
        generic_name="dichloroacetic acid",
        drug_class=DrugClass.METABOLIC_INHIBITOR,
        pk_model=PharmacokineticModel(
            bioavailability=0.9,  # Oral
            volume_of_distribution=40.0,  # L
            half_life=1.0,  # hours
            clearance=40.0,  # L/h
            protein_binding=0.0,
            tissue_penetration=0.7,  # Good penetration
        ),
        mechanism_of_action="PDK inhibitor, reverses Warburg effect",
        target_proteins=["PDK1", "PDK2", "PDK3", "PDK4"],
        molecular_weight=128.9,  # g/mol - dichloroacetate
        ic50=10.0,  # mM (high doses needed)
        ec50=5.0,  # mM
        emax=0.5,
        hill_coefficient=1.0,  # Shallow curve (metabolic)
        cell_cycle_specific=False,
        resistance_mechanisms=["Alternative metabolic pathways"],
        neurotoxicity=0.4,  # Peripheral neuropathy
        standard_dose_mg=1750,  # converted from 25.0 mg/kg (70 kg -> 1750 mg)

        dosing_interval_hours=12.0,
        route="Oral",
        fda_approved=False,  # Experimental for cancer
        approved_indications=[],  # Used off-label for lactic acidosis
    ),

    # Antiangiogenic
    "bevacizumab": Drug(
        name="Bevacizumab",
        generic_name="bevacizumab",
        drug_class=DrugClass.ANTIANGIOGENIC,
        pk_model=PharmacokineticModel(
            bioavailability=1.0,  # IV monoclonal antibody
            volume_of_distribution=3.0,  # L (plasma volume)
            half_life=20.0 * 24,  # 20 days!
            clearance=0.2,  # L/h
            protein_binding=0.0,  # Antibody
            tissue_penetration=0.2,  # Poor tumor penetration
        ),
        mechanism_of_action="VEGF-A inhibitor, blocks angiogenesis",
        target_proteins=["VEGFA"],
        molecular_weight=149000.0,  # g/mol - monoclonal antibody (very large)
        ic50=0.0005,  # μM (0.5 nM) - extremely potent
        ec50=0.001,  # μM
        emax=0.7,  # Slows growth, rarely kills directly
        hill_coefficient=1.5,  # Indirect effect
        cell_cycle_specific=False,
        standard_dose_mg=350,  # converted from 5.0 mg/kg (70 kg -> 350 mg)

        dosing_interval_hours=14 * 24,  # Every 2 weeks
        route="IV",
        fda_approved=True,
        approval_year=2004,
        approved_indications=["Colorectal cancer", "Lung cancer", "Glioblastoma", "Ovarian cancer"],
    ),

    # ============================================================================
    # ADDITIONAL CHEMOTHERAPY AGENTS
    # ============================================================================

    "carboplatin": Drug(
        name="Carboplatin",
        generic_name="carboplatin",
        drug_class=DrugClass.CHEMOTHERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=1.0,  # IV only
            volume_of_distribution=16.0,  # L
            half_life=2.0,  # hours
            clearance=8.0,  # L/h
            protein_binding=0.0,  # Minimal
            tissue_penetration=0.4,
        ),
        mechanism_of_action="DNA crosslinking, similar to cisplatin but less toxic",
        target_proteins=["DNA"],
        molecular_weight=371.3,
        ic50=5.0,  # μM - less potent than cisplatin
        ec50=8.0,
        emax=0.9,
        hill_coefficient=2.0,
        cell_cycle_specific=False,
        resistance_mutations=["ERCC1_overexpression"],
        myelosuppression=0.7,
        neurotoxicity=0.2,  # Much less than cisplatin
        standard_dose_mg=720,  # converted from 400.0 mg/m² (1.8 m² -> 720.0 mg)

        dosing_interval_hours=21 * 24,
        route="IV",
        fda_approved=True,
        approval_year=1989,
        approved_indications=["Ovarian cancer", "Lung cancer"],
    ),

    "5-fluorouracil": Drug(
        name="5-Fluorouracil",
        generic_name="5-FU",
        drug_class=DrugClass.CHEMOTHERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.5,  # Variable oral
            volume_of_distribution=22.0,  # L
            half_life=0.25,  # hours (very short)
            clearance=80.0,  # L/h (rapid)
            protein_binding=0.1,
            tissue_penetration=0.5,
        ),
        mechanism_of_action="Thymidylate synthase inhibitor, disrupts DNA synthesis",
        target_proteins=["TYMS", "DNA"],
        molecular_weight=130.1,
        ic50=1.0,  # μM
        ec50=2.0,
        emax=0.85,
        hill_coefficient=2.0,
        cell_cycle_specific=True,
        target_phases=["S"],
        resistance_mutations=["TYMS_overexpression", "DPD_deficiency"],
        myelosuppression=0.6,
        hepatotoxicity=0.3,
        standard_dose_mg=900,  # converted from 500.0 mg/m² (1.8 m² -> 900.0 mg)

        dosing_interval_hours=24.0,
        route="IV",
        fda_approved=True,
        approval_year=1962,
        approved_indications=["Colorectal cancer", "Breast cancer", "Gastric cancer"],
    ),

    "gemcitabine": Drug(
        name="Gemcitabine",
        generic_name="gemcitabine hydrochloride",
        drug_class=DrugClass.CHEMOTHERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.1,  # IV typically
            volume_of_distribution=50.0,  # L
            half_life=0.7,  # hours
            clearance=70.0,  # L/h
            protein_binding=0.0,
            tissue_penetration=0.4,
        ),
        mechanism_of_action="Nucleoside analog, inhibits DNA synthesis",
        target_proteins=["RRM1", "DNA"],
        molecular_weight=299.7,
        ic50=0.05,  # μM
        ec50=0.1,
        emax=0.9,
        hill_coefficient=2.5,
        cell_cycle_specific=True,
        target_phases=["S"],
        resistance_mutations=["RRM1_overexpression"],
        myelosuppression=0.5,
        standard_dose_mg=1800,  # converted from 1000.0 mg/m² (1.8 m² -> 1800.0 mg)

        dosing_interval_hours=7 * 24,
        route="IV",
        fda_approved=True,
        approval_year=1996,
        approved_indications=["Pancreatic cancer", "Lung cancer", "Breast cancer", "Ovarian cancer"],
    ),

    "temozolomide": Drug(
        name="Temozolomide",
        generic_name="temozolomide",
        drug_class=DrugClass.CHEMOTHERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=1.0,  # Excellent oral
            volume_of_distribution=17.0,  # L
            half_life=1.8,  # hours
            clearance=9.5,  # L/h
            protein_binding=0.15,
            tissue_penetration=0.9,  # Excellent CNS penetration
        ),
        mechanism_of_action="DNA methylation, alkylating agent",
        target_proteins=["DNA"],
        molecular_weight=194.2,
        ic50=0.3,  # μM
        ec50=0.6,
        emax=0.85,
        hill_coefficient=2.0,
        cell_cycle_specific=False,
        resistance_mutations=["MGMT_expression"],
        myelosuppression=0.4,
        standard_dose_mg=270,  # converted from 150.0 mg/m² (1.8 m² -> 270.0 mg)

        dosing_interval_hours=24.0,
        route="Oral",
        fda_approved=True,
        approval_year=1999,
        approved_indications=["Glioblastoma", "Anaplastic astrocytoma"],
    ),

    # ============================================================================
    # TARGETED THERAPY - More agents
    # ============================================================================

    "imatinib": Drug(
        name="Imatinib",
        generic_name="imatinib mesylate",
        drug_class=DrugClass.TARGETED_THERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.98,  # Excellent oral
            volume_of_distribution=435.0,  # L
            half_life=18.0,  # hours
            clearance=14.0,  # L/h
            protein_binding=0.95,
            tissue_penetration=0.5,
        ),
        mechanism_of_action="BCR-ABL tyrosine kinase inhibitor",
        target_proteins=["BCR-ABL", "KIT", "PDGFR"],
        molecular_weight=493.6,
        ic50=0.0001,  # μM (very specific)
        ec50=0.0005,
        emax=0.95,
        hill_coefficient=3.0,
        cell_cycle_specific=False,
        resistance_mutations=["BCR-ABL_T315I"],
        standard_dose_mg=400.0,
        dosing_interval_hours=24.0,
        route="Oral",
        fda_approved=True,
        approval_year=2001,
        approved_indications=["CML", "GIST"],
    ),

    "vemurafenib": Drug(
        name="Vemurafenib",
        generic_name="vemurafenib",
        drug_class=DrugClass.TARGETED_THERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.64,
            volume_of_distribution=106.0,  # L
            half_life=57.0,  # hours
            clearance=1.8,  # L/h
            protein_binding=0.99,
            tissue_penetration=0.4,
        ),
        mechanism_of_action="BRAF V600E kinase inhibitor",
        target_proteins=["BRAF"],
        molecular_weight=489.9,
        ic50=0.001,  # μM
        ec50=0.005,
        emax=0.9,
        hill_coefficient=2.5,
        cell_cycle_specific=False,
        resistance_mutations=["NRAS_mutation", "MEK_mutation"],
        standard_dose_mg=960.0,
        dosing_interval_hours=12.0,
        route="Oral",
        fda_approved=True,
        approval_year=2011,
        approved_indications=["Melanoma (BRAF V600E)"],
    ),

    "trastuzumab": Drug(
        name="Trastuzumab",
        generic_name="trastuzumab",
        drug_class=DrugClass.TARGETED_THERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=1.0,  # IV antibody
            volume_of_distribution=3.0,  # L
            half_life=28.5 * 24,  # 28.5 days
            clearance=0.127,  # L/h
            protein_binding=0.0,
            tissue_penetration=0.3,
        ),
        mechanism_of_action="HER2 antibody, blocks receptor signaling",
        target_proteins=["ERBB2"],
        molecular_weight=148000.0,  # Monoclonal antibody
        ic50=0.0001,  # μM
        ec50=0.0005,
        emax=0.75,
        hill_coefficient=2.0,
        cell_cycle_specific=False,
        resistance_mechanisms=["HER2_truncation", "PI3K_mutation"],
        cardiotoxicity=0.4,
        standard_dose_mg=420,  # converted from 6.0 mg/kg (70 kg -> 420 mg)

        dosing_interval_hours=21 * 24,
        route="IV",
        fda_approved=True,
        approval_year=1998,
        approved_indications=["Breast cancer (HER2+)", "Gastric cancer"],
    ),

    # ============================================================================
    # IMMUNOTHERAPY
    # ============================================================================

    "pembrolizumab": Drug(
        name="Pembrolizumab",
        generic_name="pembrolizumab",
        drug_class=DrugClass.IMMUNOTHERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=1.0,  # IV antibody
            volume_of_distribution=6.0,  # L
            half_life=26.0 * 24,  # 26 days
            clearance=0.23,  # L/h
            protein_binding=0.0,
            tissue_penetration=0.2,
        ),
        mechanism_of_action="PD-1 checkpoint inhibitor, restores T-cell function",
        target_proteins=["PDCD1"],
        molecular_weight=149000.0,
        ic50=0.0001,  # μM
        ec50=0.0003,
        emax=0.6,  # Variable response
        hill_coefficient=1.5,
        cell_cycle_specific=False,
        standard_dose_mg=200.0,  # mg flat dose
        dosing_interval_hours=21 * 24,
        route="IV",
        fda_approved=True,
        approval_year=2014,
        approved_indications=["Melanoma", "NSCLC", "MSI-H tumors", "Many others"],
    ),

    "nivolumab": Drug(
        name="Nivolumab",
        generic_name="nivolumab",
        drug_class=DrugClass.IMMUNOTHERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=1.0,
            volume_of_distribution=7.0,  # L
            half_life=25.0 * 24,  # 25 days
            clearance=0.28,  # L/h
            protein_binding=0.0,
            tissue_penetration=0.2,
        ),
        mechanism_of_action="PD-1 checkpoint inhibitor",
        target_proteins=["PDCD1"],
        molecular_weight=146000.0,
        ic50=0.0001,
        ec50=0.0003,
        emax=0.55,
        hill_coefficient=1.5,
        cell_cycle_specific=False,
        standard_dose_mg=240.0,  # mg flat dose
        dosing_interval_hours=14 * 24,
        route="IV",
        fda_approved=True,
        approval_year=2014,
        approved_indications=["Melanoma", "NSCLC", "RCC", "Hodgkin lymphoma"],
    ),

    # ============================================================================
    # NATURAL COMPOUNDS & VITAMINS
    # ============================================================================

    "vitamin_d3": Drug(
        name="Vitamin D3",
        generic_name="cholecalciferol",
        drug_class=DrugClass.METABOLIC_INHIBITOR,
        pk_model=PharmacokineticModel(
            bioavailability=0.8,  # Oral
            volume_of_distribution=100.0,  # L (fat-soluble)
            half_life=15.0 * 24,  # 15 days
            clearance=0.28,  # L/h
            protein_binding=0.99,  # Binds to DBP
            tissue_penetration=0.6,
        ),
        mechanism_of_action="VDR agonist, promotes differentiation and apoptosis",
        target_proteins=["VDR", "CYP24A1"],
        molecular_weight=384.6,
        ic50=50.0,  # μM (high)
        ec50=100.0,  # μM
        emax=0.3,  # Modest anti-cancer effect
        hill_coefficient=1.2,
        cell_cycle_specific=False,
        myelosuppression=0.0,
        standard_dose_mg=0.125,  # converted from 5000 IU (~125 μg)

        dosing_interval_hours=24.0,
        route="Oral",
        fda_approved=True,
        approval_year=1968,
        approved_indications=["Vitamin D deficiency"],
    ),

    "curcumin": Drug(
        name="Curcumin",
        generic_name="diferuloylmethane",
        drug_class=DrugClass.METABOLIC_INHIBITOR,
        pk_model=PharmacokineticModel(
            bioavailability=0.01,  # Very poor
            volume_of_distribution=1000.0,  # L (lipophilic)
            half_life=6.0,  # hours
            clearance=160.0,  # L/h (rapid metabolism)
            protein_binding=0.9,
            tissue_penetration=0.3,
        ),
        mechanism_of_action="NF-κB inhibitor, anti-inflammatory, pro-apoptotic",
        target_proteins=["NFKB1", "COX2", "TNF"],
        molecular_weight=368.4,
        ic50=10.0,  # μM
        ec50=20.0,
        emax=0.4,
        hill_coefficient=1.5,
        cell_cycle_specific=False,
        myelosuppression=0.0,
        standard_dose_mg=1000.0,  # mg (with piperine for absorption)
        dosing_interval_hours=8.0,
        route="Oral",
        fda_approved=False,
        approved_indications=[],
    ),

    "quercetin": Drug(
        name="Quercetin",
        generic_name="quercetin",
        drug_class=DrugClass.METABOLIC_INHIBITOR,
        pk_model=PharmacokineticModel(
            bioavailability=0.2,
            volume_of_distribution=150.0,  # L
            half_life=16.0,  # hours
            clearance=9.0,  # L/h
            protein_binding=0.95,
            tissue_penetration=0.4,
        ),
        mechanism_of_action="Antioxidant, PI3K/Akt inhibitor, senolytic",
        target_proteins=["PIK3CA", "AKT1"],
        molecular_weight=302.2,
        ic50=15.0,  # μM
        ec50=30.0,
        emax=0.35,
        hill_coefficient=1.3,
        cell_cycle_specific=False,
        standard_dose_mg=500.0,
        dosing_interval_hours=12.0,
        route="Oral",
        fda_approved=False,
        approved_indications=[],
    ),

    "resveratrol": Drug(
        name="Resveratrol",
        generic_name="resveratrol",
        drug_class=DrugClass.METABOLIC_INHIBITOR,
        pk_model=PharmacokineticModel(
            bioavailability=0.05,  # Very low
            volume_of_distribution=200.0,  # L
            half_life=9.0,  # hours
            clearance=20.0,  # L/h
            protein_binding=0.98,
            tissue_penetration=0.3,
        ),
        mechanism_of_action="SIRT1 activator, anti-inflammatory, anti-angiogenic",
        target_proteins=["SIRT1", "PTGS2"],
        molecular_weight=228.2,
        ic50=20.0,  # μM
        ec50=40.0,
        emax=0.3,
        hill_coefficient=1.2,
        cell_cycle_specific=False,
        standard_dose_mg=500.0,
        dosing_interval_hours=12.0,
        route="Oral",
        fda_approved=False,
        approved_indications=[],
    ),

    "vitamin_c": Drug(
        name="Vitamin C",
        generic_name="ascorbic acid",
        drug_class=DrugClass.METABOLIC_INHIBITOR,
        pk_model=PharmacokineticModel(
            bioavailability=0.9,  # Good oral at low doses
            volume_of_distribution=20.0,  # L
            half_life=2.0,  # hours
            clearance=10.0,  # L/h
            protein_binding=0.25,
            tissue_penetration=0.8,
        ),
        mechanism_of_action="Pro-oxidant at high doses, generates H2O2 in tumor",
        target_proteins=["Extracellular"],
        molecular_weight=176.1,
        ic50=500.0,  # μM (very high doses needed IV)
        ec50=1000.0,
        emax=0.5,  # At very high IV doses
        hill_coefficient=1.0,
        cell_cycle_specific=False,
        standard_dose_mg=1000.0,  # mg oral (50-100g IV for cancer)
        dosing_interval_hours=8.0,
        route="Oral",
        fda_approved=True,
        approval_year=1939,
        approved_indications=["Scurvy", "Vitamin C deficiency"],
    ),

    "egcg": Drug(
        name="EGCG",
        generic_name="epigallocatechin gallate",
        drug_class=DrugClass.METABOLIC_INHIBITOR,
        pk_model=PharmacokineticModel(
            bioavailability=0.1,  # Poor
            volume_of_distribution=100.0,  # L
            half_life=4.0,  # hours
            clearance=25.0,  # L/h
            protein_binding=0.9,
            tissue_penetration=0.4,
        ),
        mechanism_of_action="EGFR inhibitor, antioxidant, pro-apoptotic",
        target_proteins=["EGFR", "VEGFR"],
        molecular_weight=458.4,
        ic50=10.0,  # μM
        ec50=20.0,
        emax=0.35,
        hill_coefficient=1.4,
        cell_cycle_specific=False,
        standard_dose_mg=400.0,  # mg (green tea extract)
        dosing_interval_hours=8.0,
        route="Oral",
        fda_approved=False,
        approved_indications=[],
    ),

    # ============================================================================
    # EXPERIMENTAL/OFF-LABEL DRUGS
    # ============================================================================

    "ivermectin": Drug(
        name="Ivermectin",
        generic_name="ivermectin",
        drug_class=DrugClass.METABOLIC_INHIBITOR,
        pk_model=PharmacokineticModel(
            bioavailability=0.5,
            volume_of_distribution=47.0,  # L/kg = ~3500L
            half_life=18.0,  # hours
            clearance=2.0,  # L/h
            protein_binding=0.93,
            tissue_penetration=0.4,
        ),
        mechanism_of_action="PAK1 inhibitor, Akt/mTOR inhibition, anti-mitotic",
        target_proteins=["PAK1", "AKT1", "MTOR"],
        molecular_weight=875.1,
        ic50=5.0,  # μM
        ec50=10.0,
        emax=0.5,
        hill_coefficient=1.8,
        cell_cycle_specific=False,
        neurotoxicity=0.2,
        standard_dose_mg=14,  # converted from 0.2 mg/kg (70 kg -> 14 mg)

        dosing_interval_hours=7 * 24,
        route="Oral",
        fda_approved=True,
        approval_year=1996,
        approved_indications=["Parasitic infections"],
    ),

    "fenbendazole": Drug(
        name="Fenbendazole",
        generic_name="fenbendazole",
        drug_class=DrugClass.METABOLIC_INHIBITOR,
        pk_model=PharmacokineticModel(
            bioavailability=0.5,
            volume_of_distribution=500.0,  # L (estimated)
            half_life=12.0,  # hours
            clearance=40.0,  # L/h
            protein_binding=0.85,
            tissue_penetration=0.5,
        ),
        mechanism_of_action="Tubulin inhibitor, disrupts microtubules, GLUT inhibitor",
        target_proteins=["TUBB", "SLC2A1"],
        molecular_weight=299.3,
        ic50=0.5,  # μM
        ec50=1.0,
        emax=0.6,
        hill_coefficient=2.0,
        cell_cycle_specific=True,
        target_phases=["M"],
        standard_dose_mg=222.0,  # mg (anecdotal human dose)
        dosing_interval_hours=24.0,
        route="Oral",
        fda_approved=False,  # Veterinary only
        approved_indications=["Parasitic infections (animals)"],
    ),

    "mebendazole": Drug(
        name="Mebendazole",
        generic_name="mebendazole",
        drug_class=DrugClass.METABOLIC_INHIBITOR,
        pk_model=PharmacokineticModel(
            bioavailability=0.2,  # Poor oral
            volume_of_distribution=100.0,  # L
            half_life=3.0,  # hours
            clearance=30.0,  # L/h
            protein_binding=0.95,
            tissue_penetration=0.4,
        ),
        mechanism_of_action="Tubulin polymerization inhibitor, VEGFR2 inhibitor",
        target_proteins=["TUBB", "KDR"],
        molecular_weight=295.3,
        ic50=0.3,  # μM
        ec50=0.8,
        emax=0.65,
        hill_coefficient=2.2,
        cell_cycle_specific=True,
        target_phases=["M"],
        hepatotoxicity=0.2,
        standard_dose_mg=100.0,  # mg (standard antiparasitic)
        dosing_interval_hours=12.0,
        route="Oral",
        fda_approved=True,
        approval_year=1974,
        approved_indications=["Helminthic infections"],
    ),

    "hydroxychloroquine": Drug(
        name="Hydroxychloroquine",
        generic_name="hydroxychloroquine sulfate",
        drug_class=DrugClass.METABOLIC_INHIBITOR,
        pk_model=PharmacokineticModel(
            bioavailability=0.74,
            volume_of_distribution=5000.0,  # L (huge Vd)
            half_life=40.0 * 24,  # 40 days!
            clearance=4.0,  # L/h
            protein_binding=0.5,
            tissue_penetration=0.6,
        ),
        mechanism_of_action="Autophagy inhibitor, raises lysosomal pH",
        target_proteins=["Autophagy pathway"],
        molecular_weight=335.9,
        ic50=10.0,  # μM
        ec50=20.0,
        emax=0.4,
        hill_coefficient=1.3,
        cell_cycle_specific=False,
        standard_dose_mg=400.0,  # mg
        dosing_interval_hours=24.0,
        route="Oral",
        fda_approved=True,
        approval_year=1955,
        approved_indications=["Malaria", "Lupus", "Rheumatoid arthritis"],
    ),

    "aspirin": Drug(
        name="Aspirin",
        generic_name="acetylsalicylic acid",
        drug_class=DrugClass.METABOLIC_INHIBITOR,
        pk_model=PharmacokineticModel(
            bioavailability=0.8,
            volume_of_distribution=11.0,  # L
            half_life=0.3,  # hours (very short)
            clearance=35.0,  # L/h
            protein_binding=0.8,
            tissue_penetration=0.5,
        ),
        mechanism_of_action="COX-2 inhibitor, anti-inflammatory, anti-platelet",
        target_proteins=["PTGS1", "PTGS2"],
        molecular_weight=180.2,
        ic50=50.0,  # μM (for cancer)
        ec50=100.0,
        emax=0.25,  # Modest anti-cancer effect
        hill_coefficient=1.0,
        cell_cycle_specific=False,
        standard_dose_mg=325.0,  # mg
        dosing_interval_hours=24.0,
        route="Oral",
        fda_approved=True,
        approval_year=1899,
        approved_indications=["Pain", "Inflammation", "Cardiovascular prophylaxis"],
    ),

    # ============================================================================
    # MORE CHEMOTHERAPY AGENTS
    # ============================================================================

    "docetaxel": Drug(
        name="Docetaxel",
        generic_name="docetaxel",
        drug_class=DrugClass.CHEMOTHERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.05,  # IV typically
            volume_of_distribution=113.0,  # L
            half_life=11.1,  # hours
            clearance=21.0,  # L/h
            protein_binding=0.95,
            tissue_penetration=0.45,
        ),
        mechanism_of_action="Microtubule stabilization, mitotic arrest",
        target_proteins=["TUBB"],
        molecular_weight=807.9,
        ic50=0.008,  # μM - very potent
        ec50=0.02,
        emax=0.95,
        hill_coefficient=3.0,
        cell_cycle_specific=True,
        target_phases=["M"],
        resistance_mutations=["TUBB3_overexpression", "MDR1_overexpression"],
        myelosuppression=0.8,
        neurotoxicity=0.5,
        standard_dose_mg=135,  # converted from 75.0 mg/m² (1.8 m² -> 135.0 mg)

        dosing_interval_hours=21 * 24,
        route="IV",
        fda_approved=True,
        approval_year=1996,
        approved_indications=["Breast cancer", "Prostate cancer", "NSCLC", "Gastric cancer"],
    ),

    "oxaliplatin": Drug(
        name="Oxaliplatin",
        generic_name="oxaliplatin",
        drug_class=DrugClass.CHEMOTHERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=1.0,  # IV only
            volume_of_distribution=440.0,  # L
            half_life=14.0,  # hours (terminal)
            clearance=10.0,  # L/h
            protein_binding=0.9,
            tissue_penetration=0.4,
        ),
        mechanism_of_action="DNA crosslinking, platinum-based",
        target_proteins=["DNA"],
        molecular_weight=397.3,
        ic50=2.0,  # μM
        ec50=3.5,
        emax=0.9,
        hill_coefficient=2.0,
        cell_cycle_specific=False,
        resistance_mutations=["ERCC1_overexpression"],
        myelosuppression=0.5,
        neurotoxicity=0.7,  # Peripheral neuropathy
        standard_dose_mg=153,  # converted from 85.0 mg/m² (1.8 m² -> 153.0 mg)

        dosing_interval_hours=14 * 24,
        route="IV",
        fda_approved=True,
        approval_year=2002,
        approved_indications=["Colorectal cancer"],
    ),

    "etoposide": Drug(
        name="Etoposide",
        generic_name="etoposide",
        drug_class=DrugClass.CHEMOTHERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.5,  # Oral variable
            volume_of_distribution=18.0,  # L
            half_life=7.0,  # hours
            clearance=2.5,  # L/h
            protein_binding=0.97,
            tissue_penetration=0.5,
        ),
        mechanism_of_action="Topoisomerase II inhibitor, DNA strand breaks",
        target_proteins=["TOP2A"],
        molecular_weight=588.6,
        ic50=0.8,  # μM
        ec50=1.5,
        emax=0.9,
        hill_coefficient=2.2,
        cell_cycle_specific=True,
        target_phases=["S", "G2"],
        resistance_mutations=["TOP2A_mutation", "MDR1_overexpression"],
        myelosuppression=0.7,
        standard_dose_mg=180,  # converted from 100.0 mg/m² (1.8 m² -> 180.0 mg)

        dosing_interval_hours=24.0,
        route="IV",
        fda_approved=True,
        approval_year=1983,
        approved_indications=["Testicular cancer", "SCLC", "Lymphoma"],
    ),

    "vincristine": Drug(
        name="Vincristine",
        generic_name="vincristine sulfate",
        drug_class=DrugClass.CHEMOTHERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.05,  # IV only (vesicant)
            volume_of_distribution=1640.0,  # L (huge Vd)
            half_life=85.0,  # hours
            clearance=19.0,  # L/h
            protein_binding=0.75,
            tissue_penetration=0.3,
        ),
        mechanism_of_action="Tubulin binding, microtubule depolymerization",
        target_proteins=["TUBB"],
        molecular_weight=923.0,
        ic50=0.001,  # μM (very potent)
        ec50=0.005,
        emax=0.85,
        hill_coefficient=3.5,
        cell_cycle_specific=True,
        target_phases=["M"],
        resistance_mutations=["MDR1_overexpression"],
        myelosuppression=0.3,  # Less than other chemos
        neurotoxicity=0.8,  # Dose-limiting
        standard_dose_mg=2,  # approx 2.0 mg cap (1.4 mg/m² × 1.8 m²)

        dosing_interval_hours=7 * 24,
        route="IV",
        fda_approved=True,
        approval_year=1963,
        approved_indications=["Leukemia", "Lymphoma", "Neuroblastoma"],
    ),

    "bleomycin": Drug(
        name="Bleomycin",
        generic_name="bleomycin sulfate",
        drug_class=DrugClass.CHEMOTHERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=1.0,  # IV/IM/SC
            volume_of_distribution=22.0,  # L
            half_life=2.0,  # hours
            clearance=11.0,  # L/h
            protein_binding=0.1,
            tissue_penetration=0.4,
        ),
        mechanism_of_action="DNA strand breaks via free radical generation",
        target_proteins=["DNA"],
        molecular_weight=1415.6,  # Glycopeptide
        ic50=0.5,  # μM
        ec50=1.0,
        emax=0.8,
        hill_coefficient=1.8,
        cell_cycle_specific=True,
        target_phases=["G2", "M"],
        resistance_mechanisms=["Bleomycin hydrolase"],
        myelosuppression=0.1,  # Minimal
        standard_dose_mg=18,  # converted from 10.0 units/m² (~mg) for 1.8 m²

        dosing_interval_hours=7 * 24,
        route="IV",
        fda_approved=True,
        approval_year=1973,
        approved_indications=["Hodgkin lymphoma", "Testicular cancer"],
    ),

    "capecitabine": Drug(
        name="Capecitabine",
        generic_name="capecitabine",
        drug_class=DrugClass.CHEMOTHERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.75,  # Good oral
            volume_of_distribution=110.0,  # L
            half_life=0.75,  # hours (short)
            clearance=145.0,  # L/h
            protein_binding=0.54,
            tissue_penetration=0.6,  # Prodrug activated in tumor
        ),
        mechanism_of_action="Oral 5-FU prodrug, thymidylate synthase inhibitor",
        target_proteins=["TYMS"],
        molecular_weight=359.4,
        ic50=2.0,  # μM
        ec50=4.0,
        emax=0.85,
        hill_coefficient=2.0,
        cell_cycle_specific=True,
        target_phases=["S"],
        resistance_mutations=["TYMS_overexpression", "DPD_deficiency"],
        myelosuppression=0.5,
        standard_dose_mg=2250,  # converted from 1250.0 mg/m² (1.8 m² -> 2250.0 mg)

        dosing_interval_hours=12.0,
        route="Oral",
        fda_approved=True,
        approval_year=1998,
        approved_indications=["Colorectal cancer", "Breast cancer"],
    ),

    "cytarabine": Drug(
        name="Cytarabine",
        generic_name="cytosine arabinoside",
        drug_class=DrugClass.CHEMOTHERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.2,  # Poor oral, IV typically
            volume_of_distribution=100.0,  # L
            half_life=1.0,  # hours (very short)
            clearance=100.0,  # L/h (rapid)
            protein_binding=0.13,
            tissue_penetration=0.5,
        ),
        mechanism_of_action="Nucleoside analog, inhibits DNA polymerase",
        target_proteins=["POLA", "DNA"],
        molecular_weight=243.2,
        ic50=0.1,  # μM
        ec50=0.3,
        emax=0.95,
        hill_coefficient=2.5,
        cell_cycle_specific=True,
        target_phases=["S"],
        resistance_mutations=["DCK_deficiency"],
        myelosuppression=0.9,  # Severe
        standard_dose_mg=180,  # converted from 100.0 mg/m² (1.8 m² -> 180.0 mg)

        dosing_interval_hours=12.0,
        route="IV",
        fda_approved=True,
        approval_year=1969,
        approved_indications=["AML", "ALL", "Lymphoma"],
    ),

    # ============================================================================
    # MORE TARGETED THERAPIES
    # ============================================================================

    "osimertinib": Drug(
        name="Osimertinib",
        generic_name="osimertinib",
        drug_class=DrugClass.TARGETED_THERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.7,  # Oral
            volume_of_distribution=918.0,  # L
            half_life=48.0,  # hours
            clearance=14.2,  # L/h
            protein_binding=0.95,
            tissue_penetration=0.6,  # Good CNS penetration
        ),
        mechanism_of_action="3rd-gen EGFR TKI, targets T790M resistance mutation",
        target_proteins=["EGFR"],
        molecular_weight=499.6,
        ic50=0.0015,  # μM (1.5 nM)
        ec50=0.005,
        emax=0.9,
        hill_coefficient=2.8,
        cell_cycle_specific=False,
        resistance_mutations=["EGFR_C797S"],
        standard_dose_mg=80.0,
        dosing_interval_hours=24.0,
        route="Oral",
        fda_approved=True,
        approval_year=2015,
        approved_indications=["NSCLC (EGFR T790M mutation)"],
    ),

    "dabrafenib": Drug(
        name="Dabrafenib",
        generic_name="dabrafenib",
        drug_class=DrugClass.TARGETED_THERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.95,
            volume_of_distribution=70.0,  # L
            half_life=8.0,  # hours
            clearance=8.75,  # L/h
            protein_binding=0.995,
            tissue_penetration=0.5,
        ),
        mechanism_of_action="BRAF V600E kinase inhibitor",
        target_proteins=["BRAF"],
        molecular_weight=519.6,
        ic50=0.002,  # μM
        ec50=0.008,
        emax=0.9,
        hill_coefficient=2.5,
        cell_cycle_specific=False,
        resistance_mutations=["NRAS_mutation", "MEK_mutation"],
        standard_dose_mg=150.0,
        dosing_interval_hours=12.0,
        route="Oral",
        fda_approved=True,
        approval_year=2013,
        approved_indications=["Melanoma (BRAF V600E/K)"],
    ),

    "crizotinib": Drug(
        name="Crizotinib",
        generic_name="crizotinib",
        drug_class=DrugClass.TARGETED_THERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.43,
            volume_of_distribution=1772.0,  # L (huge)
            half_life=42.0,  # hours
            clearance=60.0,  # L/h
            protein_binding=0.91,
            tissue_penetration=0.4,
        ),
        mechanism_of_action="ALK/ROS1/MET tyrosine kinase inhibitor",
        target_proteins=["ALK", "ROS1", "MET"],
        molecular_weight=450.3,
        ic50=0.00025,  # μM (0.25 nM) - very potent
        ec50=0.001,
        emax=0.95,
        hill_coefficient=3.0,
        cell_cycle_specific=False,
        resistance_mutations=["ALK_L1196M", "ALK_G1269A"],
        standard_dose_mg=250.0,
        dosing_interval_hours=12.0,
        route="Oral",
        fda_approved=True,
        approval_year=2011,
        approved_indications=["NSCLC (ALK/ROS1+)"],
    ),

    "lapatinib": Drug(
        name="Lapatinib",
        generic_name="lapatinib",
        drug_class=DrugClass.TARGETED_THERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.25,  # Variable with food
            volume_of_distribution=2400.0,  # L
            half_life=24.0,  # hours
            clearance=100.0,  # L/h
            protein_binding=0.99,
            tissue_penetration=0.4,
        ),
        mechanism_of_action="Dual EGFR/HER2 tyrosine kinase inhibitor",
        target_proteins=["EGFR", "ERBB2"],
        molecular_weight=581.1,
        ic50=0.01,  # μM
        ec50=0.03,
        emax=0.8,
        hill_coefficient=2.2,
        cell_cycle_specific=False,
        resistance_mechanisms=["PI3K_mutation"],
        standard_dose_mg=1250.0,
        dosing_interval_hours=24.0,
        route="Oral",
        fda_approved=True,
        approval_year=2007,
        approved_indications=["Breast cancer (HER2+)"],
    ),

    "sunitinib": Drug(
        name="Sunitinib",
        generic_name="sunitinib malate",
        drug_class=DrugClass.TARGETED_THERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.85,
            volume_of_distribution=2230.0,  # L
            half_life=40.0,  # hours (with metabolite 80-110h)
            clearance=34.0,  # L/h
            protein_binding=0.95,
            tissue_penetration=0.5,
        ),
        mechanism_of_action="Multi-kinase inhibitor (VEGFR, PDGFR, KIT)",
        target_proteins=["KDR", "PDGFRA", "KIT"],
        molecular_weight=398.5,
        ic50=0.01,  # μM
        ec50=0.03,
        emax=0.75,
        hill_coefficient=2.0,
        cell_cycle_specific=False,
        standard_dose_mg=50.0,
        dosing_interval_hours=24.0,
        route="Oral",
        fda_approved=True,
        approval_year=2006,
        approved_indications=["RCC", "GIST"],
    ),

    "sorafenib": Drug(
        name="Sorafenib",
        generic_name="sorafenib tosylate",
        drug_class=DrugClass.TARGETED_THERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.4,
            volume_of_distribution=212.0,  # L
            half_life=27.0,  # hours
            clearance=8.0,  # L/h
            protein_binding=0.995,
            tissue_penetration=0.45,
        ),
        mechanism_of_action="Multi-kinase inhibitor (RAF, VEGFR, PDGFR)",
        target_proteins=["BRAF", "KDR", "PDGFR"],
        molecular_weight=464.8,
        ic50=0.006,  # μM
        ec50=0.02,
        emax=0.7,
        hill_coefficient=1.8,
        cell_cycle_specific=False,
        standard_dose_mg=400.0,
        dosing_interval_hours=12.0,
        route="Oral",
        fda_approved=True,
        approval_year=2005,
        approved_indications=["RCC", "HCC"],
    ),

    # ============================================================================
    # MORE IMMUNOTHERAPY
    # ============================================================================

    "atezolizumab": Drug(
        name="Atezolizumab",
        generic_name="atezolizumab",
        drug_class=DrugClass.IMMUNOTHERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=1.0,  # IV antibody
            volume_of_distribution=6.9,  # L
            half_life=27.0 * 24,  # 27 days
            clearance=0.25,  # L/h
            protein_binding=0.0,
            tissue_penetration=0.2,
        ),
        mechanism_of_action="PD-L1 checkpoint inhibitor",
        target_proteins=["CD274"],  # PD-L1
        molecular_weight=145000.0,
        ic50=0.0001,
        ec50=0.0003,
        emax=0.55,
        hill_coefficient=1.5,
        cell_cycle_specific=False,
        standard_dose_mg=1200.0,  # mg flat dose
        dosing_interval_hours=21 * 24,
        route="IV",
        fda_approved=True,
        approval_year=2016,
        approved_indications=["NSCLC", "Urothelial carcinoma"],
    ),

    "durvalumab": Drug(
        name="Durvalumab",
        generic_name="durvalumab",
        drug_class=DrugClass.IMMUNOTHERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=1.0,
            volume_of_distribution=5.6,  # L
            half_life=18.0 * 24,  # 18 days
            clearance=0.31,  # L/h
            protein_binding=0.0,
            tissue_penetration=0.2,
        ),
        mechanism_of_action="PD-L1 checkpoint inhibitor",
        target_proteins=["CD274"],
        molecular_weight=146000.0,
        ic50=0.0001,
        ec50=0.0003,
        emax=0.5,
        hill_coefficient=1.5,
        cell_cycle_specific=False,
        standard_dose_mg=700,  # converted from 10.0 mg/kg (70 kg -> 700 mg)

        dosing_interval_hours=14 * 24,
        route="IV",
        fda_approved=True,
        approval_year=2017,
        approved_indications=["NSCLC", "SCLC"],
    ),

    "ipilimumab": Drug(
        name="Ipilimumab",
        generic_name="ipilimumab",
        drug_class=DrugClass.IMMUNOTHERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=1.0,
            volume_of_distribution=7.2,  # L
            half_life=15.4 * 24,  # 15.4 days
            clearance=0.47,  # L/h
            protein_binding=0.0,
            tissue_penetration=0.15,
        ),
        mechanism_of_action="CTLA-4 checkpoint inhibitor",
        target_proteins=["CTLA4"],
        molecular_weight=148000.0,
        ic50=0.0002,
        ec50=0.0005,
        emax=0.4,  # Lower than PD-1 inhibitors alone
        hill_coefficient=1.3,
        cell_cycle_specific=False,
        standard_dose_mg=210,  # converted from 3.0 mg/kg (70 kg -> 210 mg)

        dosing_interval_hours=21 * 24,
        route="IV",
        fda_approved=True,
        approval_year=2011,
        approved_indications=["Melanoma", "RCC"],
    ),

    # ============================================================================
    # HORMONE THERAPIES
    # ============================================================================

    "tamoxifen": Drug(
        name="Tamoxifen",
        generic_name="tamoxifen citrate",
        drug_class=DrugClass.HORMONE_THERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.9,
            volume_of_distribution=1400.0,  # L (huge)
            half_life=5.0 * 24,  # 5-7 days
            clearance=12.0,  # L/h
            protein_binding=0.99,
            tissue_penetration=0.5,
        ),
        mechanism_of_action="Selective estrogen receptor modulator (SERM)",
        target_proteins=["ESR1", "ESR2"],
        molecular_weight=563.6,
        ic50=10.0,  # μM (anti-proliferative, not cytotoxic)
        ec50=20.0,
        emax=0.6,  # Growth inhibition, not kill
        hill_coefficient=1.5,
        cell_cycle_specific=False,
        standard_dose_mg=20.0,
        dosing_interval_hours=24.0,
        route="Oral",
        fda_approved=True,
        approval_year=1977,
        approved_indications=["Breast cancer (ER+)"],
    ),

    "letrozole": Drug(
        name="Letrozole",
        generic_name="letrozole",
        drug_class=DrugClass.HORMONE_THERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.99,
            volume_of_distribution=87.0,  # L
            half_life=42.0,  # hours
            clearance=2.1,  # L/h
            protein_binding=0.6,
            tissue_penetration=0.5,
        ),
        mechanism_of_action="Aromatase inhibitor, blocks estrogen synthesis",
        target_proteins=["CYP19A1"],  # Aromatase
        molecular_weight=285.3,
        ic50=15.0,  # μM
        ec50=30.0,
        emax=0.7,
        hill_coefficient=1.8,
        cell_cycle_specific=False,
        standard_dose_mg=2.5,
        dosing_interval_hours=24.0,
        route="Oral",
        fda_approved=True,
        approval_year=1997,
        approved_indications=["Breast cancer (ER+ postmenopausal)"],
    ),

    "enzalutamide": Drug(
        name="Enzalutamide",
        generic_name="enzalutamide",
        drug_class=DrugClass.HORMONE_THERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.84,
            volume_of_distribution=110.0,  # L
            half_life=5.8 * 24,  # 5.8 days
            clearance=0.56,  # L/h
            protein_binding=0.97,
            tissue_penetration=0.5,
        ),
        mechanism_of_action="Androgen receptor antagonist",
        target_proteins=["AR"],
        molecular_weight=464.4,
        ic50=8.0,  # μM
        ec50=15.0,
        emax=0.75,
        hill_coefficient=2.0,
        cell_cycle_specific=False,
        standard_dose_mg=160.0,
        dosing_interval_hours=24.0,
        route="Oral",
        fda_approved=True,
        approval_year=2012,
        approved_indications=["Prostate cancer (castration-resistant)"],
    ),

    "anastrozole": Drug(
        name="Anastrozole",
        generic_name="anastrozole",
        drug_class=DrugClass.HORMONE_THERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.83,
            volume_of_distribution=50.0,  # L
            half_life=46.8,  # hours
            clearance=1.1,  # L/h
            protein_binding=0.4,
            tissue_penetration=0.5,
        ),
        mechanism_of_action="Aromatase inhibitor",
        target_proteins=["CYP19A1"],
        molecular_weight=293.4,
        ic50=12.0,  # μM
        ec50=25.0,
        emax=0.7,
        hill_coefficient=1.7,
        cell_cycle_specific=False,
        standard_dose_mg=1.0,
        dosing_interval_hours=24.0,
        route="Oral",
        fda_approved=True,
        approval_year=1995,
        approved_indications=["Breast cancer (ER+)"],
    ),

    # ============================================================================
    # MORE NATURAL COMPOUNDS & SUPPLEMENTS
    # ============================================================================

    "artemisinin": Drug(
        name="Artemisinin",
        generic_name="artemisinin",
        drug_class=DrugClass.METABOLIC_INHIBITOR,
        pk_model=PharmacokineticModel(
            bioavailability=0.3,
            volume_of_distribution=200.0,  # L
            half_life=2.0,  # hours (very short)
            clearance=100.0,  # L/h
            protein_binding=0.75,
            tissue_penetration=0.6,
        ),
        mechanism_of_action="Iron-catalyzed ROS generation, anti-angiogenic",
        target_proteins=["Ferritin", "VEGF"],
        molecular_weight=282.3,
        ic50=5.0,  # μM
        ec50=10.0,
        emax=0.5,
        hill_coefficient=1.8,
        cell_cycle_specific=False,
        standard_dose_mg=200.0,
        dosing_interval_hours=12.0,
        route="Oral",
        fda_approved=False,
        approved_indications=[],  # Antimalarial, not FDA for cancer
    ),

    "berberine": Drug(
        name="Berberine",
        generic_name="berberine",
        drug_class=DrugClass.METABOLIC_INHIBITOR,
        pk_model=PharmacokineticModel(
            bioavailability=0.005,  # Very poor
            volume_of_distribution=7000.0,  # L (huge tissue distribution)
            half_life=5.0,  # hours
            clearance=1400.0,  # L/h
            protein_binding=0.7,
            tissue_penetration=0.7,  # Despite low bioavailability
        ),
        mechanism_of_action="AMPK activator, mitochondrial dysfunction",
        target_proteins=["PRKAA1"],  # AMPK
        molecular_weight=336.4,
        ic50=20.0,  # μM
        ec50=40.0,
        emax=0.45,
        hill_coefficient=1.5,
        cell_cycle_specific=False,
        standard_dose_mg=500.0,
        dosing_interval_hours=8.0,
        route="Oral",
        fda_approved=False,
        approved_indications=[],
    ),

    "cbd": Drug(
        name="CBD",
        generic_name="cannabidiol",
        drug_class=DrugClass.METABOLIC_INHIBITOR,
        pk_model=PharmacokineticModel(
            bioavailability=0.06,  # Very low oral
            volume_of_distribution=32000.0,  # L (massive lipophilic distribution)
            half_life=18.0,  # hours
            clearance=1700.0,  # L/h
            protein_binding=0.94,
            tissue_penetration=0.8,
        ),
        mechanism_of_action="CB1/CB2 modulation, anti-inflammatory, pro-apoptotic",
        target_proteins=["CNR1", "CNR2"],
        molecular_weight=314.5,
        ic50=15.0,  # μM
        ec50=30.0,
        emax=0.35,
        hill_coefficient=1.3,
        cell_cycle_specific=False,
        standard_dose_mg=25.0,  # mg (varies widely)
        dosing_interval_hours=12.0,
        route="Oral",
        fda_approved=True,
        approval_year=2018,
        approved_indications=["Epilepsy (Epidiolex)"],
    ),

    "melatonin": Drug(
        name="Melatonin",
        generic_name="melatonin",
        drug_class=DrugClass.METABOLIC_INHIBITOR,
        pk_model=PharmacokineticModel(
            bioavailability=0.15,  # Low oral
            volume_of_distribution=1000.0,  # L
            half_life=0.8,  # hours (very short)
            clearance=1250.0,  # L/h (rapid)
            protein_binding=0.6,
            tissue_penetration=0.8,
        ),
        mechanism_of_action="Antioxidant, circadian regulation, immune modulation",
        target_proteins=["MTNR1A", "MTNR1B"],
        molecular_weight=232.3,
        ic50=50.0,  # μM (high doses needed)
        ec50=100.0,
        emax=0.25,
        hill_coefficient=1.0,
        cell_cycle_specific=False,
        standard_dose_mg=20.0,  # mg (high dose for cancer)
        dosing_interval_hours=24.0,
        route="Oral",
        fda_approved=False,  # OTC supplement
        approved_indications=[],
    ),

    "omega3_dha": Drug(
        name="Omega-3 DHA",
        generic_name="docosahexaenoic acid",
        drug_class=DrugClass.METABOLIC_INHIBITOR,
        pk_model=PharmacokineticModel(
            bioavailability=0.9,
            volume_of_distribution=10000.0,  # L (incorporates into membranes)
            half_life=20.0 * 24,  # ~20 days
            clearance=20.0,  # L/h
            protein_binding=0.99,
            tissue_penetration=0.9,
        ),
        mechanism_of_action="Anti-inflammatory, membrane disruption, PPAR agonist",
        target_proteins=["PPARA", "PPARD", "PPARG"],
        molecular_weight=328.5,
        ic50=100.0,  # μM
        ec50=200.0,
        emax=0.2,
        hill_coefficient=1.0,
        cell_cycle_specific=False,
        standard_dose_mg=2000.0,  # mg
        dosing_interval_hours=24.0,
        route="Oral",
        fda_approved=True,
        approval_year=2004,
        approved_indications=["Hypertriglyceridemia"],
    ),

    "sulforaphane": Drug(
        name="Sulforaphane",
        generic_name="sulforaphane",
        drug_class=DrugClass.METABOLIC_INHIBITOR,
        pk_model=PharmacokineticModel(
            bioavailability=0.8,  # From broccoli sprouts
            volume_of_distribution=700.0,  # L
            half_life=2.0,  # hours
            clearance=350.0,  # L/h
            protein_binding=0.5,
            tissue_penetration=0.7,
        ),
        mechanism_of_action="NRF2 activator, phase II detox enzymes, HDAC inhibitor",
        target_proteins=["NFE2L2", "HDAC"],
        molecular_weight=177.3,
        ic50=10.0,  # μM
        ec50=20.0,
        emax=0.4,
        hill_coefficient=1.5,
        cell_cycle_specific=False,
        standard_dose_mg=30.0,  # mg (from ~30g broccoli sprouts)
        dosing_interval_hours=24.0,
        route="Oral",
        fda_approved=False,
        approved_indications=[],
    ),

    # ============================================================================
    # MISSING DRUGS FROM CLINICAL TRIALS - TRIPLE-CHECKED
    # ============================================================================

    "cyclophosphamide": Drug(
        name="Cyclophosphamide",
        generic_name="cyclophosphamide",
        drug_class=DrugClass.CHEMOTHERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.75,  # Oral prodrug
            volume_of_distribution=40.0,  # L
            half_life=7.5,  # hours (parent drug)
            clearance=5.3,  # L/h
            protein_binding=0.13,  # Low protein binding
            tissue_penetration=0.8,  # Excellent penetration
            prodrug=True,  # Activated by liver enzymes
        ),
        mechanism_of_action="DNA alkylating agent (prodrug), crosslinks DNA",
        target_proteins=["DNA"],
        molecular_weight=261.1,  # g/mol - cyclophosphamide monohydrate
        ic50=50.0,  # μM (high, needs activation)
        ec50=100.0,
        emax=0.9,
        hill_coefficient=2.0,
        cell_cycle_specific=False,  # Alkylating agents work on all phases
        resistance_mutations=["ALDH1_overexpression", "GSTP1_overexpression"],
        myelosuppression=0.85,  # Severe myelosuppression
        cardiotoxicity=0.4,
        hepatotoxicity=0.3,
        standard_dose_mg=1080,  # converted from 600.0 mg/m² (1.8 m² -> 1080.0 mg)

        dosing_interval_hours=21 * 24,  # Every 3 weeks
        route="IV",
        fda_approved=True,
        approval_year=1959,
        approved_indications=["Breast cancer", "Lymphoma", "Multiple myeloma"],
    ),

    "irinotecan": Drug(
        name="Irinotecan",
        generic_name="irinotecan hydrochloride",
        drug_class=DrugClass.CHEMOTHERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=1.0,  # IV only
            volume_of_distribution=110.0,  # L
            half_life=10.0,  # hours
            clearance=13.3,  # L/h
            protein_binding=0.95,
            tissue_penetration=0.45,
            prodrug=True,  # Converted to SN-38 (active metabolite)
            active_metabolites=["SN-38"],
        ),
        mechanism_of_action="Topoisomerase I inhibitor (prodrug→SN-38)",
        target_proteins=["TOP1"],
        molecular_weight=677.2,  # g/mol - irinotecan HCl
        ic50=0.1,  # μM (SN-38 is 100-1000x more potent)
        ec50=0.5,
        emax=0.85,
        hill_coefficient=2.5,
        cell_cycle_specific=True,
        target_phases=["S"],  # S-phase specific
        resistance_mutations=["TOP1_mutation", "ABCG2_overexpression"],
        myelosuppression=0.75,
        standard_dose_mg=324,  # converted from 180.0 mg/m² (1.8 m² -> 324.0 mg)

        dosing_interval_hours=14 * 24,  # Every 2 weeks
        route="IV",
        fda_approved=True,
        approval_year=1996,
        approved_indications=["Colorectal cancer"],
    ),

    "leucovorin": Drug(
        name="Leucovorin",
        generic_name="folinic acid",
        drug_class=DrugClass.CHEMOTHERAPY,  # Technically a rescue agent/modifier
        pk_model=PharmacokineticModel(
            bioavailability=0.97,  # Excellent oral absorption
            volume_of_distribution=12.0,  # L
            half_life=6.2,  # hours
            clearance=1.9,  # L/h
            protein_binding=0.15,  # Low protein binding
            tissue_penetration=0.6,
        ),
        mechanism_of_action="Reduced folate, potentiates 5-FU by stabilizing TS-FdUMP complex",
        target_proteins=["TYMS"],  # Enhances 5-FU binding to thymidylate synthase
        molecular_weight=511.5,  # g/mol - calcium salt
        ic50=1000.0,  # μM (not cytotoxic alone, potentiates 5-FU)
        ec50=2000.0,
        emax=0.1,  # Minimal direct effect
        hill_coefficient=1.0,
        cell_cycle_specific=False,
        myelosuppression=0.0,  # Actually rescues from myelosuppression
        standard_dose_mg=360,  # converted from 200.0 mg/m² (1.8 m² -> 360.0 mg)

        dosing_interval_hours=48.0,  # With 5-FU
        route="IV",
        fda_approved=True,
        approval_year=1952,
        approved_indications=["Colorectal cancer (with 5-FU)", "Rescue from methotrexate"],
    ),

    "oxaliplatin": Drug(
        name="Oxaliplatin",
        generic_name="oxaliplatin",
        drug_class=DrugClass.CHEMOTHERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=1.0,  # IV only
            volume_of_distribution=440.0,  # L
            half_life=14.0,  # hours (terminal: 273h due to protein binding)
            clearance=10.0,  # L/h
            protein_binding=0.90,  # Extensive protein binding
            tissue_penetration=0.4,  # Moderate tumor penetration
        ),
        mechanism_of_action="Platinum-based DNA crosslinking agent",
        target_proteins=["DNA"],
        molecular_weight=397.3,  # g/mol - oxaliplatin
        ic50=1.0,  # μM
        ec50=2.5,
        emax=0.92,
        hill_coefficient=2.2,
        cell_cycle_specific=False,
        resistance_mutations=["ERCC1_overexpression", "XRCC1_polymorphism"],
        myelosuppression=0.5,  # Lower than cisplatin
        neurotoxicity=0.8,  # Severe peripheral neuropathy (dose-limiting)
        standard_dose_mg=153,  # converted from 85.0 mg/m² (1.8 m² -> 153.0 mg)

        dosing_interval_hours=14 * 24,  # Every 2 weeks
        route="IV",
        fda_approved=True,
        approval_year=2002,
        approved_indications=["Colorectal cancer"],
    ),

    "pemetrexed": Drug(
        name="Pemetrexed",
        generic_name="pemetrexed disodium",
        drug_class=DrugClass.CHEMOTHERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=1.0,  # IV only
            volume_of_distribution=16.0,  # L
            half_life=3.5,  # hours
            clearance=5.6,  # L/h
            protein_binding=0.81,
            tissue_penetration=0.5,
        ),
        mechanism_of_action="Multitargeted antifolate, inhibits TS, DHFR, GARFT",
        target_proteins=["TYMS", "DHFR", "GARFT"],
        molecular_weight=597.5,  # g/mol - disodium salt
        ic50=0.05,  # μM
        ec50=0.2,
        emax=0.88,
        hill_coefficient=2.3,
        cell_cycle_specific=True,
        target_phases=["S"],
        resistance_mutations=["TYMS_overexpression"],
        myelosuppression=0.7,
        standard_dose_mg=900,  # converted from 500.0 mg/m² (1.8 m² -> 900.0 mg)

        dosing_interval_hours=21 * 24,  # Every 3 weeks
        route="IV",
        fda_approved=True,
        approval_year=2004,
        approved_indications=["NSCLC (non-squamous)", "Mesothelioma"],
    ),

    "docetaxel": Drug(
        name="Docetaxel",
        generic_name="docetaxel",
        drug_class=DrugClass.CHEMOTHERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.08,  # Poor oral, usually IV
            volume_of_distribution=113.0,  # L
            half_life=11.1,  # hours
            clearance=21.0,  # L/h
            protein_binding=0.94,
            tissue_penetration=0.5,
        ),
        mechanism_of_action="Microtubule stabilization, mitotic arrest",
        target_proteins=["TUBB"],  # Beta-tubulin
        molecular_weight=807.9,  # g/mol - docetaxel
        ic50=0.008,  # μM - very potent
        ec50=0.03,
        emax=0.95,
        hill_coefficient=3.2,  # Very steep M-phase
        cell_cycle_specific=True,
        target_phases=["M"],
        resistance_mutations=["TUBB3_overexpression", "MDR1_overexpression"],
        myelosuppression=0.85,
        neurotoxicity=0.5,
        standard_dose_mg=135,  # converted from 75.0 mg/m² (1.8 m² -> 135.0 mg)

        dosing_interval_hours=21 * 24,  # Every 3 weeks
        route="IV",
        fda_approved=True,
        approval_year=1996,
        approved_indications=["Breast cancer", "NSCLC", "Prostate cancer"],
    ),

    "etoposide": Drug(
        name="Etoposide",
        generic_name="etoposide",
        drug_class=DrugClass.CHEMOTHERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.5,  # Oral available
            volume_of_distribution=18.0,  # L
            half_life=7.0,  # hours
            clearance=2.5,  # L/h
            protein_binding=0.97,
            tissue_penetration=0.3,  # Poor CNS penetration
        ),
        mechanism_of_action="Topoisomerase II inhibitor, DNA strand breaks",
        target_proteins=["TOP2A", "TOP2B"],
        molecular_weight=588.6,  # g/mol - etoposide
        ic50=0.5,  # μM
        ec50=1.5,
        emax=0.85,
        hill_coefficient=2.0,
        cell_cycle_specific=True,
        target_phases=["S", "G2"],
        resistance_mutations=["MDR1_overexpression", "TOP2A_mutation"],
        myelosuppression=0.8,
        standard_dose_mg=180,  # converted from 100.0 mg/m² (1.8 m² -> 180.0 mg)

        dosing_interval_hours=24.0,  # Daily x3-5 days
        route="IV",
        fda_approved=True,
        approval_year=1983,
        approved_indications=["Testicular cancer", "Lung cancer", "Lymphoma"],
    ),

    "cabazitaxel": Drug(
        name="Cabazitaxel",
        generic_name="cabazitaxel",
        drug_class=DrugClass.CHEMOTHERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=1.0,  # IV only
            volume_of_distribution=4870.0,  # L - huge Vd
            half_life=95.0,  # hours (terminal phase)
            clearance=48.5,  # L/h
            protein_binding=0.89,
            tissue_penetration=0.6,  # Better BBB penetration than paclitaxel
        ),
        mechanism_of_action="Microtubule stabilization, overcomes MDR1",
        target_proteins=["TUBB"],
        molecular_weight=835.9,  # g/mol - cabazitaxel
        ic50=0.002,  # μM - very potent
        ec50=0.01,
        emax=0.95,
        hill_coefficient=3.0,
        cell_cycle_specific=True,
        target_phases=["M"],
        resistance_mechanisms=["Not a P-gp substrate"],  # Overcomes MDR1
        myelosuppression=0.9,  # Severe neutropenia
        neurotoxicity=0.4,
        standard_dose_mg=45,  # converted from 25.0 mg/m² (1.8 m² -> 45.0 mg)

        dosing_interval_hours=21 * 24,  # Every 3 weeks
        route="IV",
        fda_approved=True,
        approval_year=2010,
        approved_indications=["Prostate cancer (castration-resistant, post-docetaxel)"],
    ),

    "carmustine": Drug(
        name="Carmustine",
        generic_name="carmustine",
        drug_class=DrugClass.CHEMOTHERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=1.0,  # IV or wafer implant
            volume_of_distribution=27.0,  # L
            half_life=0.28,  # hours (very short, 17 min)
            clearance=97.0,  # L/h - rapid
            protein_binding=0.75,
            tissue_penetration=0.9,  # Excellent CNS penetration (lipophilic)
        ),
        mechanism_of_action="Nitrosourea alkylating agent, DNA/RNA crosslinking",
        target_proteins=["DNA", "RNA"],
        molecular_weight=214.1,  # g/mol - carmustine
        ic50=10.0,  # μM
        ec50=30.0,
        emax=0.9,
        hill_coefficient=1.8,
        cell_cycle_specific=False,
        resistance_mutations=["MGMT_overexpression", "GSTP1_overexpression"],
        myelosuppression=0.9,  # Severe, delayed (4-6 weeks)
        hepatotoxicity=0.5,
        standard_dose_mg=360,  # converted from 200.0 mg/m² (1.8 m² -> 360.0 mg)

        dosing_interval_hours=42 * 24,  # Every 6 weeks (due to delayed myelosuppression)
        route="IV",
        fda_approved=True,
        approval_year=1977,
        approved_indications=["Glioblastoma", "Lymphoma"],
    ),

    "lomustine": Drug(
        name="Lomustine",
        generic_name="lomustine",
        drug_class=DrugClass.CHEMOTHERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.45,  # Oral capsule
            volume_of_distribution=1000.0,  # L - wide distribution
            half_life=16.0,  # hours (metabolites: 48-72h)
            clearance=62.5,  # L/h
            protein_binding=0.5,
            tissue_penetration=0.85,  # Excellent CNS penetration
            active_metabolites=["Cyclohexyl isocyanate"],
        ),
        mechanism_of_action="Nitrosourea alkylating agent, DNA crosslinking",
        target_proteins=["DNA"],
        molecular_weight=233.7,  # g/mol - lomustine
        ic50=15.0,  # μM
        ec50=40.0,
        emax=0.88,
        hill_coefficient=1.7,
        cell_cycle_specific=False,
        resistance_mutations=["MGMT_overexpression"],
        myelosuppression=0.95,  # Very severe, delayed
        hepatotoxicity=0.6,
        standard_dose_mg=234,  # converted from 130.0 mg/m² (1.8 m² -> 234.0 mg)

        dosing_interval_hours=42 * 24,  # Every 6 weeks
        route="Oral",
        fda_approved=True,
        approval_year=1976,
        approved_indications=["Glioblastoma", "Hodgkin lymphoma"],
    ),

    "nab-paclitaxel": Drug(
        name="Nab-Paclitaxel",
        generic_name="paclitaxel albumin-bound",
        drug_class=DrugClass.CHEMOTHERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=1.0,  # IV only
            volume_of_distribution=632.0,  # L - larger than paclitaxel
            half_life=27.0,  # hours
            clearance=15.0,  # L/h
            protein_binding=0.89,  # Bound to albumin nanoparticles
            tissue_penetration=0.7,  # Better than paclitaxel via SPARC/gp60 pathway
        ),
        mechanism_of_action="Albumin-bound paclitaxel, microtubule stabilization, better tumor penetration",
        target_proteins=["TUBB", "SPARC"],  # SPARC-mediated transcytosis
        molecular_weight=853.9,  # g/mol - paclitaxel (same as paclitaxel)
        ic50=0.008,  # μM - similar to paclitaxel
        ec50=0.04,
        emax=0.95,
        hill_coefficient=3.1,
        cell_cycle_specific=True,
        target_phases=["M"],
        resistance_mutations=["TUBB3_overexpression"],  # Less MDR1 resistance
        myelosuppression=0.8,
        neurotoxicity=0.65,
        standard_dose_mg=468,  # converted from 260.0 mg/m² (1.8 m² -> 468.0 mg)

        dosing_interval_hours=21 * 24,  # Every 3 weeks
        route="IV",
        fda_approved=True,
        approval_year=2005,
        approved_indications=["Breast cancer", "NSCLC", "Pancreatic cancer"],
    ),

    "olaparib": Drug(
        name="Olaparib",
        generic_name="olaparib",
        drug_class=DrugClass.TARGETED_THERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.45,  # Oral tablet
            volume_of_distribution=158.0,  # L
            half_life=14.9,  # hours
            clearance=7.6,  # L/h
            protein_binding=0.82,
            tissue_penetration=0.5,
        ),
        mechanism_of_action="PARP1/2 inhibitor, synthetic lethality in BRCA-mutated tumors",
        target_proteins=["PARP1", "PARP2"],
        molecular_weight=434.5,  # g/mol - olaparib
        ic50=0.005,  # μM (5 nM) - very potent PARP inhibitor
        ec50=0.02,
        emax=0.85,  # High efficacy in BRCA1/2-mutated cancers
        hill_coefficient=2.5,
        cell_cycle_specific=False,  # Blocks DNA repair, not cell cycle specific
        resistance_mutations=["BRCA1_reversion", "BRCA2_reversion", "TP53BP1_loss"],
        myelosuppression=0.6,
        standard_dose_mg=300.0,  # mg BID
        dosing_interval_hours=12.0,
        route="Oral",
        fda_approved=True,
        approval_year=2014,
        approved_indications=["Ovarian cancer (BRCA-mutated)", "Breast cancer (BRCA-mutated)"],
    ),

    "niraparib": Drug(
        name="Niraparib",
        generic_name="niraparib tosylate",
        drug_class=DrugClass.TARGETED_THERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.73,  # Oral capsule
            volume_of_distribution=1220.0,  # L - large
            half_life=36.0,  # hours
            clearance=16.2,  # L/h
            protein_binding=0.83,
            tissue_penetration=0.55,
        ),
        mechanism_of_action="PARP1/2 inhibitor, synthetic lethality",
        target_proteins=["PARP1", "PARP2"],
        molecular_weight=320.3,  # g/mol - niraparib free base
        ic50=0.004,  # μM (3.8 nM for PARP1)
        ec50=0.015,
        emax=0.82,
        hill_coefficient=2.4,
        cell_cycle_specific=False,
        resistance_mutations=["BRCA_reversion", "P-gp_overexpression"],
        myelosuppression=0.75,  # Thrombocytopenia is dose-limiting
        standard_dose_mg=300.0,  # mg QD (200mg for lower body weight)
        dosing_interval_hours=24.0,
        route="Oral",
        fda_approved=True,
        approval_year=2017,
        approved_indications=["Ovarian cancer"],
    ),

    "pertuzumab": Drug(
        name="Pertuzumab",
        generic_name="pertuzumab",
        drug_class=DrugClass.TARGETED_THERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=1.0,  # IV monoclonal antibody
            volume_of_distribution=6.0,  # L (central compartment)
            half_life=18.0 * 24,  # 18 days
            clearance=0.24,  # L/h
            protein_binding=0.0,  # Antibodies don't bind plasma proteins classically
            tissue_penetration=0.15,  # Limited tumor penetration (large molecule)
        ),
        mechanism_of_action="HER2 dimerization inhibitor (monoclonal antibody)",
        target_proteins=["ERBB2"],  # HER2
        molecular_weight=148000.0,  # g/mol - typical IgG1 antibody
        ic50=0.0001,  # μM - very low for antibody
        ec50=0.0003,
        emax=0.75,  # High efficacy with trastuzumab
        hill_coefficient=1.5,
        cell_cycle_specific=False,
        resistance_mutations=["HER2_loss", "PI3K_mutation"],
        cardiotoxicity=0.3,  # Lower than trastuzumab
        standard_dose_mg=420.0,  # mg loading dose (840mg), then 420mg
        dosing_interval_hours=21 * 24,  # Every 3 weeks
        route="IV",
        fda_approved=True,
        approval_year=2012,
        approved_indications=["Breast cancer (HER2+)"],
    ),

    "trametinib": Drug(
        name="Trametinib",
        generic_name="trametinib",
        drug_class=DrugClass.TARGETED_THERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.72,  # Oral tablet
            volume_of_distribution=1103.0,  # L
            half_life=127.0,  # hours (~5.3 days)
            clearance=4.9,  # L/h
            protein_binding=0.972,  # Very high
            tissue_penetration=0.5,
        ),
        mechanism_of_action="MEK1/2 inhibitor, MAPK pathway",
        target_proteins=["MAP2K1", "MAP2K2"],  # MEK1, MEK2
        molecular_weight=615.4,  # g/mol - trametinib
        ic50=0.0009,  # μM (0.92 nM for MEK1) - very potent
        ec50=0.004,
        emax=0.88,  # High efficacy with BRAF inhibitors
        hill_coefficient=2.8,
        cell_cycle_specific=False,
        resistance_mutations=["MEK_mutation", "KRAS_amplification"],
        standard_dose_mg=2.0,  # mg QD
        dosing_interval_hours=24.0,
        route="Oral",
        fda_approved=True,
        approval_year=2013,
        approved_indications=["Melanoma (BRAF V600E/K, with dabrafenib)"],
    ),

    "cobimetinib": Drug(
        name="Cobimetinib",
        generic_name="cobimetinib",
        drug_class=DrugClass.TARGETED_THERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.46,  # Oral tablet, increased with food
            volume_of_distribution=806.0,  # L
            half_life=44.0,  # hours
            clearance=13.0,  # L/h
            protein_binding=0.95,
            tissue_penetration=0.5,
        ),
        mechanism_of_action="MEK1/2 inhibitor, MAPK pathway",
        target_proteins=["MAP2K1", "MAP2K2"],
        molecular_weight=531.4,  # g/mol - cobimetinib
        ic50=0.0014,  # μM (1.4 nM for MEK1)
        ec50=0.006,
        emax=0.86,
        hill_coefficient=2.6,
        cell_cycle_specific=False,
        resistance_mutations=["NRAS_mutation", "MEK1_mutation"],
        standard_dose_mg=60.0,  # mg QD (21 days on, 7 days off)
        dosing_interval_hours=24.0,
        route="Oral",
        fda_approved=True,
        approval_year=2015,
        approved_indications=["Melanoma (BRAF V600E/K, with vemurafenib)"],
    ),

    "dabrafenib": Drug(
        name="Dabrafenib",
        generic_name="dabrafenib mesylate",
        drug_class=DrugClass.TARGETED_THERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.95,  # Oral capsule
            volume_of_distribution=70.0,  # L
            half_life=8.0,  # hours (parent), 21-22h (metabolites)
            clearance=8.8,  # L/h
            protein_binding=0.995,
            tissue_penetration=0.55,
            active_metabolites=["Hydroxy-dabrafenib", "Desmethyl-dabrafenib"],
        ),
        mechanism_of_action="BRAF V600E kinase inhibitor",
        target_proteins=["BRAF"],
        molecular_weight=519.6,  # g/mol - dabrafenib mesylate
        ic50=0.0008,  # μM (0.8 nM for BRAF V600E) - very potent
        ec50=0.004,
        emax=0.92,
        hill_coefficient=2.9,
        cell_cycle_specific=False,
        resistance_mutations=["NRAS_mutation", "MEK_mutation", "COT_overexpression"],
        standard_dose_mg=150.0,  # mg BID
        dosing_interval_hours=12.0,
        route="Oral",
        fda_approved=True,
        approval_year=2013,
        approved_indications=["Melanoma (BRAF V600E/K)", "NSCLC (BRAF V600E)"],
    ),

    "osimertinib": Drug(
        name="Osimertinib",
        generic_name="osimertinib mesylate",
        drug_class=DrugClass.TARGETED_THERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.77,  # Oral tablet
            volume_of_distribution=986.0,  # L
            half_life=48.0,  # hours
            clearance=14.2,  # L/h
            protein_binding=0.95,
            tissue_penetration=0.65,  # Good CNS penetration
        ),
        mechanism_of_action="3rd generation EGFR TKI, targets T790M resistance mutation",
        target_proteins=["EGFR"],
        molecular_weight=499.6,  # g/mol - osimertinib
        ic50=0.0012,  # μM (1.2 nM for EGFR T790M) - very potent
        ec50=0.006,
        emax=0.93,
        hill_coefficient=2.7,
        cell_cycle_specific=False,
        resistance_mutations=["EGFR_C797S", "MET_amplification"],
        standard_dose_mg=80.0,  # mg QD
        dosing_interval_hours=24.0,
        route="Oral",
        fda_approved=True,
        approval_year=2015,
        approved_indications=["NSCLC (EGFR T790M or exon 19/L858R)"],
    ),

    "abiraterone": Drug(
        name="Abiraterone",
        generic_name="abiraterone acetate",
        drug_class=DrugClass.HORMONE_THERAPY,
        pk_model=PharmacokineticModel(
            bioavailability=0.05,  # Very low oral, increased with food
            volume_of_distribution=19700.0,  # L - massive
            half_life=15.0,  # hours (abiraterone acetate: 7-17h)
            clearance=1313.0,  # L/h
            protein_binding=0.99,
            tissue_penetration=0.5,
            prodrug=True,  # Abiraterone acetate → abiraterone
        ),
        mechanism_of_action="CYP17 inhibitor, blocks androgen synthesis",
        target_proteins=["CYP17A1"],
        molecular_weight=391.6,  # g/mol - abiraterone acetate
        ic50=5.0,  # μM (for CYP17 inhibition)
        ec50=10.0,
        emax=0.8,  # Hormonal suppression, not direct cytotoxicity
        hill_coefficient=2.0,
        cell_cycle_specific=False,
        resistance_mutations=["AR_amplification", "AR_V7_splice_variant"],
        hepatotoxicity=0.4,
        standard_dose_mg=1000.0,  # mg QD (must be fasted or 250mg with food)
        dosing_interval_hours=24.0,
        route="Oral",
        fda_approved=True,
        approval_year=2011,
        approved_indications=["Prostate cancer (metastatic castration-resistant)"],
    ),

    "radium-223": Drug(
        name="Radium-223",
        generic_name="radium-223 dichloride",
        drug_class=DrugClass.CHEMOTHERAPY,  # Radiopharmaceutical
        pk_model=PharmacokineticModel(
            bioavailability=1.0,  # IV injection
            volume_of_distribution=0.1,  # L (extremely small, bone-seeking)
            half_life=11.4 * 24,  # 11.4 days (physical half-life)
            clearance=0.0061,  # L/h (very slow, bone localization)
            protein_binding=0.0,  # Binds to bone hydroxyapatite, not plasma proteins
            tissue_penetration=0.95,  # Highly selective for bone metastases
        ),
        mechanism_of_action="Alpha-emitting radioisotope, targets bone metastases with high-LET radiation",
        target_proteins=["Hydroxyapatite"],  # Bone targeting
        molecular_weight=223.0,  # g/mol - atomic mass
        ic50=0.0001,  # Not applicable (radiation damage, not concentration-dependent)
        ec50=0.0001,
        emax=0.7,  # Palliative, bone pain reduction
        hill_coefficient=1.0,
        cell_cycle_specific=False,  # Radiation kills all phases
        myelosuppression=0.6,
        standard_dose_mg=2.03e-06,  # approx 2.0e-6 mg (55 kBq/kg → 70 kg)

        dosing_interval_hours=28 * 24,  # Every 4 weeks x6 doses
        route="IV",
        fda_approved=True,
        approval_year=2013,
        approved_indications=["Prostate cancer (bone metastases)"],
    ),
}


class DrugSimulator:
    """
    Simulate drug administration and response in tumor
    """

    def __init__(self, drug: Drug):
        self.drug = drug
        self.doses: List[Tuple[float, float]] = []  # (time, dose_mg)
        self.time = 0.0  # hours

    def administer_dose(self, dose_mg: float, time_hours: float):
        """Record a drug administration"""
        self.doses.append((time_hours, dose_mg))

    def get_plasma_concentration(self, time_hours: float, weight_kg: float = 70.0) -> float:
        """
        Calculate current plasma concentration from all doses
        Superposition principle for multiple doses
        """
        total_concentration = 0.0

        for dose_time, dose_mg in self.doses:
            if time_hours >= dose_time:
                time_since_dose = time_hours - dose_time
                conc = self.drug.calculate_concentration(dose_mg, time_since_dose, weight_kg)
                total_concentration += conc

        return total_concentration

    def get_tumor_concentration(self, time_hours: float, weight_kg: float = 70.0) -> float:
        """Get drug concentration in tumor"""
        plasma_conc = self.get_plasma_concentration(time_hours, weight_kg)
        return self.drug.calculate_tumor_concentration(plasma_conc)

    def get_effect(self, time_hours: float, weight_kg: float = 70.0) -> float:
        """Get therapeutic effect at given time"""
        tumor_conc = self.get_tumor_concentration(time_hours, weight_kg)
        return self.drug.calculate_effect(tumor_conc)

    def simulate_regimen(self, duration_days: float, weight_kg: float = 70.0) -> Dict:
        """
        Simulate standard dosing regimen
        Returns concentration and effect over time
        """
        duration_hours = duration_days * 24.0
        dosing_interval = self.drug.dosing_interval_hours

        # Administer doses
        current_time = 0.0
        while current_time <= duration_hours:
            self.administer_dose(self.drug.standard_dose_mg, current_time)
            current_time += dosing_interval

        # Calculate concentrations over time
        time_points = np.linspace(0, duration_hours, 1000)
        plasma_conc = [self.get_plasma_concentration(t, weight_kg) for t in time_points]
        tumor_conc = [self.get_tumor_concentration(t, weight_kg) for t in time_points]
        effects = [self.get_effect(t, weight_kg) for t in time_points]

        return {
            'time_hours': time_points,
            'time_days': time_points / 24.0,
            'plasma_concentration_uM': plasma_conc,
            'tumor_concentration_uM': tumor_conc,
            'therapeutic_effect': effects,
            'doses': self.doses,
        }


def get_drug_from_database(drug_name: str) -> Optional[Drug]:
    """Retrieve drug from database by name"""
    return DRUG_DATABASE.get(drug_name.lower())


def list_available_drugs() -> List[str]:
    """List all drugs in database"""
    return list(DRUG_DATABASE.keys())
