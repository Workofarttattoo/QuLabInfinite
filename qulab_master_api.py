"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

QuLabInfinite Master API - Unified Interface for 80+ Scientific Labs
=====================================================================

This module provides a comprehensive, unified interface to all QuLabInfinite laboratories.
It handles imports gracefully, categorizes labs by domain, provides search functionality,
and offers a complete API for lab discovery and execution.

Features:
    - Unified interface for 80+ specialized scientific laboratories
    - Automatic categorization by domain (Physics, Chemistry, Biology, etc.)
    - Graceful error handling for missing dependencies
    - Search functionality across all labs
    - Demonstration and capability inspection
    - Production-ready with comprehensive logging

Usage:
    >>> from qulab_master_api import QuLabMasterAPI
    >>>
    >>> # Initialize the master API
    >>> api = QuLabMasterAPI()
    >>>
    >>> # List all available labs
    >>> labs = api.list_labs()
    >>> print(f"Total labs: {len(labs)}")
    >>>
    >>> # Get specific lab
    >>> quantum_lab = api.get_lab("quantum_mechanics")
    >>>
    >>> # Search for labs
    >>> bio_labs = api.search_labs("biology")
    >>>
    >>> # Run demonstration
    >>> result = api.run_demo("quantum_mechanics")
    >>>
    >>> # Get capabilities
    >>> caps = api.get_capabilities("materials_science")
"""

import sys
import os
import importlib
import logging
import traceback
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class LabDomain(Enum):
    """Scientific domains for lab categorization"""
    PHYSICS = "Physics"
    CHEMISTRY = "Chemistry"
    BIOLOGY = "Biology"
    ENGINEERING = "Engineering"
    MEDICINE = "Medicine"
    EARTH_SCIENCE = "Earth Science"
    COMPUTER_SCIENCE = "Computer Science"
    MATHEMATICS = "Mathematics"
    MATERIALS = "Materials Science"
    QUANTUM = "Quantum Science"
    UNKNOWN = "Unknown"


@dataclass
class LabMetadata:
    """Metadata for a laboratory"""
    name: str
    display_name: str
    description: str
    domain: LabDomain
    module_path: str
    class_name: str
    available: bool = False
    instance: Optional[Any] = None
    error_message: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


class QuLabMasterAPI:
    """
    Unified Master API for QuLabInfinite

    Provides comprehensive access to all 80+ scientific laboratories with
    intelligent categorization, search, and execution capabilities.
    """

    def __init__(self, auto_load: bool = True, verbose: bool = True):
        """
        Initialize the Master API

        Args:
            auto_load: Automatically load all labs on initialization
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self._labs: Dict[str, LabMetadata] = {}
        self._domain_index: Dict[LabDomain, List[str]] = {}
        self._keyword_index: Dict[str, List[str]] = {}

        if self.verbose:
            logger.info("Initializing QuLabInfinite Master API")

        # Define lab registry with all 80+ labs
        self._lab_definitions = self._build_lab_registry()

        if auto_load:
            self.load_all_labs()

    def _build_lab_registry(self) -> Dict[str, Dict[str, Any]]:
        """
        Build comprehensive registry of all labs

        Returns:
            Dictionary mapping lab names to their metadata
        """
        return {
            # Physics Labs
            "quantum_mechanics": {
                "display_name": "Quantum Mechanics Lab",
                "description": "Quantum mechanics simulations and calculations",
                "domain": LabDomain.PHYSICS,
                "module": "quantum_mechanics_lab",
                "class": "QuantumMechanicsLab",
                "keywords": ["quantum", "mechanics", "wavefunction", "schrodinger"]
            },
            "quantum_computing": {
                "display_name": "Quantum Computing Lab",
                "description": "Quantum algorithms and circuit design",
                "domain": LabDomain.QUANTUM,
                "module": "quantum_computing_lab",
                "class": "QuantumComputingLab",
                "keywords": ["quantum", "computing", "qubits", "algorithms"]
            },
            "quantum_lab": {
                "display_name": "Advanced Quantum Lab",
                "description": "Advanced quantum simulations and optimization",
                "domain": LabDomain.QUANTUM,
                "module": "quantum_lab.quantum_lab",
                "class": "QuantumLab",
                "keywords": ["quantum", "simulation", "optimization"]
            },
            "particle_physics": {
                "display_name": "Particle Physics Lab",
                "description": "Particle physics simulations and analysis",
                "domain": LabDomain.PHYSICS,
                "module": "particle_physics_lab",
                "class": "ParticlePhysicsLab",
                "keywords": ["particle", "physics", "collider", "standard model"]
            },
            "nuclear_physics": {
                "display_name": "Nuclear Physics Lab",
                "description": "Nuclear reactions and decay processes",
                "domain": LabDomain.PHYSICS,
                "module": "nuclear_physics_lab",
                "class": "NuclearPhysicsLab",
                "keywords": ["nuclear", "physics", "fission", "fusion"]
            },
            "nuclear_lab": {
                "display_name": "Nuclear Lab",
                "description": "Advanced nuclear physics research",
                "domain": LabDomain.PHYSICS,
                "module": "nuclear_physics_lab.nuclear_lab",
                "class": "NuclearLab",
                "keywords": ["nuclear", "reactor", "isotopes"]
            },
            "condensed_matter_physics": {
                "display_name": "Condensed Matter Physics Lab",
                "description": "Solid state and condensed matter physics",
                "domain": LabDomain.PHYSICS,
                "module": "condensed_matter_physics_lab",
                "class": "CondensedMatterPhysicsLab",
                "keywords": ["condensed matter", "solid state", "superconductivity"]
            },
            "plasma_physics": {
                "display_name": "Plasma Physics Lab",
                "description": "Plasma dynamics and fusion research",
                "domain": LabDomain.PHYSICS,
                "module": "plasma_physics_lab",
                "class": "PlasmaPhysicsLab",
                "keywords": ["plasma", "physics", "fusion", "tokamak"]
            },
            "astrophysics": {
                "display_name": "Astrophysics Lab",
                "description": "Astrophysics simulations and cosmology",
                "domain": LabDomain.PHYSICS,
                "module": "astrophysics_lab",
                "class": "AstrophysicsLab",
                "keywords": ["astrophysics", "cosmology", "stars", "galaxies"]
            },
            "fluid_dynamics": {
                "display_name": "Fluid Dynamics Lab",
                "description": "Computational fluid dynamics simulations",
                "domain": LabDomain.PHYSICS,
                "module": "fluid_dynamics_lab",
                "class": "FluidDynamicsLab",
                "keywords": ["fluid", "dynamics", "cfd", "flow"]
            },
            "thermodynamics": {
                "display_name": "Thermodynamics Lab",
                "description": "Thermodynamic systems and processes",
                "domain": LabDomain.PHYSICS,
                "module": "thermodynamics_lab",
                "class": "ThermodynamicsLab",
                "keywords": ["thermodynamics", "heat", "entropy", "energy"]
            },
            "electromagnetism": {
                "display_name": "Electromagnetism Lab",
                "description": "Electromagnetic field theory and applications",
                "domain": LabDomain.PHYSICS,
                "module": "electromagnetism_lab",
                "class": "ElectromagnetismLab",
                "keywords": ["electromagnetism", "maxwell", "fields", "waves"]
            },
            "optics_photonics": {
                "display_name": "Optics & Photonics Lab",
                "description": "Optical systems and photonics",
                "domain": LabDomain.PHYSICS,
                "module": "optics_and_photonics_lab",
                "class": "OpticsPhotonicsLab",
                "keywords": ["optics", "photonics", "lasers", "light"]
            },

            # Chemistry Labs
            "chemistry": {
                "display_name": "Chemistry Lab",
                "description": "General chemistry experiments and synthesis",
                "domain": LabDomain.CHEMISTRY,
                "module": "chemistry_lab.chemistry_lab",
                "class": "ChemistryLab",
                "keywords": ["chemistry", "synthesis", "reactions"]
            },
            "organic_chemistry": {
                "display_name": "Organic Chemistry Lab",
                "description": "Organic synthesis and reaction mechanisms",
                "domain": LabDomain.CHEMISTRY,
                "module": "organic_chemistry_lab",
                "class": "OrganicChemistryLab",
                "keywords": ["organic", "chemistry", "synthesis", "molecules"]
            },
            "inorganic_chemistry": {
                "display_name": "Inorganic Chemistry Lab",
                "description": "Inorganic compounds and coordination chemistry",
                "domain": LabDomain.CHEMISTRY,
                "module": "inorganic_chemistry_lab",
                "class": "InorganicChemistryLab",
                "keywords": ["inorganic", "chemistry", "coordination", "metals"]
            },
            "physical_chemistry": {
                "display_name": "Physical Chemistry Lab",
                "description": "Physical chemistry and kinetics",
                "domain": LabDomain.CHEMISTRY,
                "module": "physical_chemistry_lab",
                "class": "PhysicalChemistryLab",
                "keywords": ["physical", "chemistry", "kinetics", "spectroscopy"]
            },
            "analytical_chemistry": {
                "display_name": "Analytical Chemistry Lab",
                "description": "Analytical techniques and instrumentation",
                "domain": LabDomain.CHEMISTRY,
                "module": "analytical_chemistry_lab",
                "class": "AnalyticalChemistryLab",
                "keywords": ["analytical", "chemistry", "chromatography", "spectroscopy"]
            },
            "computational_chemistry": {
                "display_name": "Computational Chemistry Lab",
                "description": "Molecular modeling and computational chemistry",
                "domain": LabDomain.CHEMISTRY,
                "module": "computational_chemistry_lab",
                "class": "ComputationalChemistryLab",
                "keywords": ["computational", "chemistry", "modeling", "dft"]
            },
            "polymer_chemistry": {
                "display_name": "Polymer Chemistry Lab",
                "description": "Polymer synthesis and characterization",
                "domain": LabDomain.CHEMISTRY,
                "module": "polymer_chemistry_lab",
                "class": "PolymerChemistryLab",
                "keywords": ["polymer", "chemistry", "plastics", "materials"]
            },
            "biochemistry": {
                "display_name": "Biochemistry Lab",
                "description": "Biochemical pathways and biomolecules",
                "domain": LabDomain.CHEMISTRY,
                "module": "biochemistry_lab",
                "class": "BiochemistryLab",
                "keywords": ["biochemistry", "metabolism", "enzymes", "proteins"]
            },
            "electrochemistry": {
                "display_name": "Electrochemistry Lab",
                "description": "Electrochemical systems and batteries",
                "domain": LabDomain.CHEMISTRY,
                "module": "electrochemistry_lab",
                "class": "ElectrochemistryLab",
                "keywords": ["electrochemistry", "batteries", "fuel cells", "redox"]
            },
            "catalysis": {
                "display_name": "Catalysis Lab",
                "description": "Catalytic reactions and catalyst design",
                "domain": LabDomain.CHEMISTRY,
                "module": "catalysis_lab",
                "class": "CatalysisLab",
                "keywords": ["catalysis", "catalyst", "reactions", "kinetics"]
            },
            "atmospheric_chemistry": {
                "display_name": "Atmospheric Chemistry Lab",
                "description": "Atmospheric chemical processes",
                "domain": LabDomain.CHEMISTRY,
                "module": "atmospheric_chemistry_lab",
                "class": "AtmosphericChemistryLab",
                "keywords": ["atmospheric", "chemistry", "air quality", "ozone"]
            },

            # Materials Science Labs
            "materials_science": {
                "display_name": "Materials Science Lab",
                "description": "Material properties and characterization",
                "domain": LabDomain.MATERIALS,
                "module": "materials_science_lab",
                "class": "MaterialsScienceLab",
                "keywords": ["materials", "science", "properties", "characterization"]
            },
            "materials_lab": {
                "display_name": "Advanced Materials Lab",
                "description": "Advanced materials research and discovery",
                "domain": LabDomain.MATERIALS,
                "module": "materials_lab.materials_lab",
                "class": "MaterialsLab",
                "keywords": ["materials", "discovery", "properties"]
            },
            "materials_chemistry": {
                "display_name": "Materials Chemistry Lab",
                "description": "Materials synthesis and chemical properties",
                "domain": LabDomain.MATERIALS,
                "module": "materials_chemistry_lab",
                "class": "MaterialsChemistryLab",
                "keywords": ["materials", "chemistry", "synthesis"]
            },

            # Biology Labs
            "molecular_biology": {
                "display_name": "Molecular Biology Lab",
                "description": "Molecular biology and gene expression",
                "domain": LabDomain.BIOLOGY,
                "module": "molecular_biology_lab",
                "class": "MolecularBiologyLab",
                "keywords": ["molecular", "biology", "dna", "rna", "genes"]
            },
            "cell_biology": {
                "display_name": "Cell Biology Lab",
                "description": "Cell structure and function",
                "domain": LabDomain.BIOLOGY,
                "module": "cell_biology_lab",
                "class": "CellBiologyLab",
                "keywords": ["cell", "biology", "cellular", "organelles"]
            },
            "genetics": {
                "display_name": "Genetics Lab",
                "description": "Genetic analysis and inheritance",
                "domain": LabDomain.BIOLOGY,
                "module": "genetics_lab",
                "class": "GeneticsLab",
                "keywords": ["genetics", "heredity", "alleles", "mutations"]
            },
            "genomics": {
                "display_name": "Genomics Lab",
                "description": "Genome sequencing and analysis",
                "domain": LabDomain.BIOLOGY,
                "module": "genomics_lab",
                "class": "GenomicsLab",
                "keywords": ["genomics", "sequencing", "genome", "bioinformatics"]
            },
            "genomics_advanced": {
                "display_name": "Advanced Genomics Lab",
                "description": "Advanced genomics research",
                "domain": LabDomain.BIOLOGY,
                "module": "genomics_lab.simulation",
                "class": "GenomicsLaboratory",
                "keywords": ["genomics", "advanced", "research"]
            },
            "bioinformatics": {
                "display_name": "Bioinformatics Lab",
                "description": "Computational biology and data analysis",
                "domain": LabDomain.BIOLOGY,
                "module": "bioinformatics_lab",
                "class": "BioinformaticsLab",
                "keywords": ["bioinformatics", "computational", "biology", "data"]
            },
            "microbiology": {
                "display_name": "Microbiology Lab",
                "description": "Microbial cultures and analysis",
                "domain": LabDomain.BIOLOGY,
                "module": "microbiology_lab",
                "class": "MicrobiologyLab",
                "keywords": ["microbiology", "bacteria", "microbes", "cultures"]
            },
            "immunology": {
                "display_name": "Immunology Lab",
                "description": "Immune system function and response",
                "domain": LabDomain.BIOLOGY,
                "module": "immunology_lab",
                "class": "ImmunologyLab",
                "keywords": ["immunology", "immune", "antibodies", "cells"]
            },
            "immunology_advanced": {
                "display_name": "Advanced Immunology Lab",
                "description": "Advanced immunology research",
                "domain": LabDomain.BIOLOGY,
                "module": "immunology_lab.immunology_lab",
                "class": "ImmunologyLab",
                "keywords": ["immunology", "advanced", "research"]
            },
            "neuroscience": {
                "display_name": "Neuroscience Lab",
                "description": "Neuroscience and brain function",
                "domain": LabDomain.BIOLOGY,
                "module": "neuroscience_lab",
                "class": "NeuroscienceLab",
                "keywords": ["neuroscience", "brain", "neurons", "cognition"]
            },
            "neuroscience_advanced": {
                "display_name": "Advanced Neuroscience Lab",
                "description": "Advanced neuroscience research",
                "domain": LabDomain.BIOLOGY,
                "module": "neuroscience_lab.neuroscience_lab",
                "class": "NeuroscienceLab",
                "keywords": ["neuroscience", "advanced", "neural"]
            },
            "developmental_biology": {
                "display_name": "Developmental Biology Lab",
                "description": "Developmental processes and embryology",
                "domain": LabDomain.BIOLOGY,
                "module": "developmental_biology_lab",
                "class": "DevelopmentalBiologyLab",
                "keywords": ["developmental", "biology", "embryology", "growth"]
            },
            "evolutionary_biology": {
                "display_name": "Evolutionary Biology Lab",
                "description": "Evolution and population genetics",
                "domain": LabDomain.BIOLOGY,
                "module": "evolutionary_biology_lab",
                "class": "EvolutionaryBiologyLab",
                "keywords": ["evolutionary", "biology", "evolution", "selection"]
            },
            "ecology": {
                "display_name": "Ecology Lab",
                "description": "Ecosystems and environmental interactions",
                "domain": LabDomain.BIOLOGY,
                "module": "ecology_lab",
                "class": "EcologyLab",
                "keywords": ["ecology", "ecosystems", "environment", "populations"]
            },
            "astrobiology": {
                "display_name": "Astrobiology Lab",
                "description": "Life in space and extreme environments",
                "domain": LabDomain.BIOLOGY,
                "module": "astrobiology_lab.astrobiology_lab",
                "class": "AstrobiologyLab",
                "keywords": ["astrobiology", "space", "life", "extremophiles"]
            },
            "proteomics": {
                "display_name": "Proteomics Lab",
                "description": "Protein analysis and characterization",
                "domain": LabDomain.BIOLOGY,
                "module": "proteomics_lab",
                "class": "ProteomicsLab",
                "keywords": ["proteomics", "proteins", "mass spectrometry"]
            },
            "protein_folding": {
                "display_name": "Protein Folding Lab",
                "description": "Protein structure and folding dynamics",
                "domain": LabDomain.BIOLOGY,
                "module": "protein_folding_lab_lab",
                "class": "ProteinFoldingLab",
                "keywords": ["protein", "folding", "structure", "dynamics"]
            },
            "protein_engineering": {
                "display_name": "Protein Engineering Lab",
                "description": "Protein design and engineering",
                "domain": LabDomain.BIOLOGY,
                "module": "protein_engineering_lab.protein_engineering_lab",
                "class": "ProteinEngineeringLab",
                "keywords": ["protein", "engineering", "design"]
            },
            "biological_quantum": {
                "display_name": "Biological Quantum Lab",
                "description": "Quantum effects in biological systems",
                "domain": LabDomain.QUANTUM,
                "module": "biological_quantum_lab",
                "class": "BiologicalQuantumLab",
                "keywords": ["biological", "quantum", "photosynthesis"]
            },

            # Medicine Labs
            "oncology": {
                "display_name": "Oncology Lab",
                "description": "Cancer biology and treatment",
                "domain": LabDomain.MEDICINE,
                "module": "oncology_lab",
                "class": "OncologyLab",
                "keywords": ["oncology", "cancer", "tumor", "treatment"]
            },
            "oncology_advanced": {
                "display_name": "Advanced Oncology Lab",
                "description": "Advanced oncology research",
                "domain": LabDomain.MEDICINE,
                "module": "oncology_lab.oncology_lab",
                "class": "OncologyLab",
                "keywords": ["oncology", "advanced", "cancer research"]
            },
            "realistic_tumor": {
                "display_name": "Realistic Tumor Lab",
                "description": "Realistic tumor growth modeling",
                "domain": LabDomain.MEDICINE,
                "module": "realistic_tumor_lab",
                "class": "TumorEvolutionLab",
                "keywords": ["tumor", "growth", "evolution", "modeling"]
            },
            "cardiology": {
                "display_name": "Cardiology Lab",
                "description": "Cardiovascular system and diseases",
                "domain": LabDomain.MEDICINE,
                "module": "cardiology_lab",
                "class": "CardiologyLab",
                "keywords": ["cardiology", "heart", "cardiovascular"]
            },
            "cardiology_advanced": {
                "display_name": "Advanced Cardiology Lab",
                "description": "Advanced cardiology research",
                "domain": LabDomain.MEDICINE,
                "module": "cardiology_lab.cardiology_lab",
                "class": "CardiologyLab",
                "keywords": ["cardiology", "advanced", "heart disease"]
            },
            "cardiovascular_plaque": {
                "display_name": "Cardiovascular Plaque Lab",
                "description": "Plaque formation and atherosclerosis",
                "domain": LabDomain.MEDICINE,
                "module": "cardiovascular_plaque_lab",
                "class": "CardiovascularPlaqueLab",
                "keywords": ["cardiovascular", "plaque", "atherosclerosis"]
            },
            "cardiac_fibrosis": {
                "display_name": "Cardiac Fibrosis Lab",
                "description": "Cardiac fibrosis prediction and analysis",
                "domain": LabDomain.MEDICINE,
                "module": "cardiac_fibrosis_predictor_lab",
                "class": "CardiacFibrosisLab",
                "keywords": ["cardiac", "fibrosis", "heart disease"]
            },
            "neurology": {
                "display_name": "Neurology Lab",
                "description": "Neurological disorders and treatment",
                "domain": LabDomain.MEDICINE,
                "module": "neurology_lab",
                "class": "NeurologyLab",
                "keywords": ["neurology", "neurological", "brain disorders"]
            },
            "drug_design": {
                "display_name": "Drug Design Lab",
                "description": "Computational drug design and discovery",
                "domain": LabDomain.MEDICINE,
                "module": "drug_design_lab",
                "class": "DrugDesignLab",
                "keywords": ["drug", "design", "discovery", "molecules"]
            },
            "pharmacology": {
                "display_name": "Pharmacology Lab",
                "description": "Drug action and pharmacokinetics",
                "domain": LabDomain.MEDICINE,
                "module": "pharmacology_lab",
                "class": "PharmacologyLab",
                "keywords": ["pharmacology", "drugs", "pharmacokinetics"]
            },
            "toxicology": {
                "display_name": "Toxicology Lab",
                "description": "Toxicity testing and assessment",
                "domain": LabDomain.MEDICINE,
                "module": "toxicology_lab",
                "class": "ToxicologyLab",
                "keywords": ["toxicology", "toxicity", "safety", "assessment"]
            },
            "medical_imaging": {
                "display_name": "Medical Imaging Lab",
                "description": "Medical imaging analysis and processing",
                "domain": LabDomain.MEDICINE,
                "module": "medical_imaging_lab",
                "class": "MedicalImagingLab",
                "keywords": ["medical", "imaging", "mri", "ct", "xray"]
            },
            "clinical_trials": {
                "display_name": "Clinical Trials Simulation Lab",
                "description": "Clinical trial design and simulation",
                "domain": LabDomain.MEDICINE,
                "module": "clinical_trials_simulation_lab",
                "class": "ClinicalTrialsLab",
                "keywords": ["clinical", "trials", "simulation", "testing"]
            },
            "drug_interaction": {
                "display_name": "Drug Interaction Simulator Lab",
                "description": "Drug-drug interaction prediction",
                "domain": LabDomain.MEDICINE,
                "module": "drug_interaction_simulator_lab",
                "class": "DrugInteractionLab",
                "keywords": ["drug", "interaction", "safety", "pharmacology"]
            },

            # Engineering Labs
            "chemical_engineering": {
                "display_name": "Chemical Engineering Lab",
                "description": "Chemical process design and optimization",
                "domain": LabDomain.ENGINEERING,
                "module": "chemical_engineering_lab",
                "class": "ChemicalEngineeringLab",
                "keywords": ["chemical", "engineering", "process", "design"]
            },
            "biomedical_engineering": {
                "display_name": "Biomedical Engineering Lab",
                "description": "Biomedical devices and systems",
                "domain": LabDomain.ENGINEERING,
                "module": "biomedical_engineering_lab",
                "class": "BiomedicalEngineeringLab",
                "keywords": ["biomedical", "engineering", "devices", "medical"]
            },
            "mechanical_engineering": {
                "display_name": "Mechanical Engineering Lab",
                "description": "Mechanical systems and design",
                "domain": LabDomain.ENGINEERING,
                "module": "mechanical_engineering_lab",
                "class": "MechanicalEngineeringLab",
                "keywords": ["mechanical", "engineering", "design", "systems"]
            },
            "electrical_engineering": {
                "display_name": "Electrical Engineering Lab",
                "description": "Electrical circuits and systems",
                "domain": LabDomain.ENGINEERING,
                "module": "electrical_engineering_lab",
                "class": "ElectricalEngineeringLab",
                "keywords": ["electrical", "engineering", "circuits", "electronics"]
            },
            "structural_engineering": {
                "display_name": "Structural Engineering Lab",
                "description": "Structural analysis and design",
                "domain": LabDomain.ENGINEERING,
                "module": "structural_engineering_lab",
                "class": "StructuralEngineeringLab",
                "keywords": ["structural", "engineering", "building", "design"]
            },
            "aerospace_engineering": {
                "display_name": "Aerospace Engineering Lab",
                "description": "Aerospace systems and flight dynamics",
                "domain": LabDomain.ENGINEERING,
                "module": "aerospace_engineering_lab",
                "class": "AerospaceEngineeringLab",
                "keywords": ["aerospace", "engineering", "flight", "rockets"]
            },
            "environmental_engineering": {
                "display_name": "Environmental Engineering Lab",
                "description": "Environmental systems and sustainability",
                "domain": LabDomain.ENGINEERING,
                "module": "environmental_engineering_lab",
                "class": "EnvironmentalEngineeringLab",
                "keywords": ["environmental", "engineering", "sustainability"]
            },
            "robotics": {
                "display_name": "Robotics Lab",
                "description": "Robotics and autonomous systems",
                "domain": LabDomain.ENGINEERING,
                "module": "robotics_lab",
                "class": "RoboticsLab",
                "keywords": ["robotics", "robots", "autonomous", "control"]
            },
            "control_systems": {
                "display_name": "Control Systems Lab",
                "description": "Control theory and system design",
                "domain": LabDomain.ENGINEERING,
                "module": "control_systems_lab",
                "class": "ControlSystemsLab",
                "keywords": ["control", "systems", "feedback", "automation"]
            },

            # Earth Science Labs
            "geology": {
                "display_name": "Geology Lab",
                "description": "Geological processes and structures",
                "domain": LabDomain.EARTH_SCIENCE,
                "module": "geology_lab",
                "class": "GeologyLab",
                "keywords": ["geology", "rocks", "minerals", "earth"]
            },
            "seismology": {
                "display_name": "Seismology Lab",
                "description": "Earthquake analysis and seismic waves",
                "domain": LabDomain.EARTH_SCIENCE,
                "module": "seismology_lab",
                "class": "SeismologyLab",
                "keywords": ["seismology", "earthquakes", "seismic", "waves"]
            },
            "geophysics": {
                "display_name": "Geophysics Lab",
                "description": "Geophysical processes and measurements",
                "domain": LabDomain.EARTH_SCIENCE,
                "module": "geophysics_lab.geophysics_lab",
                "class": "GeophysicsLab",
                "keywords": ["geophysics", "earth", "magnetic", "gravity"]
            },
            "meteorology": {
                "display_name": "Meteorology Lab",
                "description": "Weather systems and forecasting",
                "domain": LabDomain.EARTH_SCIENCE,
                "module": "meteorology_lab",
                "class": "MeteorologyLab",
                "keywords": ["meteorology", "weather", "atmosphere", "forecast"]
            },
            "atmospheric_science": {
                "display_name": "Atmospheric Science Lab",
                "description": "Atmospheric processes and climate",
                "domain": LabDomain.EARTH_SCIENCE,
                "module": "atmospheric_science_lab.atmospheric_science_lab",
                "class": "AtmosphericScienceLab",
                "keywords": ["atmospheric", "science", "climate", "air"]
            },
            "oceanography": {
                "display_name": "Oceanography Lab",
                "description": "Ocean systems and marine science",
                "domain": LabDomain.EARTH_SCIENCE,
                "module": "oceanography_lab",
                "class": "OceanographyLab",
                "keywords": ["oceanography", "ocean", "marine", "sea"]
            },
            "hydrology": {
                "display_name": "Hydrology Lab",
                "description": "Water systems and hydrology",
                "domain": LabDomain.EARTH_SCIENCE,
                "module": "hydrology_lab",
                "class": "HydrologyLab",
                "keywords": ["hydrology", "water", "rivers", "groundwater"]
            },
            "climate_modeling": {
                "display_name": "Climate Modeling Lab",
                "description": "Climate simulation and prediction",
                "domain": LabDomain.EARTH_SCIENCE,
                "module": "climate_modeling_lab",
                "class": "ClimateModelingLab",
                "keywords": ["climate", "modeling", "simulation", "global warming"]
            },
            "renewable_energy": {
                "display_name": "Renewable Energy Lab",
                "description": "Renewable energy systems and optimization",
                "domain": LabDomain.EARTH_SCIENCE,
                "module": "renewable_energy_lab",
                "class": "RenewableEnergyLab",
                "keywords": ["renewable", "energy", "solar", "wind", "sustainable"]
            },
            "carbon_capture": {
                "display_name": "Carbon Capture Lab",
                "description": "Carbon capture and storage technologies",
                "domain": LabDomain.EARTH_SCIENCE,
                "module": "carbon_capture_lab",
                "class": "CarbonCaptureLab",
                "keywords": ["carbon", "capture", "ccs", "climate"]
            },

            # Computer Science Labs
            "machine_learning": {
                "display_name": "Machine Learning Lab",
                "description": "Machine learning algorithms and models",
                "domain": LabDomain.COMPUTER_SCIENCE,
                "module": "machine_learning_lab",
                "class": "MachineLearningLab",
                "keywords": ["machine learning", "ml", "ai", "algorithms"]
            },
            "deep_learning": {
                "display_name": "Deep Learning Lab",
                "description": "Deep neural networks and architectures",
                "domain": LabDomain.COMPUTER_SCIENCE,
                "module": "deep_learning_lab",
                "class": "DeepLearningLab",
                "keywords": ["deep learning", "neural networks", "dl", "ai"]
            },
            "neural_networks": {
                "display_name": "Neural Networks Lab",
                "description": "Neural network design and training",
                "domain": LabDomain.COMPUTER_SCIENCE,
                "module": "neural_networks_lab",
                "class": "NeuralNetworksLab",
                "keywords": ["neural networks", "nn", "backpropagation"]
            },
            "natural_language_processing": {
                "display_name": "Natural Language Processing Lab",
                "description": "NLP and language models",
                "domain": LabDomain.COMPUTER_SCIENCE,
                "module": "natural_language_processing_lab",
                "class": "NLPLab",
                "keywords": ["nlp", "language", "text", "processing"]
            },
            "computer_vision": {
                "display_name": "Computer Vision Lab",
                "description": "Image processing and computer vision",
                "domain": LabDomain.COMPUTER_SCIENCE,
                "module": "computer_vision_lab",
                "class": "ComputerVisionLab",
                "keywords": ["computer vision", "image", "vision", "detection"]
            },
            "cryptography": {
                "display_name": "Cryptography Lab",
                "description": "Cryptographic algorithms and security",
                "domain": LabDomain.COMPUTER_SCIENCE,
                "module": "cryptography_lab",
                "class": "CryptographyLab",
                "keywords": ["cryptography", "encryption", "security", "crypto"]
            },
            "signal_processing": {
                "display_name": "Signal Processing Lab",
                "description": "Digital signal processing and analysis",
                "domain": LabDomain.COMPUTER_SCIENCE,
                "module": "signal_processing_lab",
                "class": "SignalProcessingLab",
                "keywords": ["signal", "processing", "dsp", "fourier"]
            },

            # Mathematics Labs
            "algorithm_design": {
                "display_name": "Algorithm Design Lab",
                "description": "Algorithm design and analysis",
                "domain": LabDomain.MATHEMATICS,
                "module": "algorithm_design_lab",
                "class": "AlgorithmDesignLab",
                "keywords": ["algorithms", "design", "complexity", "optimization"]
            },
            "graph_theory": {
                "display_name": "Graph Theory Lab",
                "description": "Graph algorithms and networks",
                "domain": LabDomain.MATHEMATICS,
                "module": "graph_theory_lab",
                "class": "GraphTheoryLab",
                "keywords": ["graph", "theory", "networks", "algorithms"]
            },
            "optimization_theory": {
                "display_name": "Optimization Theory Lab",
                "description": "Mathematical optimization methods",
                "domain": LabDomain.MATHEMATICS,
                "module": "optimization_theory_lab",
                "class": "OptimizationTheoryLab",
                "keywords": ["optimization", "theory", "linear", "nonlinear"]
            },

            # Specialized Labs
            "frequency": {
                "display_name": "Frequency Lab",
                "description": "Electromagnetic frequency analysis",
                "domain": LabDomain.PHYSICS,
                "module": "frequency_lab.frequency_lab",
                "class": "FrequencyLab",
                "keywords": ["frequency", "electromagnetic", "spectrum"]
            },
            "cognitive_science": {
                "display_name": "Cognitive Science Lab",
                "description": "Cognitive processes and modeling",
                "domain": LabDomain.COMPUTER_SCIENCE,
                "module": "cognitive_science_lab.cognitive_lab",
                "class": "CognitiveScienceLab",
                "keywords": ["cognitive", "science", "psychology", "cognition"]
            },
        }

    def load_all_labs(self) -> Dict[str, bool]:
        """
        Load all laboratories

        Returns:
            Dictionary mapping lab names to their load success status
        """
        results = {}

        if self.verbose:
            logger.info(f"Loading {len(self._lab_definitions)} laboratories...")

        for lab_name, lab_config in self._lab_definitions.items():
            success = self.load_lab(lab_name)
            results[lab_name] = success

        loaded_count = sum(1 for s in results.values() if s)
        total_count = len(results)

        if self.verbose:
            logger.info(f"Loaded {loaded_count}/{total_count} laboratories successfully")
            if loaded_count < total_count:
                failed = [name for name, success in results.items() if not success]
                logger.warning(f"Failed to load: {', '.join(failed[:5])}" +
                             (f" and {len(failed)-5} more" if len(failed) > 5 else ""))

        return results

    def load_lab(self, lab_name: str) -> bool:
        """
        Load a specific laboratory

        Args:
            lab_name: Name of the lab to load

        Returns:
            True if loaded successfully, False otherwise
        """
        if lab_name not in self._lab_definitions:
            logger.error(f"Unknown lab: {lab_name}")
            return False

        config = self._lab_definitions[lab_name]

        try:
            # Import the module
            module = importlib.import_module(config["module"])

            # Get the class
            lab_class = getattr(module, config["class"])

            # Try to instantiate
            try:
                instance = lab_class()
            except Exception as e:
                # Some labs might need specific parameters
                instance = None
                if self.verbose:
                    logger.debug(f"Could not instantiate {lab_name}: {e}")

            # Extract capabilities
            capabilities = []
            if instance:
                for attr in dir(instance):
                    if not attr.startswith('_') and callable(getattr(instance, attr)):
                        capabilities.append(attr)

            # Create metadata
            metadata = LabMetadata(
                name=lab_name,
                display_name=config["display_name"],
                description=config["description"],
                domain=config["domain"],
                module_path=config["module"],
                class_name=config["class"],
                available=True,
                instance=instance,
                capabilities=capabilities,
                keywords=config["keywords"]
            )

            self._labs[lab_name] = metadata

            # Index by domain
            if metadata.domain not in self._domain_index:
                self._domain_index[metadata.domain] = []
            self._domain_index[metadata.domain].append(lab_name)

            # Index by keywords
            for keyword in metadata.keywords:
                keyword_lower = keyword.lower()
                if keyword_lower not in self._keyword_index:
                    self._keyword_index[keyword_lower] = []
                self._keyword_index[keyword_lower].append(lab_name)

            return True

        except ImportError as e:
            error_msg = f"Import error: {str(e)}"
            if self.verbose:
                logger.debug(f"Failed to load {lab_name}: {error_msg}")

            # Still register with error
            metadata = LabMetadata(
                name=lab_name,
                display_name=config["display_name"],
                description=config["description"],
                domain=config["domain"],
                module_path=config["module"],
                class_name=config["class"],
                available=False,
                error_message=error_msg,
                keywords=config["keywords"]
            )
            self._labs[lab_name] = metadata
            return False

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"Failed to load {lab_name}: {error_msg}")

            metadata = LabMetadata(
                name=lab_name,
                display_name=config["display_name"],
                description=config["description"],
                domain=config["domain"],
                module_path=config["module"],
                class_name=config["class"],
                available=False,
                error_message=error_msg,
                keywords=config["keywords"]
            )
            self._labs[lab_name] = metadata
            return False

    def list_labs(self, domain: Optional[LabDomain] = None,
                  available_only: bool = False) -> List[Dict[str, Any]]:
        """
        List all laboratories

        Args:
            domain: Filter by specific domain (optional)
            available_only: Only return successfully loaded labs

        Returns:
            List of lab information dictionaries
        """
        labs = []

        for lab_name, metadata in self._labs.items():
            if domain and metadata.domain != domain:
                continue

            if available_only and not metadata.available:
                continue

            labs.append({
                "name": lab_name,
                "display_name": metadata.display_name,
                "description": metadata.description,
                "domain": metadata.domain.value,
                "available": metadata.available,
                "capabilities_count": len(metadata.capabilities),
                "error": metadata.error_message
            })

        # Sort by domain and name
        labs.sort(key=lambda x: (x["domain"], x["name"]))

        return labs

    def get_lab(self, lab_name: str) -> Optional[Any]:
        """
        Get a specific laboratory instance

        Args:
            lab_name: Name of the lab

        Returns:
            Lab instance if available, None otherwise
        """
        if lab_name not in self._labs:
            logger.error(f"Lab not found: {lab_name}")
            return None

        metadata = self._labs[lab_name]

        if not metadata.available:
            logger.error(f"Lab not available: {lab_name} - {metadata.error_message}")
            return None

        return metadata.instance

    def get_capabilities(self, lab_name: str) -> Dict[str, Any]:
        """
        Get capabilities of a specific lab

        Args:
            lab_name: Name of the lab

        Returns:
            Dictionary with lab capabilities and metadata
        """
        if lab_name not in self._labs:
            return {"error": f"Lab not found: {lab_name}"}

        metadata = self._labs[lab_name]

        return {
            "name": lab_name,
            "display_name": metadata.display_name,
            "description": metadata.description,
            "domain": metadata.domain.value,
            "available": metadata.available,
            "module": metadata.module_path,
            "class": metadata.class_name,
            "capabilities": metadata.capabilities,
            "keywords": metadata.keywords,
            "error": metadata.error_message
        }

    def search_labs(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for labs by keyword or description

        Args:
            query: Search query string

        Returns:
            List of matching lab information
        """
        query_lower = query.lower()
        matches = []

        # Search in keyword index first
        matching_names = set()
        for keyword, lab_names in self._keyword_index.items():
            if query_lower in keyword:
                matching_names.update(lab_names)

        # Search in descriptions and names
        for lab_name, metadata in self._labs.items():
            if (query_lower in lab_name.lower() or
                query_lower in metadata.display_name.lower() or
                query_lower in metadata.description.lower()):
                matching_names.add(lab_name)

        # Build result list
        for lab_name in matching_names:
            metadata = self._labs[lab_name]
            matches.append({
                "name": lab_name,
                "display_name": metadata.display_name,
                "description": metadata.description,
                "domain": metadata.domain.value,
                "available": metadata.available,
                "relevance_score": self._calculate_relevance(query_lower, metadata)
            })

        # Sort by relevance
        matches.sort(key=lambda x: x["relevance_score"], reverse=True)

        return matches

    def _calculate_relevance(self, query: str, metadata: LabMetadata) -> float:
        """Calculate relevance score for search results"""
        score = 0.0

        # Exact name match
        if query == metadata.name.lower():
            score += 100.0
        elif query in metadata.name.lower():
            score += 50.0

        # Display name match
        if query in metadata.display_name.lower():
            score += 30.0

        # Description match
        if query in metadata.description.lower():
            score += 20.0

        # Keyword matches
        for keyword in metadata.keywords:
            if query in keyword.lower():
                score += 10.0

        return score

    def run_demo(self, lab_name: str, **kwargs) -> Dict[str, Any]:
        """
        Run a demonstration of a specific lab

        Args:
            lab_name: Name of the lab
            **kwargs: Additional arguments for the demo

        Returns:
            Dictionary with demo results
        """
        lab = self.get_lab(lab_name)

        if not lab:
            return {"error": f"Lab not available: {lab_name}"}

        try:
            # Try common demo methods
            if hasattr(lab, 'run_demo'):
                result = lab.run_demo(**kwargs)
            elif hasattr(lab, 'demo'):
                result = lab.demo(**kwargs)
            elif hasattr(lab, 'example'):
                result = lab.example(**kwargs)
            elif hasattr(lab, 'test'):
                result = lab.test(**kwargs)
            else:
                return {
                    "error": f"Lab {lab_name} has no demo method",
                    "suggestion": f"Available methods: {', '.join(self._labs[lab_name].capabilities[:10])}"
                }

            return {
                "lab": lab_name,
                "success": True,
                "result": result
            }

        except Exception as e:
            logger.error(f"Demo failed for {lab_name}: {e}")
            return {
                "lab": lab_name,
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about loaded labs

        Returns:
            Dictionary with statistics
        """
        total = len(self._labs)
        available = sum(1 for m in self._labs.values() if m.available)
        unavailable = total - available

        by_domain = {}
        for domain, lab_names in self._domain_index.items():
            available_count = sum(1 for name in lab_names
                                if self._labs[name].available)
            by_domain[domain.value] = {
                "total": len(lab_names),
                "available": available_count
            }

        return {
            "total_labs": total,
            "available_labs": available,
            "unavailable_labs": unavailable,
            "success_rate": f"{(available/total*100):.1f}%",
            "by_domain": by_domain,
            "total_domains": len(self._domain_index),
            "total_keywords": len(self._keyword_index)
        }

    def export_catalog(self, output_path: Optional[str] = None) -> str:
        """
        Export complete lab catalog to JSON

        Args:
            output_path: Path to save JSON file (optional)

        Returns:
            JSON string of catalog
        """
        catalog = {
            "metadata": {
                "title": "QuLabInfinite Laboratory Catalog",
                "version": "1.0.0",
                "total_labs": len(self._labs),
                "generated": "2025-11-10"
            },
            "statistics": self.get_statistics(),
            "labs": []
        }

        for lab_name, metadata in sorted(self._labs.items()):
            catalog["labs"].append({
                "name": lab_name,
                "display_name": metadata.display_name,
                "description": metadata.description,
                "domain": metadata.domain.value,
                "module": metadata.module_path,
                "class": metadata.class_name,
                "available": metadata.available,
                "capabilities": metadata.capabilities,
                "keywords": metadata.keywords,
                "error": metadata.error_message
            })

        json_str = json.dumps(catalog, indent=2)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(json_str)
            logger.info(f"Catalog exported to: {output_path}")

        return json_str

    def __repr__(self) -> str:
        """String representation"""
        stats = self.get_statistics()
        return (f"QuLabMasterAPI("
                f"total={stats['total_labs']}, "
                f"available={stats['available_labs']}, "
                f"domains={stats['total_domains']})")


def main():
    """
    Command-line interface for QuLabInfinite Master API
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="QuLabInfinite Master API - Unified Laboratory Interface"
    )
    parser.add_argument(
        "command",
        choices=["list", "search", "info", "demo", "stats", "export"],
        help="Command to execute"
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="Target lab name or search query"
    )
    parser.add_argument(
        "--domain",
        choices=[d.value for d in LabDomain],
        help="Filter by domain"
    )
    parser.add_argument(
        "--available-only",
        action="store_true",
        help="Show only available labs"
    )
    parser.add_argument(
        "--output",
        help="Output file path (for export command)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    # Initialize API
    api = QuLabMasterAPI(verbose=not args.quiet)

    # Execute command
    if args.command == "list":
        domain = LabDomain(args.domain) if args.domain else None
        labs = api.list_labs(domain=domain, available_only=args.available_only)

        print(f"\n{'='*80}")
        print(f"QuLabInfinite Laboratory Catalog")
        print(f"{'='*80}\n")

        current_domain = None
        for lab in labs:
            if lab["domain"] != current_domain:
                current_domain = lab["domain"]
                print(f"\n{current_domain}")
                print("-" * len(current_domain))

            status = "✓" if lab["available"] else "✗"
            print(f"  {status} {lab['display_name']}")
            print(f"     {lab['description']}")
            if lab["error"]:
                print(f"     Error: {lab['error']}")

        print(f"\n{len(labs)} labs total\n")

    elif args.command == "search":
        if not args.target:
            print("Error: search query required")
            return 1

        results = api.search_labs(args.target)

        print(f"\nSearch results for: '{args.target}'")
        print(f"Found {len(results)} matches\n")

        for result in results:
            status = "✓" if result["available"] else "✗"
            print(f"{status} {result['display_name']}")
            print(f"   {result['description']}")
            print(f"   Domain: {result['domain']} | Relevance: {result['relevance_score']:.1f}")
            print()

    elif args.command == "info":
        if not args.target:
            print("Error: lab name required")
            return 1

        info = api.get_capabilities(args.target)

        if "error" in info:
            print(f"Error: {info['error']}")
            return 1

        print(f"\n{info['display_name']}")
        print("=" * len(info['display_name']))
        print(f"\nDescription: {info['description']}")
        print(f"Domain: {info['domain']}")
        print(f"Module: {info['module']}")
        print(f"Class: {info['class']}")
        print(f"Available: {'Yes' if info['available'] else 'No'}")

        if info['error']:
            print(f"Error: {info['error']}")

        if info['capabilities']:
            print(f"\nCapabilities ({len(info['capabilities'])}):")
            for cap in info['capabilities'][:20]:
                print(f"  - {cap}")
            if len(info['capabilities']) > 20:
                print(f"  ... and {len(info['capabilities']) - 20} more")

        if info['keywords']:
            print(f"\nKeywords: {', '.join(info['keywords'])}")
        print()

    elif args.command == "demo":
        if not args.target:
            print("Error: lab name required")
            return 1

        print(f"\nRunning demo for: {args.target}\n")
        result = api.run_demo(args.target)

        if "error" in result:
            print(f"Error: {result['error']}")
            if "suggestion" in result:
                print(f"Suggestion: {result['suggestion']}")
            return 1

        print("Demo completed successfully!")
        print(f"\nResult:")
        print(json.dumps(result, indent=2))
        print()

    elif args.command == "stats":
        stats = api.get_statistics()

        print(f"\nQuLabInfinite Statistics")
        print("=" * 40)
        print(f"Total Labs: {stats['total_labs']}")
        print(f"Available: {stats['available_labs']}")
        print(f"Unavailable: {stats['unavailable_labs']}")
        print(f"Success Rate: {stats['success_rate']}")
        print(f"\nBy Domain:")

        for domain, counts in sorted(stats['by_domain'].items()):
            print(f"  {domain}: {counts['available']}/{counts['total']}")

        print()

    elif args.command == "export":
        output_path = args.output or "qulab_catalog.json"
        catalog = api.export_catalog(output_path)
        print(f"Catalog exported to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
