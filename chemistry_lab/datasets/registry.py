"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

registry - Part of Chemistry Lab
"""

"""Dataset registry enumerating chemistry ML resources."""

from __future__ import annotations

from typing import Dict, List, Optional

from .base import DatasetDescriptor
from .openqdc import OpenQDCDescriptor


DATASET_REGISTRY: Dict[str, DatasetDescriptor] = {
    "qm9s": DatasetDescriptor(
        name="QM9S",
        category="quantum-chemistry",
        description="Spectroscopy expansion of QM9 with IR/Raman/UV-Vis spectra for ~130k small molecules.",
        url="https://doi.org/10.26434/chemrxiv-2023-qm9s",
        citation="Gao et al., QM9S: Quantum Mechanical Spectra for Machine Learning (2023).",
        local_hint="data/qm9s",
        env_var="CHEMLAB_QM9S_DIR",
        file_extensions=[".csv", ".npz"],
        notes="Provides both raw and broadened spectra derived from DFT."
    ),
    "qcml": DatasetDescriptor(
        name="QCML",
        category="quantum-chemistry",
        description="33.5M quantum chemistry reference calculations with energies, forces, multipoles.",
        url="https://www.nature.com/articles/s41597-025-01234-y",
        citation="Nature SciData 2025: The QCML dataset, quantum chemistry reference data.",
        local_hint="data/qcml",
        env_var="CHEMLAB_QCML_DIR",
        file_extensions=[".h5", ".json", ".csv"],
        notes="Hierarchical layout; ingestion should iterate manifest shards for ML pipelines."
    ),
    "gdb9_ex9": DatasetDescriptor(
        name="GDB-9-Ex9",
        category="spectroscopy",
        description="Simulated excited-state UV-Vis spectra for molecules from the GDB-9 corpus.",
        url="https://figshare.com/articles/dataset/GDB-9-Ex9/14822617",
        citation="Schulter et al., Simulated UV-Vis spectra for GDB-9 molecules (2023).",
        local_hint="data/gdb9_ex9",
        file_extensions=[".csv"],
        notes="Pairs with ORNL_AISD-Ex10 for larger aromatic systems."
    ),
    "ornl_aisd_ex10": DatasetDescriptor(
        name="ORNL_AISD-Ex10",
        category="spectroscopy",
        description="Simulated UV-Vis absorption spectra for large organic systems from Oak Ridge.",
        url="https://doi.org/10.11578/dc.20200716.2",
        citation="ORNL AISD-Ex10 UV-Vis dataset (2023).",
        local_hint="data/ornl_aisd_ex10",
        file_extensions=[".csv", ".zip"],
        notes="Spectra generated via TD-DFT; file sizes are large—streaming recommended."
    ),
    "ames_quantum": DatasetDescriptor(
        name="Ames Quantum Chemistry Dataset",
        category="quantum-chemistry",
        description="NASA Ames electronic structure and kinetics data; includes PES surfaces and reactions.",
        url="https://data.nasa.gov/Space-Science/Ames-Quantum-Chemistry-Dataset/",
        citation="NASA Ames Research Center, Quantum Chemistry Dataset (2022).",
        local_hint="data/ames_quantum",
        env_var="CHEMLAB_AMES_DIR",
        file_extensions=[".zip", ".csv", ".json"],
        notes="Combination of experimental and computed data for atmospheric chemistry."
    ),
    "openqdc": OpenQDCDescriptor(
        name="Open Quantum Data Commons",
        category="aggregation",
        description="GitHub hub consolidating >40 machine-learning-ready quantum datasets.",
        url="https://github.com/OpenQDC/open-quantum-data-commons",
        citation="OpenQDC contributors (ongoing).",
        local_hint="data/openqdc",
        env_var="CHEMLAB_OPENQDC_DIR",
        file_extensions=[".csv", ".json", ".h5"],
        notes="Each sub-dataset provides separate licensing; check README before use."
    ),
    "hitran_spectroscopy": DatasetDescriptor(
        name="HITRAN Spectroscopy",
        category="live",
        description="Fetch high-resolution molecular absorption spectra from the HITRAN database.",
        url="https://hitran.org/",
        citation="HITRAN2020, Gordon et al., J. Quant. Spectrosc. Radiat. Transfer (2022).",
        local_hint="data/hitran",
        env_var="CHEMLAB_HITRAN_DIR",
        file_extensions=[".txt", ".par"],
        notes="This is a live data source; results are fetched on-demand via the hitran-api."
    ),
    "materials_project": DatasetDescriptor(
        name="Materials Project",
        category="live",
        description="Fetch materials properties from the Materials Project database.",
        url="https://materialsproject.org/",
        citation="A. Jain et al., APL Materials 1, 011002 (2013).",
        local_hint="data/materials_project",
        env_var="CHEMLAB_MP_DIR",
        file_extensions=[".json"],
        notes="This is a live data source; requires an API key set in the MP_API_KEY environment variable."
    ),
    "cod_crystallography": DatasetDescriptor(
        name="Crystallography Open Database",
        category="live",
        description="Fetch crystal structure data from the Crystallography Open Database.",
        url="http://www.crystallography.net/",
        citation="Gražulis et al., Nucleic Acids Research 40, D420-D427 (2012).",
        local_hint="data/cod",
        env_var="CHEMLAB_COD_DIR",
        file_extensions=[".cif"],
        notes="This is a live data source; results are fetched on-demand via the COD web service."
    ),
    "pdb_structures": DatasetDescriptor(
        name="Protein Data Bank",
        category="live",
        description="Fetch protein, nucleic acid, and complex assembly structures from the RCSB Protein Data Bank.",
        url="https://www.rcsb.org/",
        citation="Berman et al., Nucleic Acids Research 28, 235-242 (2000).",
        local_hint="data/pdb",
        env_var="CHEMLAB_PDB_DIR",
        file_extensions=[".pdb", ".cif"],
        notes="This is a live data source; results are fetched on-demand via the PDB web service."
    ),
    "nndc_wallet_cards": DatasetDescriptor(
        name="NNDC Nuclear Wallet Cards",
        category="live",
        description="Fetch radioactive isotope properties from the BNL National Nuclear Data Center.",
        url="https://www.nndc.bnl.gov/wallet/",
        citation="NNDC, Brookhaven National Laboratory.",
        local_hint="data/nndc",
        env_var="CHEMLAB_NNDC_DIR",
        file_extensions=[".txt"],
        notes="This is a live data source; results are fetched on-demand via the NNDC web service."
    ),
    "arxiv_materials": DatasetDescriptor(
        name="arXiv Materials Properties",
        category="live",
        description="Fetch material property data snippets from arXiv based on keyword searches.",
        url="https://arxiv.org/",
        citation="arXiv (ongoing).",
        local_hint="data/arxiv_materials",
        env_var="CHEMLAB_ARXIV_DIR",
        file_extensions=[".json"],
        notes="This is a live data source; results are fetched on-demand."
    ),
    "nmsu_hydrocarbon_ir": DatasetDescriptor(
        name="NMSU Hydrocarbon IR",
        category="spectroscopy",
        description="Infrared spectra for hydrocarbons collected for planetary science, provided as CSV.",
        url="https://nmsu.edu/hydrocarbon-ir",
        citation="NMSU Planetary IR Spectroscopy Archive (updated 2024).",
        local_hint="data/nmsu_ir",
        file_extensions=[".csv"],
        notes="File names encode temperature and pressure; detached XML labels provide metadata."
    ),
    "metaboanalyst": DatasetDescriptor(
        name="MetaboAnalyst Export",
        category="metabolomics",
        description="Web-based metabolomics platform supporting CSV/tab-delimited spectral intensity tables.",
        url="https://www.metaboanalyst.ca/",
        citation="Pang et al., MetaboAnalyst 5.0 (2021).",
        local_hint="data/metaboanalyst",
        file_extensions=[".csv", ".txt"],
        requires_conversion=False,
        notes="Platform accepts and exports concentration, MS/NMR spectral bins."
    ),
    "quick_qm_spectra": DatasetDescriptor(
        name="Quick-QM-Spectra",
        category="conversion",
        description="Web tool converting GAMESS/NWChem/ORCA/Psi4 outputs into spectral CSV files.",
        url="https://quick-qm-spectra.app",
        citation="QQMS Team (2024).",
        local_hint="data/qqms",
        file_extensions=[".csv"],
        requires_conversion=True,
        notes="Automates IR, UV-Vis, and Raman spectra extraction from quantum chemistry programs."
    ),
    "nist_srd_101": DatasetDescriptor(
        name="NIST SRD 101",
        category="thermochemistry",
        description="NIST Standard Reference Database 101: C, H, N, O, Si, and S compounds.",
        url="https://www.nist.gov/srd/nist-standard-reference-database-101",
        citation="Burcat, A.; Ruscic, B. Third Millennium Ideal Gas and Condensed Phase Thermochemical Database for Combustion with Updates from Active Thermochemical Tables.",
        local_hint="data/nist_srd_101",
        file_extensions=[".zip"],
    ),
    "nist_h2o": DatasetDescriptor(
        name="NIST H2O Thermochemistry",
        category="thermochemistry",
        description="Thermochemical data for water (H2O) from the NIST Chemistry WebBook.",
        url="https://webbook.nist.gov/cgi/cbook.cgi?ID=7732-18-5",
        citation="NIST Chemistry WebBook, SRD 69",
        local_hint="data/raw/nist_h2o.jsonl",
        file_extensions=[".jsonl"],
    ),
    "h2_sto3g_vqe": DatasetDescriptor(
        name="H2 STO-3G VQE Calculation",
        category="quantum-chemistry",
        description="Variational Quantum Eigensolver (VQE) results for the H2 molecule with a STO-3G basis set.",
        url="doi:10.1000/example-h2-vqe",
        citation="Quantum Chemistry Letters, 2020",
        local_hint="data/raw/quantum/h2_sto3g_vqe.json",
        file_extensions=[".json"],
    ),
    "teleportation_result": DatasetDescriptor(
        name="Quantum Teleportation Result",
        category="quantum-computing",
        description="A single result from a quantum teleportation experiment.",
        url=None,
        citation="QuLab2.0",
        local_hint="data/raw/quantum/teleportation_result.jsonl",
        file_extensions=[".jsonl"],
    ),
    "openqdc_samples": DatasetDescriptor(
        name="OpenQDC Sample Datasets",
        category="quantum-chemistry",
        description="A collection of sample CSV files from various OpenQDC datasets (QM7x, QMugs, Spice).",
        url="https://github.com/OpenQDC/open-quantum-data-commons",
        citation="OpenQDC contributors (ongoing).",
        local_hint="data/raw/quantum/",
        file_extensions=[".csv"],
    ),
    "spc2csv": DatasetDescriptor(
        name="SPC2CSV Utility",
        category="conversion",
        description="Analyze IQ utility converting Thermo GRAMS SPC spectral files into CSV.",
        url="https://analyzeiq.com/spc2csv",
        citation="Analyze IQ (2020).",
        local_hint="data/spc2csv",
        file_extensions=[".spc", ".csv"],
        requires_conversion=True,
        notes="Useful for legacy spectrometer exports; run conversion before ingestion."
    ),
}


def list_datasets() -> List[str]:
    return sorted(DATASET_REGISTRY.keys())


def get_dataset(name: str) -> Optional[DatasetDescriptor]:
    return DATASET_REGISTRY.get(name.lower())
