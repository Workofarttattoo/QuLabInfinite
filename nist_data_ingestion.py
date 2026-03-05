#!/usr/bin/env python3
"""
NIST Data Ingestion System for QuLab Infinite
Ingests materials science data from various NIST databases

Supported NIST Databases:
- NIST Chemistry WebBook (thermodynamics, spectroscopy)
- NIST Materials Genome Initiative data
- NIST Structural Databases (crystallography)
- NIST Physical Reference Data
- NIST Atomic Spectra Database
- NIST Thermophysical Properties of Fluids

Author: QuLab Infinite AI Assistant
Date: March 2026
"""

import requests
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import time
import logging
from dataclasses import dataclass
from urllib.parse import urlencode, urljoin
import xml.etree.ElementTree as ET

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class NISTDatabase:
    """NIST database configuration"""
    name: str
    base_url: str
    api_type: str  # 'rest', 'soap', 'download'
    description: str
    data_types: List[str]

class NISTDataIngester:
    """Main NIST data ingestion system"""

    def __init__(self, data_dir: str = "nist_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # NIST database configurations
        self.databases = {
            'chemistry_webbook': NISTDatabase(
                name='NIST Chemistry WebBook',
                base_url='https://webbook.nist.gov/cgi/',
                api_type='rest',
                description='Thermodynamic, spectroscopic, and thermophysical data',
                data_types=['thermodynamics', 'spectroscopy', 'kinetics']
            ),
            'materials_genome': NISTDatabase(
                name='NIST Materials Genome',
                base_url='https://materialsdata.nist.gov/',
                api_type='rest',
                description='Materials properties and computational data',
                data_types=['properties', 'computational', 'experimental']
            ),
            'crystallography': NISTDatabase(
                name='NIST Crystallography',
                base_url='https://www.nist.gov/programs-projects/crystallographic-databases',
                api_type='download',
                description='Crystal structures and diffraction data',
                data_types=['structures', 'diffraction', 'symmetry']
            ),
            'atomic_spectra': NISTDatabase(
                name='NIST Atomic Spectra',
                base_url='https://physics.nist.gov/PhysRefData/ASD/',
                api_type='download',
                description='Atomic and ionic spectra data',
                data_types=['spectra', 'energy_levels', 'transitions']
            ),
            'fluid_properties': NISTDatabase(
                name='NIST Fluid Properties',
                base_url='https://webbook.nist.gov/chemistry/fluid/',
                api_type='rest',
                description='Thermophysical properties of fluids',
                data_types=['density', 'viscosity', 'thermal_conductivity']
            )
        }

        # Request session with proper headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'QuLab-Infinite-NIST-Ingester/1.0',
            'Accept': 'application/json, text/html, application/xml',
        })

    def search_chemistry_webbook(self, compound: str, data_type: str = 'thermo') -> Dict[str, Any]:
        """
        Search NIST Chemistry WebBook for compound data

        Args:
            compound: Chemical formula or name
            data_type: 'thermo', 'spectra', 'reaction', 'fluid'

        Returns:
            Dictionary containing compound data
        """
        logger.info(f"Searching Chemistry WebBook for {compound} ({data_type})")

        try:
            # Construct search URL
            if data_type == 'thermo':
                url = f"{self.databases['chemistry_webbook'].base_url}cbook.cgi?Name={compound}&Units=SI"
            elif data_type == 'spectra':
                url = f"{self.databases['chemistry_webbook'].base_url}cbook.cgi?Name={compound}&SpectrumType=IR"
            else:
                url = f"{self.databases['chemistry_webbook'].base_url}cbook.cgi?Name={compound}"

            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            # Parse the HTML response (basic parsing)
            data = {
                'compound': compound,
                'data_type': data_type,
                'url': url,
                'status': 'found' if 'NIST Chemistry WebBook' in response.text else 'not_found',
                'timestamp': time.time()
            }

            # Save raw data
            filename = f"chemistry_webbook_{compound}_{data_type}_{int(time.time())}.html"
            filepath = self.data_dir / 'chemistry_webbook' / filename
            filepath.parent.mkdir(exist_ok=True)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)

            data['local_file'] = str(filepath)

            return data

        except Exception as e:
            logger.error(f"Error searching Chemistry WebBook for {compound}: {e}")
            return {'compound': compound, 'error': str(e), 'timestamp': time.time()}

    def get_materials_genome_data(self, material_class: str = 'metals') -> List[Dict[str, Any]]:
        """
        Retrieve materials data from NIST Materials Genome Initiative

        Args:
            material_class: Type of materials to retrieve ('metals', 'ceramics', 'polymers', etc.)

        Returns:
            List of material property datasets
        """
        logger.info(f"Retrieving Materials Genome data for {material_class}")

        try:
            # NIST Materials Data Repository API
            base_url = "https://materialsdata.nist.gov/rest/"

            # Search for datasets
            search_url = f"{base_url}search"
            params = {
                'q': material_class,
                'wt': 'json',
                'rows': 100
            }

            response = self.session.get(search_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            datasets = []
            for doc in data.get('response', {}).get('docs', []):
                dataset = {
                    'id': doc.get('id'),
                    'title': doc.get('title'),
                    'description': doc.get('description', ''),
                    'material_class': material_class,
                    'timestamp': time.time()
                }
                datasets.append(dataset)

            # Save datasets info
            filename = f"materials_genome_{material_class}_{int(time.time())}.json"
            filepath = self.data_dir / 'materials_genome' / filename
            filepath.parent.mkdir(exist_ok=True)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(datasets, f, indent=2)

            logger.info(f"Saved {len(datasets)} datasets from Materials Genome")
            return datasets

        except Exception as e:
            logger.error(f"Error retrieving Materials Genome data: {e}")
            return []

    def download_crystallography_data(self) -> List[str]:
        """
        Download crystallography data files from NIST

        Returns:
            List of downloaded file paths
        """
        logger.info("Downloading crystallography data")

        try:
            # NIST Crystal Data: Powder Diffraction File (PDF)
            # This is a simplified example - actual implementation would need
            # specific file URLs and authentication

            download_urls = [
                "https://www.nist.gov/system/files/documents/2017/05/09/example_crystal_data.zip",
                # Add actual NIST crystal data URLs here
            ]

            downloaded_files = []

            for url in download_urls:
                try:
                    response = self.session.get(url, timeout=60)
                    response.raise_for_status()

                    filename = url.split('/')[-1]
                    filepath = self.data_dir / 'crystallography' / filename
                    filepath.parent.mkdir(exist_ok=True)

                    with open(filepath, 'wb') as f:
                        f.write(response.content)

                    downloaded_files.append(str(filepath))
                    logger.info(f"Downloaded {filename}")

                except Exception as e:
                    logger.error(f"Error downloading {url}: {e}")

            return downloaded_files

        except Exception as e:
            logger.error(f"Error in crystallography download: {e}")
            return []

    def get_atomic_spectra_data(self, element: str) -> Dict[str, Any]:
        """
        Retrieve atomic spectra data for an element

        Args:
            element: Chemical element symbol (e.g., 'Si', 'C', 'O')

        Returns:
            Dictionary containing spectra data
        """
        logger.info(f"Retrieving atomic spectra for {element}")

        try:
            # NIST Atomic Spectra Database
            base_url = "https://physics.nist.gov/cgi-bin/ASD/energy1.pl"

            params = {
                'spectra': element,
                'format': '1',  # Plain text format
                'units': '1',   # Energy in eV
            }

            response = self.session.get(base_url, params=params, timeout=30)
            response.raise_for_status()

            data = {
                'element': element,
                'data_type': 'atomic_spectra',
                'content': response.text,
                'timestamp': time.time()
            }

            # Save spectra data
            filename = f"atomic_spectra_{element}_{int(time.time())}.txt"
            filepath = self.data_dir / 'atomic_spectra' / filename
            filepath.parent.mkdir(exist_ok=True)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)

            data['local_file'] = str(filepath)

            return data

        except Exception as e:
            logger.error(f"Error retrieving atomic spectra for {element}: {e}")
            return {'element': element, 'error': str(e), 'timestamp': time.time()}

    def get_fluid_properties(self, compound: str, property_type: str = 'density') -> Dict[str, Any]:
        """
        Retrieve fluid properties data

        Args:
            compound: Chemical compound name
            property_type: Property to retrieve ('density', 'viscosity', etc.)

        Returns:
            Dictionary containing fluid property data
        """
        logger.info(f"Retrieving {property_type} data for {compound}")

        try:
            # NIST Webbook Fluid Properties
            base_url = "https://webbook.nist.gov/cgi/fluid.cgi"

            params = {
                'Action': 'Data',
                'Wide': 'on',
                'ID': compound,
                'Type': property_type,
                'Digits': '5',
                'PLow': '0.01',
                'PHigh': '10',
                'PInc': '1',
                'TLow': '200',
                'THigh': '400',
                'TInc': '50',
                'RefState': 'DEF'
            }

            response = self.session.get(base_url, params=params, timeout=30)
            response.raise_for_status()

            data = {
                'compound': compound,
                'property': property_type,
                'data_type': 'fluid_properties',
                'content': response.text,
                'timestamp': time.time()
            }

            # Save fluid data
            filename = f"fluid_properties_{compound}_{property_type}_{int(time.time())}.html"
            filepath = self.data_dir / 'fluid_properties' / filename
            filepath.parent.mkdir(exist_ok=True)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)

            data['local_file'] = str(filepath)

            return data

        except Exception as e:
            logger.error(f"Error retrieving fluid properties for {compound}: {e}")
            return {'compound': compound, 'property': property_type, 'error': str(e), 'timestamp': time.time()}

    def create_aerogel_relevant_datasets(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create datasets specifically relevant to aerogel research

        Returns:
            Dictionary of categorized datasets
        """
        logger.info("Creating aerogel-relevant datasets")

        datasets = {
            'silica_precursors': [],
            'solvent_properties': [],
            'thermal_properties': [],
            'surface_chemistry': [],
            'mechanical_properties': []
        }

        # Silica precursors for aerogels
        silica_compounds = ['SiO2', 'Si(OH)4', 'Si(OC2H5)4', 'Si(OCH3)4']
        for compound in silica_compounds:
            data = self.search_chemistry_webbook(compound, 'thermo')
            datasets['silica_precursors'].append(data)
            time.sleep(1)  # Rate limiting

        # Solvent properties (important for aerogel processing)
        solvents = ['ethanol', 'methanol', 'acetone', 'CO2']
        for solvent in solvents:
            data = self.get_fluid_properties(solvent, 'density')
            datasets['solvent_properties'].append(data)
            time.sleep(1)

        # Thermal properties materials
        thermal_materials = ['SiO2', 'Al2O3', 'TiO2']
        for material in thermal_materials:
            data = self.search_chemistry_webbook(material, 'thermo')
            datasets['thermal_properties'].append(data)
            time.sleep(1)

        # Save consolidated dataset
        filename = f"aerogel_datasets_{int(time.time())}.json"
        filepath = self.data_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(datasets, f, indent=2)

        logger.info(f"Saved aerogel datasets to {filepath}")
        return datasets

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report of ingested data

        Returns:
            Dictionary containing ingestion statistics and summary
        """
        report = {
            'ingestion_timestamp': time.time(),
            'databases_ingested': list(self.databases.keys()),
            'data_directory': str(self.data_dir),
            'total_files_ingested': 0,
            'data_categories': {},
            'errors': []
        }

        # Count files in each subdirectory
        for subdir in self.data_dir.iterdir():
            if subdir.is_dir():
                file_count = len(list(subdir.glob('*')))
                report['data_categories'][subdir.name] = file_count
                report['total_files_ingested'] += file_count

        # Save report
        report_file = self.data_dir / f"ingestion_report_{int(time.time())}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        return report


def main():
    """Main ingestion function"""
    logger.info("Starting NIST Data Ingestion for QuLab Infinite")

    ingester = NISTDataIngester()

    # Create aerogel-relevant datasets
    logger.info("Creating aerogel-relevant datasets...")
    datasets = ingester.create_aerogel_relevant_datasets()

    # Generate report
    report = ingester.generate_report()

    logger.info("NIST Data Ingestion Complete")
    logger.info(f"Total files ingested: {report['total_files_ingested']}")
    logger.info(f"Data categories: {report['data_categories']}")

    # Print summary
    print("\n" + "="*50)
    print("NIST DATA INGESTION SUMMARY")
    print("="*50)
    print(f"Total files ingested: {report['total_files_ingested']}")
    print(f"Data directory: {report['data_directory']}")
    print("\nData Categories:")
    for category, count in report['data_categories'].items():
        print(f"  {category}: {count} files")

    print("\nAerogel-relevant datasets created:")
    for category, items in datasets.items():
        print(f"  {category}: {len(items)} items")

    print("\nIngestion complete! Data saved to nist_data/ directory")


if __name__ == "__main__":
    main()
