from __future__ import annotations
from typing import Iterable, Optional, Tuple
from ..schemas import RecordChem, Provenance
import requests
from bs4 import BeautifulSoup
import argparse
from .base import DataSource

def _parse_value_with_uncertainty(value_str: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parses a string like '-241.826 ± 0.040' into a value and uncertainty.
    Returns (value, uncertainty). Uncertainty is None if not present.
    """
    if '±' in value_str:
        parts = value_str.split('±')
        try:
            value = float(parts[0].strip())
            uncertainty = float(parts[1].strip().split(' ')[0])
            return value, uncertainty
        except (ValueError, IndexError):
            return None, None
    else:
        try:
            value = float(value_str.strip().split(' ')[0])
            return value, None
        except (ValueError, IndexError):
            return None, None

class NISTThermoSource(DataSource):
    name = "nist_thermo"
    description = "Loads thermodynamic data from the NIST Chemistry WebBook."

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument("--nist-cas-id", required=True, help="CAS ID for NIST thermo source (e.g., 7732-18-5 for water)")
        parser.add_argument("--nist-substance-name", required=True, help="Substance name for NIST thermo source (e.g., H2O)")

    def load(self, args: argparse.Namespace) -> Iterable[RecordChem]:
        base_url = "https://webbook.nist.gov"
        url = f"{base_url}/cgi/cbook.cgi?ID={args.nist_cas_id}&Mask=1"
        
        prov = Provenance(
            source="NIST Chemistry WebBook",
            url=url,
            license="PUBLIC-DOMAIN",
            notes="Scraped from gas phase thermochemistry data page."
        )
        
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        h2_tag = soup.find('h2', text='Gas phase thermochemistry data')
        if not h2_tag:
            h2_tag = soup.find('h2', id='Thermo-Gas')
        
        if not h2_tag:
            raise ValueError("Could not find the 'Gas phase thermochemistry data' section.")

        table = h2_tag.find_next_sibling('table')
        
        if not table:
            raise ValueError("Could not find the data table after the 'Gas phase thermochemistry data' section.")
            
        for row in table.find_all('tr'):
            cols = row.find_all(['th', 'td'])
            if len(cols) < 2:
                continue
            
            quantity = cols[0].text.strip()
            value = cols[1].text.strip()
            
            try:
                value_num, uncertainty = _parse_value_with_uncertainty(value)
                if value_num is None:
                    continue

                record_data = {
                    "substance": args.nist_substance_name,
                    "phase": "gas",
                    "provenance": prov.copy(deep=True),
                    "pressure_pa": 100000,
                    "temperature_k": 298.15,
                }

                if uncertainty is not None:
                    record_data["provenance"].extra['uncertainty'] = uncertainty

                if 'ΔfH°' in quantity:
                    record_data["enthalpy_j_per_mol"] = value_num * 1000
                    record_data["tags"] = ["enthalpy_of_formation"]
                    yield RecordChem(**record_data)

                elif 'S°' in quantity:
                    record_data["entropy_j_per_mol_k"] = value_num
                    record_data["tags"] = ["standard_entropy"]
                    yield RecordChem(**record_data)

            except (ValueError, IndexError):
                continue

        h3_tag = soup.find('h3', text='Gas Phase Heat Capacity (Shomate Equation)')
        if h3_tag:
            shomate_table = h3_tag.find_next_sibling('table')
            if shomate_table:
                shomate_coeffs = {}
                rows = shomate_table.find_all('tr')
                if len(rows) > 1:
                    temp_ranges = [th.text.strip() for th in rows[0].find_all('td')]
                    
                    for i, temp_range in enumerate(temp_ranges):
                        shomate_coeffs[temp_range] = {}
                        for row in rows[1:]:
                            cols = row.find_all(['th', 'td'])
                            if len(cols) > i + 1:
                                coeff_name = cols[0].text.strip()
                                try:
                                    coeff_value = float(cols[i+1].text.strip())
                                    shomate_coeffs[temp_range][coeff_name] = coeff_value
                                except (ValueError, IndexError):
                                    continue
                if shomate_coeffs:
                    prov.extra['shomate_coeffs'] = shomate_coeffs
