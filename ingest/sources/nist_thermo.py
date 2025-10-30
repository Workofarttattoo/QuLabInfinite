from __future__ import annotations
from typing import Iterable
from ..schemas import RecordChem, Provenance
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def load_live(substance_cas_id: str = "7732-18-5", substance_name: str = "H2O") -> Iterable[RecordChem]:
    """
    Load thermodynamic data for a substance from the NIST Chemistry WebBook.
    This function scrapes the 'Gas phase thermochemistry data' page.
    """
    base_url = "https://webbook.nist.gov"
    url = f"{base_url}/cgi/cbook.cgi?ID={substance_cas_id}&Mask=1"
    
    prov = Provenance(
        source="NIST Chemistry WebBook",
        url=url,
        license="PUBLIC-DOMAIN",
        notes="Scraped from gas phase thermochemistry data page."
    )
    
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the h2 tag with the specified text
    h2_tag = soup.find('h2', text='Gas phase thermochemistry data')
    if not h2_tag:
        raise ValueError("Could not find the 'Gas phase thermochemistry data' section.")

    # Find the table that immediately follows the h2 tag.
    table = h2_tag.find_next_sibling('table')
    
    if not table:
        raise ValueError("Could not find the data table after the 'Gas phase thermochemistry data' section.")
        
    # This table has multiple tbody elements, so we need to iterate through all of them.
    for row in table.find_all('tr'):
        cols = row.find_all(['th', 'td'])
        if len(cols) < 2:
            continue
        
        quantity = cols[0].text.strip()
        value = cols[1].text.strip()
        
        # We are interested in enthalpy of formation and entropy.
        # Note: This is a simplified parser for the key-value table.
        # A more robust parser would be needed for all data types.
        
        try:
            if 'ΔfH°' in quantity:
                # Value is in format '-241.826 ± 0.040'. We'll take the first part.
                enthalpy_kj_per_mol = float(value.split(' ')[0])
                yield RecordChem(
                    substance=substance_name,
                    phase="gas",
                    enthalpy_j_per_mol=enthalpy_kj_per_mol * 1000,
                    provenance=prov,
                    # These fields are required but not all are present in this table
                    pressure_pa=100000, # Standard pressure
                    temperature_k=298.15, # Standard temperature
                    tags=["enthalpy_of_formation"]
                )
            elif 'S°' in quantity:
                entropy_j_per_mol_k = float(value.split(' ')[0])
                yield RecordChem(
                    substance=substance_name,
                    phase="gas",
                    entropy_j_per_mol_k=entropy_j_per_mol_k,
                    provenance=prov,
                    pressure_pa=100000,
                    temperature_k=298.15,
                    tags=["standard_entropy"]
                )

        except (ValueError, IndexError):
            continue
