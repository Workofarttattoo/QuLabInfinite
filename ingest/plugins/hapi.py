from __future__ import annotations
from typing import Iterable
from ..schemas import RecordChem, Provenance
import hapi as hapi
import argparse
from .base import DataSource

class HAPISource(DataSource):
    name = "hapi"
    description = "Loads spectroscopic data from the HITRAN database using the HAPI library."

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument("--hapi-molecule-name", required=True, help="Molecule name for HAPI source (e.g., CO2)")
        parser.add_argument("--hapi-min-wavenumber", type=float, required=True, help="Minimum wavenumber in cm-1 for HAPI source")
        parser.add_argument("--hapi-max-wavenumber", type=float, required=True, help="Maximum wavenumber in cm-1 for HAPI source")

    def load(self, args: argparse.Namespace) -> Iterable[RecordChem]:
        molecule_name = args.hapi_molecule_name
        min_wavenumber = args.hapi_min_wavenumber
        max_wavenumber = args.hapi_max_wavenumber

        prov = Provenance(
            source="HITRAN",
            url="https://hitran.org/",
            license="HITRAN-LICENSE",
            notes=f"Spectroscopic data for {molecule_name} from {min_wavenumber} to {max_wavenumber} cm-1."
        )

        molecule_map = {
            "H2O": 1, "CO2": 2, "O3": 3, "N2O": 4, "CO": 5, "CH4": 6, "O2": 7,
            "NO": 8, "SO2": 9, "NO2": 10, "NH3": 11, "HNO3": 12, "OH": 13,
            "HF": 14, "HCl": 15, "HBr": 16, "HI": 17, "ClO": 18, "OCS": 19,
            "H2CO": 20, "HOCl": 21, "N2": 22, "HCN": 23, "CH3Cl": 24, "H2O2": 25,
            "C2H2": 26, "C2H6": 27, "PH3": 28, "COF2": 29, "SF6": 30, "H2S": 31,
            "HCOOH": 32, "HO2": 33, "O": 34, "ClONO2": 35, "NO+": 36, "HOBr": 37,
            "C2H4": 38, "CH3OH": 39, "CH3Br": 40, "CH3CN": 41, "CF4": 42,
            "C4H2": 43, "HC3N": 44, "H2": 45, "CS": 46, "SO3": 47, "C2N2": 48,
            "COCl2": 49, "SO": 50, "CH3F": 51, "GeH4": 52, "CS2": 53, "CH3I": 54, "NF3": 55,
        }
        molecule_number = molecule_map.get(molecule_name)
        if not molecule_number:
            raise ValueError(f"Unknown molecule: {molecule_name}")

        table_name = f"{molecule_name}_{min_wavenumber}-{max_wavenumber}"
        
        hapi.fetch(table_name, molecule_number, 1, min_wavenumber, max_wavenumber)

        nu = hapi.getColumn(table_name, 'nu')
        sw = hapi.getColumn(table_name, 'sw')
        a = hapi.getColumn(table_name, 'a')
        gamma_air = hapi.getColumn(table_name, 'gamma_air')
        gamma_self = hapi.getColumn(table_name, 'gamma_self')
        elower = hapi.getColumn(table_name, 'elower')
        n_air = hapi.getColumn(table_name, 'n_air')
        delta_air = hapi.getColumn(table_name, 'delta_air')

        for i in range(len(nu)):
            yield RecordChem(
                substance=molecule_name,
                phase="gas",
                pressure_pa=101325.0,
                temperature_k=296.0,
                tags=[
                    f"spectral_line:{nu[i]}",
                    f"intensity:{sw[i]}",
                    f"einstein_A:{a[i]}",
                    f"gamma_air:{gamma_air[i]}",
                    f"gamma_self:{gamma_self[i]}",
                    f"elower:{elower[i]}",
                    f"n_air:{n_air[i]}",
                    f"delta_air:{delta_air[i]}",
                ],
                provenance=prov,
            )
