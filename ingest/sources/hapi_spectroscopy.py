from __future__ import annotations
from typing import Iterable
from ..schemas import RecordChem, Provenance
import hapi as hapi
import h5py
import pathlib


def load_live(molecule_name: str, min_wavenumber: float, max_wavenumber: float, h5_dir: str = "data/h5") -> Iterable[RecordChem]:
    """
    Load spectroscopic data for a molecule from the HITRAN database using the HAPI library.
    The data is stored in an HDF5 file and a single record is yielded.
    """
    prov = Provenance(
        source="HITRAN",
        url="https://hitran.org/",
        license="HITRAN-LICENSE",
        notes=f"Spectroscopic data for {molecule_name} from {min_wavenumber} to {max_wavenumber} cm-1."
    )

    # Convert molecule name to HITRAN molecule number
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

    # Define table name for fetching data
    table_name = f"{molecule_name}_{min_wavenumber}-{max_wavenumber}"
    
    # Fetch data from HITRAN
    hapi.fetch(table_name, molecule_number, 1, min_wavenumber, max_wavenumber)

    # Get column data
    columns_to_fetch = ['nu', 'sw', 'a', 'gamma_air', 'gamma_self', 'elower', 'n_air', 'delta_air']
    
    h5_path = pathlib.Path(h5_dir) / f"{table_name}.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, 'w') as hf:
        for col_name in columns_to_fetch:
            try:
                data = hapi.getColumn(table_name, col_name)
                hf.create_dataset(col_name, data=data)
            except Exception as e:
                print(f"Could not fetch column {col_name} for {table_name}: {e}")
    
    yield RecordChem(
        substance=molecule_name,
        phase="gas",
        pressure_pa=101325.0,  # Standard pressure for HITRAN
        temperature_k=296.0,  # Standard temperature for HITRAN
        tags=[
            f"wavenumber_range:{min_wavenumber}-{max_wavenumber}",
        ],
        provenance=prov,
        spectrum_hdf5_ref=str(h5_path),
    )
