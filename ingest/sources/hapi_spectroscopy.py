from __future__ import annotations
from typing import Iterable
from ..schemas import RecordChem, Provenance
import hapi as hapi
# from astroquery.hitran import Hitran
# from astropy import units as u

def load_live(molecule_name: str, min_wavenumber: float, max_wavenumber: float) -> Iterable[RecordChem]:
    """
    Load spectroscopic data for a molecule from the HITRAN database using the HAPI library.
    """
    prov = Provenance(
        source="HITRAN",
        url="https://hitran.org/",
        license="HITRAN-LICENSE",
        notes=f"Spectroscopic data for {molecule_name} from {min_wavenumber} to {max_wavenumber} cm-1."
    )

    # Convert molecule name to HITRAN molecule number
    molecule_map = {
        "H2O": 1, "CO2": 2, "O3": 3, "N2O": 4, "CO": 5, "CH4": 6,
    }
    molecule_number = molecule_map.get(molecule_name)
    if not molecule_number:
        raise ValueError(f"Unknown molecule: {molecule_name}")

    # Define table name for fetching data
    table_name = f"{molecule_name}_{min_wavenumber}-{max_wavenumber}"
    
    # Fetch data from HITRAN
    hapi.fetch(table_name, molecule_number, 1, min_wavenumber, max_wavenumber)

    # Get column data
    nu = hapi.getColumn(table_name, 'nu') # cm-1
    sw = hapi.getColumn(table_name, 'sw') # cm-1/(molecule.cm-2)
    a = hapi.getColumn(table_name, 'a') # s-1
    gamma_air = hapi.getColumn(table_name, 'gamma_air') # cm-1/atm
    gamma_self = hapi.getColumn(table_name, 'gamma_self') # cm-1/atm
    elower = hapi.getColumn(table_name, 'elower') # cm-1
    n_air = hapi.getColumn(table_name, 'n_air') # dimensionless
    delta_air = hapi.getColumn(table_name, 'delta_air') # cm-1/atm

    # Convert the HAPI table to RecordChem objects
    for i in range(len(nu)):
        yield RecordChem(
            substance=molecule_name,
            phase="gas",
            pressure_pa=101325.0,  # Standard pressure for HITRAN
            temperature_k=296.0,  # Standard temperature for HITRAN
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
