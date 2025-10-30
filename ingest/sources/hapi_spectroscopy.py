from __future__ import annotations
from typing import Iterable
from ..schemas import RecordChem, Provenance
from astroquery.hitran import Hitran
from astropy import units as u

def load_live(molecule_name: str, min_wavenumber: float, max_wavenumber: float) -> Iterable[RecordChem]:
    """
    Load spectroscopic data for a molecule from the HITRAN database using astroquery.
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

    table = Hitran.query_lines(
        molecule_number=molecule_number,
        isotopologue_number=1,
        min_frequency=min_wavenumber * u.cm**-1,
        max_frequency=max_wavenumber * u.cm**-1,
    )

    if not table:
        return

    # Convert the astroquery table to RecordChem objects
    for row in table:
        yield RecordChem(
            substance=molecule_name,
            phase="gas",
            pressure_pa=101325.0,  # Standard pressure for HITRAN
            temperature_k=296.0,  # Standard temperature for HITRAN
            tags=[
                f"spectral_line:{row['nu']}",
                f"intensity:{row['sw']}",
                f"einstein_A:{row['a']}",
                f"gamma_air:{row['gamma_air']}",
                f"gamma_self:{row['gamma_self']}",
                f"elower:{row['elower']}",
                f"n_air:{row['n_air']}",
                f"delta_air:{row['delta_air']}",
            ],
            provenance=prov,
        )
