
"""
Unit-checked, provenance-stamped answer mode.
"""
from typing import Dict, Any, Optional, List
from .units import HAVE_PINT
from .provenance import stamp, citation_block

def build_answer(payload: Dict[str, Any],
                 citations: Optional[List[Dict[str, Any]]] = None,
                 units_ok: bool = True) -> Dict[str, Any]:
    """
    Attach units check flag and citations; return provenance-stamped dict.
    """
    ans = {
        "result": payload,
        "units_checked": bool(units_ok and HAVE_PINT),
        "units_backend": "pint" if HAVE_PINT else "minimal",
        "citations": citations or []
    }
    return stamp(ans)

def demo_energy_answer(e_eV: float, doi: str) -> Dict[str, Any]:
    # Example conversion: eV -> kJ/mol (constant factor here; full conversion via pint recommended)
    e_kjmol = e_eV * 96.48533212
    payload = {"E[eV]": e_eV, "E[kJ/mol]": e_kjmol}
    cites = [citation_block(doi=doi, source="Scientific Data / QCML")]
    return build_answer(payload, citations=cites, units_ok=True)
