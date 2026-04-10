
"""
CIF/POSCAR (VASP)/PDB/XYZ structural file parsing with pymatgen/ase fallbacks.
"""
from typing import Dict, Any

def parse_cif(text: str) -> Dict[str, Any]:
    try:
        from pymatgen.core import Structure
        s = Structure.from_str(text, fmt="cif")
        return {"format": "cif", "n_sites": len(s), "formula": s.composition.reduced_formula, "lattice": s.lattice.matrix.tolist()}
    except Exception as e:
        return {"format": "cif", "error": str(e)}

def parse_poscar(text: str) -> Dict[str, Any]:
    try:
        from pymatgen.core import Structure
        s = Structure.from_str(text, fmt="poscar")
        return {"format": "poscar", "n_sites": len(s), "formula": s.composition.reduced_formula, "lattice": s.lattice.matrix.tolist()}
    except Exception as e:
        return {"format": "poscar", "error": str(e)}

def parse_xyz(text: str) -> Dict[str, Any]:
    try:
        from ase.io import read
        import io
        a = read(io.StringIO(text), format="xyz")
        return {"format": "xyz", "n_atoms": len(a), "symbols": a.get_chemical_symbols()[:10]}
    except Exception as e:
        return {"format": "xyz", "error": str(e)}

def parse_pdb(text: str) -> Dict[str, Any]:
    try:
        from Bio.PDB import PDBParser
        import io
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("X", io.StringIO(text))
        n_atoms = sum(1 for _ in structure.get_atoms())
        return {"format": "pdb", "n_atoms": n_atoms}
    except Exception as e:
        return {"format": "pdb", "error": str(e)}

def parse_structure(file_path: str) -> Dict[str, Any]:
    """
    Parse structure file - detects format from extension

    Args:
        file_path: Path to structure file

    Returns:
        Dict with parsed structure data
    """
    with open(file_path, 'r') as f:
        text = f.read()

    # Detect format from extension
    if file_path.endswith('.cif'):
        return parse_cif(text)
    elif file_path.endswith(('.vasp', '.poscar', 'POSCAR')):
        return parse_poscar(text)
    elif file_path.endswith('.xyz'):
        return parse_xyz(text)
    elif file_path.endswith('.pdb'):
        return parse_pdb(text)
    else:
        return {"error": "Unknown format", "file": file_path}
