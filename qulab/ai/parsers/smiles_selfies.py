
"""
SMILES/SELFIES utilities. Prefers RDKit and selfies when available.
"""
from typing import Optional, Dict, Any

def _rdkit_mol_from_smiles(smiles: str):
    try:
        from rdkit import Chem
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None

def _selfies_to_smiles(sf: str) -> Optional[str]:
    try:
        import selfies as sfmod
        return sfmod.decoder(sf)
    except Exception:
        return None

def parse_molecule(text: str) -> Dict[str, Any]:
    """
    Accepts SMILES or SELFIES; returns a dict with canonical_smiles and a few basic stats.
    """
    # Try SMILES first
    mol = _rdkit_mol_from_smiles(text)
    smiles = None
    if mol is not None:
        try:
            from rdkit.Chem import MolToSmiles
            smiles = MolToSmiles(mol, canonical=True)
        except Exception:
            smiles = text  # fallback
    else:
        # Try SELFIES decode -> SMILES
        smiles = _selfies_to_smiles(text)

    result = {"input": text, "canonical_smiles": smiles, "n_atoms": None, "n_bonds": None}
    if smiles and _rdkit_mol_from_smiles(smiles) is not None:
        try:
            from rdkit import Chem
            mol2 = Chem.MolFromSmiles(smiles)
            result["n_atoms"] = mol2.GetNumAtoms()
            result["n_bonds"] = mol2.GetNumBonds()
        except Exception:
            pass
    return result
