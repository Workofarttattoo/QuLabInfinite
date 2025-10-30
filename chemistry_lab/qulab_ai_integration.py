"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

QuLab AI Integration for Chemistry Lab
Adds SMILES/SELFIES parsing with provenance tracking
"""
from qulab_ai.parsers.smiles_selfies import parse_molecule
from qulab_ai.answer_mode import build_answer
from qulab_ai.provenance import citation_block

def analyze_molecule_with_provenance(smiles: str, citations: list = None) -> dict:
    """
    Parse molecule and return with full provenance tracking

    Args:
        smiles: SMILES notation string
        citations: Optional list of citation dicts

    Returns:
        Dict with parsed molecule data, provenance, and citations
    """
    # Parse the molecule
    mol_data = parse_molecule(smiles)

    # Add default citation if none provided
    if citations is None:
        citations = [citation_block(
            source="QuLab AI Scaffold v0.1",
            system="QuLabInfinite Chemistry Lab"
        )]

    # Wrap with provenance
    return build_answer(
        payload=mol_data,
        citations=citations,
        units_ok=True
    )

def batch_analyze_molecules(smiles_list: list) -> list:
    """
    Analyze multiple molecules in batch

    Args:
        smiles_list: List of SMILES strings

    Returns:
        List of analysis results with provenance
    """
    results = []
    for smiles in smiles_list:
        try:
            result = analyze_molecule_with_provenance(smiles)
            results.append({
                "smiles": smiles,
                "success": True,
                "data": result
            })
        except Exception as e:
            results.append({
                "smiles": smiles,
                "success": False,
                "error": str(e)
            })
    return results

def validate_smiles(smiles: str) -> dict:
    """
    Validate SMILES notation

    Args:
        smiles: SMILES string to validate

    Returns:
        Dict with validation result
    """
    try:
        result = parse_molecule(smiles)
        return {
            "valid": True,
            "canonical": result["canonical_smiles"],
            "n_atoms": result["n_atoms"],
            "n_bonds": result["n_bonds"]
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }

# Example usage
if __name__ == "__main__":
    # Test with ethanol
    result = analyze_molecule_with_provenance("CCO")
    print(f"Molecule: {result['result']['canonical_smiles']}")
    print(f"Atoms: {result['result']['n_atoms']}")
    print(f"Digest: {result['digest']}")
