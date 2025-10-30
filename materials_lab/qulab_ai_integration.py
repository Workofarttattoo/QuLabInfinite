"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

QuLab AI Integration for Materials Lab
Adds CIF/POSCAR parsing with provenance tracking
"""
from qulab_ai.parsers.structures import parse_structure
from qulab_ai.answer_mode import build_answer
from qulab_ai.provenance import citation_block, sha256_file
from pathlib import Path

def analyze_structure_with_provenance(file_path: str, citations: list = None) -> dict:
    """
    Parse crystal structure file and return with provenance

    Args:
        file_path: Path to CIF or POSCAR file
        citations: Optional list of citation dicts

    Returns:
        Dict with parsed structure, provenance, and citations
    """
    # Parse the structure
    structure = parse_structure(file_path)

    # Calculate file hash for provenance
    file_hash = sha256_file(file_path)

    # Add file metadata to structure
    structure["file_metadata"] = {
        "path": str(Path(file_path).name),
        "sha256": file_hash
    }

    # Add default citation if none provided
    if citations is None:
        citations = [citation_block(
            source="QuLab AI Scaffold v0.1",
            system="QuLabInfinite Materials Lab",
            file_hash=file_hash
        )]

    # Wrap with provenance
    return build_answer(
        payload=structure,
        citations=citations,
        units_ok=True
    )

def batch_analyze_structures(file_paths: list) -> list:
    """
    Analyze multiple structure files in batch

    Args:
        file_paths: List of paths to structure files

    Returns:
        List of analysis results with provenance
    """
    results = []
    for file_path in file_paths:
        try:
            result = analyze_structure_with_provenance(file_path)
            results.append({
                "file": file_path,
                "success": True,
                "data": result
            })
        except Exception as e:
            results.append({
                "file": file_path,
                "success": False,
                "error": str(e)
            })
    return results

def validate_structure_file(file_path: str) -> dict:
    """
    Validate structure file format

    Args:
        file_path: Path to structure file

    Returns:
        Dict with validation result
    """
    try:
        result = parse_structure(file_path)
        return {
            "valid": True,
            "file_type": Path(file_path).suffix,
            "has_structure": "structure" in result
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }

def get_materials_database_info() -> dict:
    """
    Get info about integrated materials database

    Returns:
        Dict with database statistics
    """
    from materials_lab.core import MaterialsDatabase

    db = MaterialsDatabase()
    return {
        "total_materials": db.count(),
        "qulab_ai_integrated": True,
        "provenance_tracking": True,
        "parsers_available": ["CIF", "POSCAR"]
    }

# Example usage
if __name__ == "__main__":
    import sys

    # Test with a structure file if provided
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        result = analyze_structure_with_provenance(file_path)
        print(f"Structure parsed: {result['result'].keys()}")
        print(f"Digest: {result['digest']}")
    else:
        print("Usage: python qulab_ai_integration.py <structure_file>")
        print("\nDatabase info:")
        try:
            info = get_materials_database_info()
            print(f"  Total materials: {info['total_materials']}")
            print(f"  QuLab AI integrated: {info['qulab_ai_integrated']}")
        except:
            print("  Materials database not available")
