"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

QuLab AI Integration for Frequency Lab
Adds JCAMP-DX parsing and spectra encoding with provenance
"""
from qulab_ai.parsers.jcamp import parse_spectrum
from qulab_ai.answer_mode import build_answer
from qulab_ai.provenance import citation_block, sha256_file
from pathlib import Path
from spectra_xrd_encoder_sprint3 import encode_curve, encode_text, contrastive_score

def analyze_spectrum_with_encoding(jcamp_path: str, caption: str = "", citations: list = None) -> dict:
    """
    Parse JCAMP-DX spectrum and encode for ML

    Args:
        jcamp_path: Path to JCAMP-DX file
        caption: Optional text caption for contrastive learning
        citations: Optional list of citation dicts

    Returns:
        Dict with spectrum data, encoding, and provenance
    """
    # Parse spectrum
    spectrum = parse_spectrum(jcamp_path)

    # Encode spectrum for ML
    encoding = encode_curve(spectrum['x'], spectrum['y'])

    # Calculate alignment score if caption provided
    alignment_score = None
    if caption:
        text_encoding = encode_text(caption)
        alignment_score = contrastive_score(encoding, text_encoding)

    # Calculate file hash for provenance
    file_hash = sha256_file(jcamp_path)

    # Build result
    result = {
        "spectrum": spectrum,
        "ml_encoding": {
            "peaks": float(encoding[0]),
            "centroid": float(encoding[1]),
            "variance": float(encoding[2]),
            "roughness": float(encoding[3])
        },
        "file_metadata": {
            "path": str(Path(jcamp_path).name),
            "sha256": file_hash
        }
    }

    if alignment_score is not None:
        result["alignment"] = {
            "caption": caption,
            "score": float(alignment_score)
        }

    # Add default citation if none provided
    if citations is None:
        citations = [citation_block(
            source="QuLab AI Scaffold v0.1",
            system="QuLabInfinite Frequency Lab",
            file_hash=file_hash
        )]

    # Wrap with provenance
    return build_answer(
        payload=result,
        citations=citations,
        units_ok=True
    )

def encode_spectrum_array(x: list, y: list, caption: str = "") -> dict:
    """
    Encode spectrum from arrays (without file)

    Args:
        x: X-axis values (frequencies, wavelengths, etc.)
        y: Y-axis values (intensity, transmittance, etc.)
        caption: Optional text caption

    Returns:
        Dict with encoding and metadata
    """
    encoding = encode_curve(x, y)

    result = {
        "ml_encoding": {
            "peaks": float(encoding[0]),
            "centroid": float(encoding[1]),
            "variance": float(encoding[2]),
            "roughness": float(encoding[3])
        },
        "data_points": len(x)
    }

    if caption:
        text_encoding = encode_text(caption)
        alignment_score = contrastive_score(encoding, text_encoding)
        result["alignment"] = {
            "caption": caption,
            "score": float(alignment_score)
        }

    return result

def batch_encode_spectra(spectra_data: list) -> list:
    """
    Encode multiple spectra in batch

    Args:
        spectra_data: List of dicts with 'x', 'y', and optional 'caption'

    Returns:
        List of encoding results
    """
    results = []
    for data in spectra_data:
        try:
            encoding = encode_spectrum_array(
                data['x'],
                data['y'],
                data.get('caption', '')
            )
            results.append({
                "success": True,
                "encoding": encoding
            })
        except Exception as e:
            results.append({
                "success": False,
                "error": str(e)
            })
    return results

def search_similar_spectra(query_encoding: list, database_encodings: list, top_k: int = 5) -> list:
    """
    Search for similar spectra using contrastive similarity

    Args:
        query_encoding: Encoding of query spectrum
        database_encodings: List of (id, encoding) tuples from database
        top_k: Number of top results to return

    Returns:
        List of (id, similarity_score) tuples, sorted by score
    """
    similarities = []
    for spec_id, encoding in database_encodings:
        score = contrastive_score(query_encoding, encoding)
        similarities.append((spec_id, float(score)))

    # Sort by score descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Analyze JCAMP file
        jcamp_path = sys.argv[1]
        caption = sys.argv[2] if len(sys.argv) > 2 else "test spectrum"
        result = analyze_spectrum_with_encoding(jcamp_path, caption)
        print(f"Spectrum analyzed:")
        print(f"  Peaks: {result['result']['ml_encoding']['peaks']}")
        print(f"  Centroid: {result['result']['ml_encoding']['centroid']:.2f}")
        print(f"  Digest: {result['digest']}")
        if 'alignment' in result['result']:
            print(f"  Alignment score: {result['result']['alignment']['score']:.4f}")
    else:
        # Demo with synthetic data
        print("Demo: Encoding synthetic XRD pattern")
        x = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        y = [0.1, 0.8, 0.3, 1.0, 0.2, 0.6, 0.1, 0.4, 0.1]
        result = encode_spectrum_array(x, y, "crystalline silicon diffraction")
        print(f"  Peaks: {result['ml_encoding']['peaks']}")
        print(f"  Centroid: {result['ml_encoding']['centroid']:.2f}°")
        print(f"  Alignment: {result['alignment']['score']:.4f}")
