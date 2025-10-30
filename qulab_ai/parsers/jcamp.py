
"""
JCAMP-DX parser for spectra (IR/Raman/NMR etc.). Uses `jcamp` if available, otherwise minimal fallback.
"""
from typing import Dict, Any

def parse_jcamp(text: str) -> Dict[str, Any]:
    try:
        import jcamp
        data = jcamp.JCAMP_reader(text)
        x = data.get("x", [])
        y = data.get("y", [])
        meta = {k: v for k, v in data.items() if k not in ("x", "y")}
        return {"format": "jcamp-dx", "points": len(x), "x": x[:1024], "y": y[:1024], "meta": meta}
    except Exception:
        # Very small fallback that extracts XY pairs from simple CSV-like JCAMP data blocks
        lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("##")]
        xs, ys = [], []
        for ln in lines:
            parts = ln.replace(",", " ").split()
            if len(parts) == 2:
                try:
                    xs.append(float(parts[0])); ys.append(float(parts[1]))
                except Exception:
                    pass
        return {"format": "jcamp-dx", "points": len(xs), "x": xs[:1024], "y": ys[:1024], "meta": {"note": "fallback-parser"}}

def parse_spectrum(file_path: str) -> Dict[str, Any]:
    """
    Parse JCAMP-DX spectrum file

    Args:
        file_path: Path to JCAMP-DX file

    Returns:
        Dict with spectrum data (x, y arrays and metadata)
    """
    with open(file_path, 'r') as f:
        text = f.read()
    return parse_jcamp(text)
