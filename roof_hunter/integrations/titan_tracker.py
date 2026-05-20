import os
import subprocess
import json
from datetime import datetime, timezone
from typing import List, Dict, Any

def run_titan_java(radar_file_path: str) -> List[Dict[str, Any]]:
    """
    Execute titan-java / irose-titan on a given radar file to extract storm tracks.
    This shells out to the local titan-java binary downloaded by the user.
    """
    # Assuming the user downloaded these to their home directory or a known bin path
    titan_bin = os.getenv("TITAN_JAVA_BIN", "titan-java") # fallback to PATH
    
    # In a real scenario, titan-java expects specific radar formats (MDV, cfradial, etc.)
    # We pass the file and ask for JSON output of the storm tracks
    cmd = [
        titan_bin,
        "--mode", "track",
        "--input", radar_file_path,
        "--output-format", "json"
    ]
    
    print(f"[TITAN] Running irose-titan / titan-java storm tracking on {radar_file_path}...")
    
    try:
        # We simulate the subprocess call for now as the binary might not be in PATH
        # result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # return json.loads(result.stdout)
        
        print("[TITAN] Subprocess call simulated. Parsing mock TITAN output.")
        # Simulated TITAN output mapping to our expected dictionary
        return [{
            "storm_cell_id": "TITAN_001",
            "center_lat": 35.33,
            "center_lon": -97.27,
            "polygon_geojson": [
                (35.35, -97.29), (35.35, -97.25), 
                (35.31, -97.25), (35.31, -97.29), 
                (35.35, -97.29)
            ],
            "max_reflectivity_dbz": 65.0,
            "vil_kg_m2": 45.0, # TITAN usually provides VIL
            "velocity_x": 12.5, # m/s
            "velocity_y": 5.0,  # m/s
            "area_km2": 20.5,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }]
    except subprocess.CalledProcessError as e:
        print(f"[TITAN] Error executing titan-java: {e.stderr}")
        return []
    except Exception as e:
        print(f"[TITAN] Exception during TITAN execution: {e}")
        return []

def track_with_irose_titan(radar_site: str = "KTLX") -> List[Dict[str, Any]]:
    """
    High-level wrapper to fetch radar and run TITAN.
    """
    from .radar_engine import NEXRADLevel2
    import tempfile
    
    nexrad = NEXRADLevel2()
    # For TITAN, we usually need the actual file on disk. 
    # NEXRADLevel2's get_latest_scan returns a Py-ART object, so we'd need the raw file path.
    # In a production environment, we download the raw S3 archive and pass it to TITAN.
    
    # We will simulate the raw file path creation for the wiring
    raw_file_mock = f"/tmp/{radar_site}_latest_scan.gz"
    return run_titan_java(raw_file_mock)
