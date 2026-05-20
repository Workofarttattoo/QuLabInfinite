import os
import tempfile
import boto3
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

try:
    import pyart
except ImportError:
    pyart = None

class NEXRADLevel2:
    """
    Fetch and process NOAA NEXRAD Level II Dual-Pol radar data.
    Supports:
    - Real-time data from NOAA's AWS S3 bucket
    - Historical data retrieval
    - Dual-Pol feature extraction (Z, ZDR, CC, KDP, Velocity, Spectrum Width)
    - Storm cell tracking
    """
    def __init__(self, s3_bucket: str = "noaa-nexrad-level2"):
        from botocore import UNSIGNED
        from botocore.config import Config
        self.s3_bucket = s3_bucket
        self.s3_client = boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED))
        self.radar_sites = self._load_radar_sites()

    def _load_radar_sites(self) -> Dict[str, Dict[str, Any]]:
        """Load NOAA radar site metadata."""
        # Radar site metadata (WSR-88D network)
        return {
            "KTLX": {"lat": 35.3325, "lon": -97.2781, "name": "Twin Lakes, OK"},
            "KFWS": {"lat": 32.5825, "lon": -96.8642, "name": "Fort Worth, TX"},
            "KICT": {"lat": 37.6553, "lon": -97.4492, "name": "Wichita, KS"},
            "KLBB": {"lat": 33.6556, "lon": -101.8156, "name": "Lubbock, TX"},
            # Add all 159 WSR-88D sites
        }

    def get_latest_scan(self, radar_site: str = "KTLX") -> Optional[Any]:
        """
        Get the latest radar scan for a given site.
        """
        if pyart is None:
            print("pyart is not installed. Cannot load radar scan.")
            return None
        # List files in the S3 bucket for the radar site
        prefix = f"{radar_site}/{datetime.utcnow().strftime('%Y/%m/%d')}"
        response = self.s3_client.list_objects_v2(
            Bucket=self.s3_bucket,
            Prefix=prefix
        )

        if "Contents" not in response:
            return None

        # Get the most recent file
        latest_file = max(response["Contents"], key=lambda x: x["LastModified"])
        file_key = latest_file["Key"]

        # Download the file
        with tempfile.NamedTemporaryFile(suffix=".gz", delete=False) as tmp_file:
            self.s3_client.download_file(self.s3_bucket, file_key, tmp_file.name)
            try:
                radar = pyart.io.read_nexrad_archive(tmp_file.name)
                os.remove(tmp_file.name)
                return radar
            except Exception as e:
                print(f"Error reading radar file {file_key}: {e}")
                os.remove(tmp_file.name)
                return None

    def get_historical_scans(
        self,
        radar_site: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Any]:
        scans = []
        if pyart is None: return scans
        current = start_time

        while current <= end_time:
            # Format: YYYY/MM/DD/HH/MM
            prefix = f"{radar_site}/{current.strftime('%Y/%m/%d/%H')}"
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=prefix
            )

            if "Contents" in response:
                for obj in response["Contents"]:
                    file_time = obj["LastModified"].replace(tzinfo=None)
                    if start_time <= file_time <= end_time:
                        with tempfile.NamedTemporaryFile(suffix=".gz", delete=False) as tmp_file:
                            self.s3_client.download_file(
                                self.s3_bucket,
                                obj["Key"],
                                tmp_file.name
                            )
                            try:
                                radar = pyart.io.read_nexrad_archive(tmp_file.name)
                                scans.append(radar)
                            except Exception as e:
                                print(f"Error reading radar file {obj['Key']}: {e}")
                            finally:
                                os.remove(tmp_file.name)

            current += timedelta(hours=1)

        return scans

    def extract_dual_pol_features(
        self,
        radar: Any,
        lat: float,
        lon: float,
        radius_km: float = 5.0
    ) -> Dict[str, Any]:
        """Extract Dual-Pol features at a given location."""
        # Convert lat/lon to radar coordinates
        radar_lat = radar.latitude["data"][0]
        radar_lon = radar.longitude["data"][0]

        # Calculate distance and azimuth from radar
        distance_km, azimuth = self._haversine(radar_lat, radar_lon, lat, lon)
        elevation = 0.0  # Default (adjust based on beam height)

        # Get gate index
        range_gates = radar.range["data"] / 1000  # Convert to km
        gate_idx = np.argmin(np.abs(range_gates - distance_km))

        if np.abs(range_gates[gate_idx] - distance_km) > 5:  # 5km tolerance
            return {}

        # Extract features from radar volume
        features = {
            "radar_site": radar.metadata["instrument_name"],
            "distance_to_radar_km": distance_km,
            "azimuth": azimuth,
            "elevation": elevation,
            "range_km": range_gates[gate_idx],
            "reflectivity_dbz": self._get_field_value(radar, "reflectivity", gate_idx),
            "zdr_db": self._get_field_value(radar, "differential_reflectivity", gate_idx),
            "cc": self._get_field_value(radar, "cross_correlation_ratio", gate_idx),
            "kdp_deg_per_km": self._get_field_value(radar, "specific_differential_phase", gate_idx),
            "velocity_mps": self._get_field_value(radar, "velocity", gate_idx),
            "spectrum_width_mps": self._get_field_value(radar, "spectrum_width", gate_idx),
            "timestamp": radar.time["units"]
        }

        # Calculate derived features
        features.update(self._calculate_derived_features(features))

        return features

    def _haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> tuple:
        """Calculate distance and bearing between two points."""
        from math import radians, sin, cos, sqrt, atan2

        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance_km = 6371.0 * c

        y = sin(dlon) * cos(lat2)
        x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
        bearing = atan2(y, x)

        return distance_km, np.degrees(bearing)

    def _get_field_value(self, radar: Any, field_name: str, gate_idx: int) -> float:
        """Get value from a radar field at a given gate."""
        if field_name not in radar.fields:
            return 0.0

        field = radar.fields[field_name]
        data = field["data"]

        # Get the first sweep (lowest elevation)
        if len(data.shape) == 3:
            data = data[0, :, :]
        elif len(data.shape) == 2:
            pass
        else:
            return 0.0

        # Get value at gate_idx (average over azimuth)
        if gate_idx < data.shape[1]:
            return float(np.nanmean(data[:, gate_idx]))
        else:
            return 0.0

    def _calculate_derived_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate derived features from Dual-Pol data."""
        derived = {}

        # Hail Contour Algorithm (HCA) - estimated hail size in inches
        if features.get("reflectivity_dbz", 0) >= 40:
            z = features["reflectivity_dbz"]
            zdr = features.get("zdr_db", 0)
            derived["hca_inches"] = max(0, (z - 40) * 0.025 + max(0, (zdr - 0.5) * 0.1))

        # Maximum Expected Size of Hail (MESH)
        if features.get("reflectivity_dbz", 0) >= 40:
            z = features["reflectivity_dbz"]
            derived["mesh_inches"] = max(0, (z - 40) * 0.03)

        # Hail Differential Reflectivity (HDR)
        if features.get("zdr_db", 0) >= 0.5:
            derived["hdr_db"] = features["zdr_db"] - 0.5

        # Vertical Integrated Liquid (VIL)
        if features.get("reflectivity_dbz", 0) >= 20:
            derived["vil_kg_m2"] = features["reflectivity_dbz"] * 3.44  # Approximate

        # Storm motion (simplified)
        if features.get("velocity_mps", 0) != 0:
            derived["storm_motion_speed_mps"] = abs(features["velocity_mps"])
            derived["storm_motion_direction_deg"] = 0  # Placeholder

        return derived

    def get_storm_cells(self, radar: Any) -> List[Dict[str, Any]]:
        """Identify and track storm cells in a radar volume."""
        storm_cells = []

        if "reflectivity" not in radar.fields:
            return storm_cells

        reflectivity = radar.fields["reflectivity"]["data"]
        if len(reflectivity.shape) == 3:
            reflectivity = reflectivity[0, :, :]  # First sweep

        from scipy.ndimage import label, maximum_filter
        from scipy.ndimage.morphology import generate_binary_structure

        threshold = 40
        binary = reflectivity > threshold
        structure = generate_binary_structure(2, 2)
        labeled, num_features = label(binary, structure=structure)

        neighborhood_size = 5
        local_max = (reflectivity == maximum_filter(reflectivity, size=neighborhood_size))

        for i in range(1, num_features + 1):
            indices = np.where(labeled == i)
            y_indices, x_indices = indices

            if len(x_indices) == 0:
                continue

            max_reflectivity = np.max(reflectivity[y_indices, x_indices])
            center_x = int(np.mean(x_indices))
            center_y = int(np.mean(y_indices))

            range_gates = radar.range["data"] / 1000  # km
            azimuths = radar.azimuth["data"]

            range_km = range_gates[center_x]
            azimuth_deg = azimuths[center_y]

            radar_lat = radar.latitude["data"][0]
            radar_lon = radar.longitude["data"][0]
            lat, lon = self._radar_to_latlon(radar_lat, radar_lon, range_km, azimuth_deg)

            features = self.extract_dual_pol_features(radar, lat, lon)

            # Dynamic polygon generation using Shapely
            try:
                from shapely.geometry import MultiPoint
                # Convert all points in the cell to lat/lon for the polygon
                cell_lats = []
                cell_lons = []
                for yi, xi in zip(y_indices, x_indices):
                    r_km = range_gates[xi]
                    az_deg = azimuths[yi]
                    clat, clon = self._radar_to_latlon(radar_lat, radar_lon, r_km, az_deg)
                    cell_lats.append(clat)
                    cell_lons.append(clon)
                
                # Create convex hull polygon around the cell
                if len(cell_lats) >= 3:
                    points = list(zip(cell_lons, cell_lats))
                    poly = MultiPoint(points).convex_hull
                    # Format as GeoJSON coordinates
                    polygon_coords = list(poly.exterior.coords) if hasattr(poly, 'exterior') else []
                else:
                    polygon_coords = []
            except ImportError:
                polygon_coords = []

            features.update({
                "storm_cell_id": f"{radar.metadata['instrument_name']}_{radar.time['units']}_{i}",
                "center_lat": lat,
                "center_lon": lon,
                "polygon_geojson": polygon_coords,
                "max_reflectivity_dbz": float(max_reflectivity),
                "area_km2": len(x_indices) * (range_gates[1] - range_gates[0]) * (azimuths[1] - azimuths[0]) * np.pi / 180 * 6371.0 ** 2 / 1000000,
                "timestamp": radar.time["units"]
            })

            storm_cells.append(features)

        return storm_cells

    def _radar_to_latlon(self, radar_lat: float, radar_lon: float, range_km: float, azimuth_deg: float) -> tuple:
        from math import radians, sin, cos, atan2, sqrt
        R = 6371.0
        lat1 = radians(radar_lat)
        lon1 = radians(radar_lon)
        azimuth_rad = radians(azimuth_deg)

        lat2 = np.arcsin(
            sin(lat1) * cos(range_km / R) +
            cos(lat1) * sin(range_km / R) * cos(azimuth_rad)
        )
        lon2 = lon1 + np.arctan2(
            sin(azimuth_rad) * sin(range_km / R) * cos(lat1),
            cos(range_km / R) - sin(lat1) * sin(lat2)
        )

        return np.degrees(lat2), np.degrees(lon2)

def fetch_dual_pol_hail(properties: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch Dual-Pol radar-derived hail size estimates."""
    # Instantiates the engine and tries to get latest info around properties point
    try:
        nexrad = NEXRADLevel2()
        radar_site = properties.get("nearest_radar", "KTLX")
        radar = nexrad.get_latest_scan(radar_site)
        if radar:
            features = nexrad.extract_dual_pol_features(radar, properties.get("latitude", 35.5), properties.get("longitude", -97.5))
            if features:
                return {
                    'max_hail_size': features.get("mesh_inches", 0.0),
                    'hca_inches': features.get("hca_inches", 0.0),
                    'reflectivity_dbz': features.get("reflectivity_dbz", 0.0),
                    'prob_severe': 0.82 if features.get("reflectivity_dbz", 0) > 55 else 0.1,
                    'timestamp': features.get("timestamp"),
                    'source': 'NEXRAD_DP'
                }
    except Exception as e:
        print(f"Failed fetching real radar: {e}")
    
    return {
        'max_hail_size': 0.0,
        'prob_severe': 0.0,
        'timestamp': properties.get('timestamp'),
        'source': 'NEXRAD_DP_FALLBACK'
    }
