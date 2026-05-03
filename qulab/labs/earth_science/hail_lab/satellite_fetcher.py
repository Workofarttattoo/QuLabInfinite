import os

import requests


class SatelliteFetcher:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GOOGLE_MAPS_API_KEY")
        self.base_url = "https://maps.googleapis.com/maps/api/staticmap"

    def fetch_roof_top(self, lat: float, lon: float, zoom: int = 20, size: str = "1024x1024") -> str | None:
        """
        Fetches a high-res satellite tile centered on the target roof.
        Zoom 20 is typically high enough for single-structure detail.
        """
        if not self.api_key:
            print("Warning: No Google Maps API key provided. Satellite fetch will fail.")
            return None

        params = {
            "center": f"{lat},{lon}",
            "zoom": zoom,
            "size": size,
            "maptype": "satellite",
            "key": self.api_key
        }

        try:
            response = requests.get(self.base_url, params=params)
            if response.status_code == 200:
                file_path = f"roof_{lat}_{lon}.png"
                with open(file_path, "wb") as f:
                    f.write(response.content)
                return file_path
            else:
                print(f"Error fetching satellite image: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Exception during satellite fetch: {e}")

        return None
