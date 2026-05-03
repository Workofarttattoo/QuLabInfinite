# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

import time
from typing import Any

import requests

try:
    import gspread
    from geopy.geocoders import Nominatim
    from oauth2client.service_account import ServiceAccountCredentials
except ImportError:
    # Mocks for environments without these dependencies
    gspread = None
    Nominatim = None
    ServiceAccountCredentials = None

class HailLeadEngine:
    """
    Lead Generation Engine for Hail Strike Intelligence.
    Pivoted to OpenStreetMap (Nominatim) for free reverse geocoding.
    """
    def __init__(self, pdl_key: str, eleven_key: str, service_account_path: str = 'service_account.json'):
        self.pdl_key = pdl_key
        self.eleven_key = eleven_key

        # Initialize Free Geocoder (Nominatim)
        if Nominatim:
            self.geocoder = Nominatim(user_agent="hail_twin_service_v1")
        else:
            self.geocoder = None

        # Initialize Google Sheets
        self.sheet = None
        if gspread and ServiceAccountCredentials:
            try:
                scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
                creds = ServiceAccountCredentials.from_json_keyfile_name(service_account_path, scope)
                self.gs_client = gspread.authorize(creds)
                self.sheet = self.gs_client.open("Hail_Leads_2026").sheet1
            except Exception as e:
                print(f"Warning: Could not initialize Google Sheets: {e}")

    def free_reverse_geocode(self, lat: float, lon: float) -> str | None:
        """Converts lat/lon to address using OpenStreetMap (Free)."""
        if not self.geocoder:
            return f"Lat: {lat}, Lon: {lon} (Geocoder not available)"

        try:
            location = self.geocoder.reverse((lat, lon), timeout=10)
            return location.address if location else None
        except Exception as e:
            print(f"Geocoding error: {e}")
            return None

    def get_homeowner_data(self, address: str) -> dict[str, Any] | None:
        """
        Queries People Data Labs.
        Refinement: Filtering for 'Likely Homeowner' profiles.
        """
        url = "https://api.peopledatalabs.com/v5/location/enrich"
        headers = {"X-Api-Key": self.pdl_key}
        # PDL works best with structured addresses
        params = {"address": address, "include_if_exists": "phone,email,title"}

        try:
            res = requests.get(url, headers=headers, params=params).json()
            if res.get('status') == 200:
                data = res.get('data', {})
                # Filter logic: We prioritize leads with phone numbers for ElevenLabs
                return {
                    "name": data.get('name', 'Resident'),
                    "phone": data.get('phone_numbers', [None])[0] if data.get('phone_numbers') else None,
                    "email": data.get('emails', [None])[0] if data.get('emails') else None,
                    "is_homeowner": data.get('is_homeowner', True) # PDL inference
                }
        except Exception as e:
            print(f"PDL Enrichment error: {e}")

        return None

    def process_strike(self, lat: float, lon: float, damage_report: dict[str, Any]) -> dict[str, Any] | None:
        """The Master Workflow called by your Physics Engine."""
        address = self.free_reverse_geocode(lat, lon)
        if not address:
            return None

        contact = self.get_homeowner_data(address)
        if not contact or not contact.get('phone'):
            return None

        # Push to Google Sheets
        if self.sheet:
            try:
                row = [
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    address,
                    contact['name'],
                    contact['phone'],
                    damage_report.get('damage_probability', 0),
                    damage_report.get('impact_energy_joules', 0)
                ]
                self.sheet.append_row(row)
                print(f"📈 Lead Saved to Sheet: {contact['name']} at {address}")
            except Exception as e:
                print(f"Error saving to sheet: {e}")

        return contact
