# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

import json
from typing import Any

from qulab.labs.earth_science.atmospheric_science_lab.atmospheric_science_lab import (
    AtmosphericScienceLab,
)
from qulab.labs.earth_science.hail_lab.hail_lead_gen import HailLeadEngine


class HailBridge:
    """
    Orchestrates the workflow from Hail Digital Twin detection to Lead Generation and Outreach.
    """
    def __init__(self, pdl_key: str, eleven_key: str):
        self.lab = AtmosphericScienceLab()
        self.lead_engine = HailLeadEngine(pdl_key, eleven_key)

    def run_strike_analysis_and_lead_gen(self, center_lat: float, center_lon: float, surface_temp_c: float) -> list[dict[str, Any]]:
        """
        Runs the full workflow:
        1. Calculate growth zone.
        2. Simulate strike zones.
        3. Identify high-precision leads.
        4. Generate outreach scripts.
        """
        # 1. Atmospheric Analysis
        growth_zone = self.lab.hail_twin.calculate_hail_growth_zone(surface_temp_c)

        # 2. Strike Zone Simulation (Monte Carlo)
        # We use a higher intensity if growth zone is thick
        storm_intensity = 1.0 + (growth_zone['growth_zone_thickness_m'] / 5000.0)
        strikes = self.lab.hail_twin.simulate_hail_strike_zones(center_lat, center_lon, storm_intensity, num_simulations=10)

        leads = []
        for strike in strikes:
            # 3. Lead Generation (only for significant impacts)
            if strike['impact_energy_joules'] > 2.0:
                contact = self.lead_engine.process_strike(strike['lat'], strike['lon'], strike)
                if contact:
                    # 4. Generate ElevenLabs Outreach Script
                    script = self.generate_outreach_script(contact, strike)
                    leads.append({
                        "contact": contact,
                        "strike": strike,
                        "outreach_script": script
                    })

        return leads

    def generate_outreach_script(self, contact: dict[str, Any], strike: dict[str, Any]) -> str:
        """
        Generates a personalized ElevenLabs script based on 3D Mesh/Physics logic.
        """
        address = contact.get('address', 'your property')
        name = contact.get('name', 'Resident')
        energy = round(strike['impact_energy_joules'], 1)

        script = (
            f"Hello {name}, this is an automated update from the Work of Art Digital Twin. "
            f"Our 3D simulation of your roof at {address} indicates that the current hail trajectory "
            f"will impact your South-facing shingles at over {energy} Joules of energy—which "
            f"exceeds the standard puncture threshold."
        )
        return script

if __name__ == "__main__":
    # Demo run
    bridge = HailBridge(pdl_key="MOCK_PDL_KEY", eleven_key="MOCK_ELEVEN_KEY")
    results = bridge.run_strike_analysis_and_lead_gen(32.7767, -96.7970, 25.0) # Dallas, TX
    print(json.dumps(results, indent=2))
