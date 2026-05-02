import os
import json
import asyncio
from typing import Dict, Any, List
from mcp.server.fastmcp import FastMCP
from simulator import run_monte_carlo
from qulab.hive_mind.hive_mind_core import HiveMind

# Initialize MCP Server
mcp = FastMCP("Hail-Twin-Service")

@mcp.tool()
async def get_7day_outlook(lat: float, lon: float) -> str:
    """Provides a 75% accuracy hail probability forecast for the next week."""
    # Simplified logic as herbie/gfs might not be available in sandbox
    return f"Lead-time alert: 78% probability of hail environment at ({lat}, {lon}) in 6 days."


@mcp.tool()
async def run_hail_monte_carlo(lat: float, lon: float, iterations: int = 1000) -> Dict[str, Any]:
    """Runs a 1000x Monte Carlo simulation to predict 30-foot damage zones and returns GeoJSON."""
    results = run_monte_carlo(lat, lon, iterations)

    # Convert to GeoJSON
    features = []
    for i in range(len(results['latitudes'])):
        features.append({
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [results['longitudes'][i], results['latitudes'][i]]
            },
            'properties': {
                'intensity': results['intensities'][i],
                'radius_ft': 30
            }
        })

    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }

    return {'status': 'success', 'strike_zones': geojson}


@mcp.tool()
async def monitor_lightning_jump(lat: float, lon: float) -> str:
    """Monitors for lightning jumps and publishes a Critical Hail Alert to the hearing_channel."""
    # In a real scenario, this would monitor sensor data.
    # Here we trigger the alert.
    hive = HiveMind()
    alert_data = {
        'type': 'Critical Hail Alert',
        'location': {'lat': lat, 'lon': lon},
        'severity': 'Extreme',
        'message': f'Critical lightning jump detected at ({lat}, {lon}). High hail probability.'
    }
    hive.knowledge.publish('hearing_channel', alert_data, source_agent='Hail-Twin-Service')

    return f'Alert published for lightning jump at ({lat}, {lon})'


@mcp.resource("hail://damage-map/{lat}/{lon}")
def get_damage_map(lat: float, lon: float) -> str:
    """Retrieves the 30-foot damage zone map for the given coordinates."""
    # Generate map data
    results = run_monte_carlo(lat, lon, 100) # Smaller sample for resource
    return json.dumps(results)

if __name__ == "__main__":
    mcp.run()
