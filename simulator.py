from qulab.labs.earth_science.atmospheric_science_lab.atmospheric_science_lab import AtmosphericScienceLab
from typing import Dict, Any

def run_monte_carlo(lat: float, lon: float, iterations: int = 1000) -> Dict[str, Any]:
    """
    Orchestrates the hail strike zone simulation.
    """
    lab = AtmosphericScienceLab()
    results = lab.simulate_hail_strike_zones(lat, lon, iterations)
    return results

if __name__ == "__main__":
    # Quick test
    res = run_monte_carlo(34.05, -118.24, 10)
    print(res)
