import os
import json
import pandas as pd
from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
from integrations.ml_models import ensemble_forecast
from integrations.vulnerability_engine import calculate_roof_vulnerability

def setup_dask_cluster(n_workers: int = 4, threads_per_worker: int = 2) -> Client:
    """Initialize a Dask cluster for parallel processing of regional leads."""
    print(f"[Dask] Starting Local Cluster with {n_workers} workers...")
    cluster = LocalCluster(
        n_workers=n_workers, 
        threads_per_worker=threads_per_worker,
        memory_limit='4GB'
    )
    client = Client(cluster)
    print(f"[Dask] Dashboard available at: {client.dashboard_link}")
    return client

def process_lead(lead_row):
    """Worker function to process a single lead."""
    # 1. Weather Ensemble Forecast
    # If latitude/longitude are missing from the CSV, default to an OKC area coordinate
    lat = lead_row.get('latitude', 35.4676)
    lon = lead_row.get('longitude', -97.5164)
    if pd.isna(lat): lat = 35.4676
    if pd.isna(lon): lon = -97.5164
    
    loc = {'lat': lat, 'lon': lon}
    forecast = ensemble_forecast(loc, horizon_hours=2)
    
    # 2. Vulnerability Engine (includes Property API fetch)
    lead_data = lead_row.to_dict()
    lead_data['latitude'] = lat
    lead_data['longitude'] = lon
    lead_data['hail_probability'] = forecast['hail_probability']
    vuln_score = calculate_roof_vulnerability(lead_data)
    
    lead_data['vulnerability_score'] = vuln_score
    return lead_data

def run_scalable_pipeline(csv_path: str, output_path: str):
    """Run the Roof Hunter pipeline over a large dataset using Dask."""
    client = setup_dask_cluster()
    
    print(f"[Pipeline] Loading dataset: {csv_path}")
    # Use Dask dataframe for out-of-core computation
    df = dd.read_csv(csv_path)
    
    # Apply processing across the cluster
    # Using meta to define output types
    print("[Pipeline] Dispatching tasks to Dask cluster...")
    processed_df = df.apply(process_lead, axis=1, meta=dict)
    
    # Compute results and save
    results = processed_df.compute()
    
    import pandas as pd
    final_df = pd.DataFrame(results.tolist())
    final_df.to_csv(output_path, index=False)
    print(f"[Pipeline] Saved {len(final_df)} processed leads to {output_path}")
    
    client.close()

if __name__ == "__main__":
    # Example usage:
    # run_scalable_pipeline("data/oklahoma_leads.csv", "data/oklahoma_processed.csv")
    pass
