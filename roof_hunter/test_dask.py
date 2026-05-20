import os
import sys

# Add QuLabInfinite to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations.dask_pipeline import run_scalable_pipeline

if __name__ == "__main__":
    # Point to the existing CSV file you mentioned earlier
    input_csv = "/Users/noone/Downloads/GitHub/roof-hunter/roof_hunter/okc_edmond_batch_CANDY.csv"
    output_csv = "/Users/noone/QuLabInfinite/roof_hunter/data/okc_edmond_processed_dask.csv"
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    if not os.path.exists(input_csv):
        print(f"Error: Could not find input CSV at {input_csv}")
        # Create a tiny dummy CSV to test the pipeline if it doesn't exist
        print("Creating a dummy test.csv instead...")
        input_csv = "test.csv"
        with open(input_csv, "w") as f:
            f.write("first_name,last_name,latitude,longitude\n")
            f.write("John,Doe,35.4676,-97.5164\n")
            f.write("Jane,Smith,35.5,-97.6\n")
            f.write("Bob,Jones,35.4,-97.4\n")
            f.write("Alice,Williams,35.3,-97.3\n")
            f.write("Charlie,Brown,35.2,-97.2\n")

    print(f"Starting Dask Pipeline Test with {input_csv}...")
    run_scalable_pipeline(input_csv, output_csv)
    print("Dask Pipeline Test Complete!")
