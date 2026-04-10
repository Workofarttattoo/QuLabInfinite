import argparse
import json
import psycopg2
from psycopg2.extras import Json
from pathlib import Path
import sys

# Add the project root to the python path to allow for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from ingest.schemas import RecordChem

# --- DATABASE CONNECTION PARAMETERS ---
# PLEASE REPLACE THESE WITH YOUR ACTUAL DATABASE CREDENTIALS
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "qulab_infinite"
DB_USER = "user"
DB_PASSWORD = "password"
# ------------------------------------

def create_table(conn):
    """Create the 'records' table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id SERIAL PRIMARY KEY,
                substance TEXT,
                phase TEXT,
                pressure_pa DOUBLE PRECISION,
                temperature_k DOUBLE PRECISION,
                volume_m3_per_mol DOUBLE PRECISION,
                enthalpy_j_per_mol DOUBLE PRECISION,
                entropy_j_per_mol_k DOUBLE PRECISION,
                composition JSONB,
                experiment_id TEXT,
                tags TEXT[],
                provenance JSONB
            )
        """)
        conn.commit()

def insert_record(conn, record: RecordChem):
    """Insert a single RecordChem object into the database."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO records (
                substance, phase, pressure_pa, temperature_k, volume_m3_per_mol,
                enthalpy_j_per_mol, entropy_j_per_mol_k, composition,
                experiment_id, tags, provenance
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                record.substance,
                record.phase,
                record.pressure_pa,
                record.temperature_k,
                record.volume_m3_per_mol,
                record.enthalpy_j_per_mol,
                record.entropy_j_per_mol_k,
                Json(record.composition) if record.composition else None,
                record.experiment_id,
                record.tags,
                Json(record.provenance.dict())
            )
        )

def main():
    parser = argparse.ArgumentParser(description="Populate the database from an ingested dataset.")
    parser.add_argument("dataset_path", type=str, help="Path to the ingested dataset file (.jsonl).")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
    except psycopg2.OperationalError as e:
        print(f"Error connecting to the database: {e}")
        print("Please ensure that PostgreSQL is running and that the connection parameters are correct.")
        return

    create_table(conn)
    print("Table 'records' created or already exists.")

    with open(dataset_path, 'r') as f:
        for line in f:
            record_data = json.loads(line)
            record = RecordChem.model_validate(record_data)
            insert_record(conn, record)

    print(f"Successfully populated database from {dataset_path}")

    conn.close()

if __name__ == "__main__":
    main()
