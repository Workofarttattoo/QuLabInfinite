import argparse
import json
from pathlib import Path
import sys
import psycopg2
from psycopg2.extras import execute_values
import h5py
import numpy as np

# Add the project root to the python path to allow for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from ingest.schemas import RecordChem, RecordMaterial

def populate_postgresql(records_path: Path, conn_string: str):
    """
    Populate a PostgreSQL database with chemical and material records.
    """
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()

    # Create tables if they don't exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chem_records (
            id SERIAL PRIMARY KEY,
            substance TEXT,
            phase TEXT,
            pressure_pa REAL,
            temperature_k REAL,
            enthalpy_j_per_mol REAL,
            entropy_j_per_mol_k REAL,
            content_hash TEXT UNIQUE
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS material_records (
            id SERIAL PRIMARY KEY,
            substance TEXT,
            material_id TEXT,
            formation_energy_per_atom_ev REAL,
            band_gap_ev REAL,
            content_hash TEXT UNIQUE
        );
    """)

    chem_data = []
    material_data = []

    with open(records_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if 'material_id' in data:
                record = RecordMaterial.model_validate(data)
                material_data.append((
                    record.substance, record.material_id, record.formation_energy_per_atom_ev,
                    record.band_gap_ev, record.content_hash()
                ))
            else:
                record = RecordChem.model_validate(data)
                chem_data.append((
                    record.substance, record.phase, record.pressure_pa, record.temperature_k,
                    record.enthalpy_j_per_mol, record.entropy_j_per_mol_k, record.content_hash()
                ))

    # Bulk insert records
    if chem_data:
        execute_values(cur,
            "INSERT INTO chem_records (substance, phase, pressure_pa, temperature_k, enthalpy_j_per_mol, entropy_j_per_mol_k, content_hash) VALUES %s ON CONFLICT (content_hash) DO NOTHING",
            chem_data)
        print(f"Inserted/updated {len(chem_data)} chemical records.")

    if material_data:
        execute_values(cur,
            "INSERT INTO material_records (substance, material_id, formation_energy_per_atom_ev, band_gap_ev, content_hash) VALUES %s ON CONFLICT (content_hash) DO NOTHING",
            material_data)
        print(f"Inserted/updated {len(material_data)} material records.")

    conn.commit()
    cur.close()
    conn.close()

def populate_hdf5(records_path: Path, hdf5_path: Path):
    """
    Populate an HDF5 file with material structures.
    """
    with h5py.File(hdf5_path, 'a') as hf:
        with open(records_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if 'material_id' in data:
                    record = RecordMaterial.model_validate(data)
                    group_name = f"materials/{record.material_id}"
                    if group_name not in hf:
                        group = hf.create_group(group_name)
                        structure = record.structure
                        group.create_dataset('lattice_matrix', data=structure['lattice']['matrix'])
                        sites = []
                        for site in structure['sites']:
                            sites.append(site['xyz'] + [ord(site['species'][0]['element'])]) # simple way to store element
                        group.create_dataset('sites', data=np.array(sites))
                        print(f"Stored structure for {record.material_id}")

def main():
    parser = argparse.ArgumentParser(description="Populate databases with ingested data.")
    parser.add_argument("records_path", type=str, help="Path to the .jsonl file with records.")
    parser.add_argument("--db-conn", type=str, default="dbname=qulab user=qulab", help="PostgreSQL connection string.")
    parser.add_argument("--hdf5-path", type=str, default="data/qulab_structures.hdf5", help="Path to HDF5 file for structures.")
    args = parser.parse_args()

    records_path = Path(args.records_path)
    if not records_path.exists():
        raise FileNotFoundError(f"Records file not found at {records_path}")

    populate_postgresql(records_path, args.db_conn)
    populate_hdf5(records_path, Path(args.hdf5_path))

if __name__ == "__main__":
    main()
