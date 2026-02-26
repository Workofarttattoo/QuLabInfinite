import argparse
import json
from pathlib import Path
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent))
from ingest.schemas import RecordChem, RecordMaterial
from database.models import RecordChemDB, RecordMaterialDB, Base

def process_records(session, records):
    """
    Optimized function to process a batch of records.
    Separates chemicals (inserts) and materials (upserts).
    Performs bulk operations to avoid N+1 queries.
    """
    chem_objects = []
    material_records = []
    material_ids = []

    # 1. Separate records by type
    for record_data in records:
        if 'material_id' in record_data:
            material_records.append(record_data)
            material_ids.append(record_data['material_id'])
        else:
            try:
                record = RecordChem.model_validate(record_data)
                chem_objects.append(RecordChemDB(**record.model_dump()))
            except Exception as e:
                print(f"Skipping chemical record due to error: {e}")

    # 2. Bulk Insert Chemicals
    if chem_objects:
        session.bulk_save_objects(chem_objects)

    # 3. Optimized Upsert for Materials
    if material_records:
        # Fetch existing records in a single query
        existing_db_records = session.query(RecordMaterialDB).filter(
            RecordMaterialDB.material_id.in_(material_ids)
        ).all()

        existing_map = {r.material_id: r for r in existing_db_records}
        new_material_objects = []

        for record_data in material_records:
            try:
                # Validate first
                record = RecordMaterial.model_validate(record_data)
                data = record.model_dump()
                mat_id = data['material_id']

                if mat_id in existing_map:
                    # Update existing object in session
                    db_record = existing_map[mat_id]
                    for key, value in data.items():
                        if hasattr(db_record, key):
                            setattr(db_record, key, value)
                else:
                    # Prepare for bulk insert
                    new_material_objects.append(RecordMaterialDB(**data))
            except Exception as e:
                print(f"Skipping material record {record_data.get('material_id')} due to error: {e}")

        # Bulk Insert New Materials
        if new_material_objects:
            session.bulk_save_objects(new_material_objects)

def main():
    parser = argparse.ArgumentParser(description="Populate a database from an ingested dataset.")
    parser.add_argument("dataset_path", type=str, help="Path to the ingested dataset file (.jsonl).")
    parser.add_argument("--db-uri", type=str, default="sqlite:///database/qulab.db", help="Database URI (e.g., 'postgresql://user:pass@host/db' or 'sqlite:///path/to/db.sqlite')")
    args = parser.parse_args()

    engine = create_engine(args.db_uri)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # Process in chunks to manage memory for very large files
    BATCH_SIZE = 1000
    batch = []

    with open(dataset_path, 'r') as f:
        for line in f:
            try:
                record_data = json.loads(line)
                batch.append(record_data)

                if len(batch) >= BATCH_SIZE:
                    process_records(session, batch)
                    batch = []
                    # Commit periodically or after batch
                    session.commit()
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}")

    # Process remaining records
    if batch:
        process_records(session, batch)
        session.commit()
    
    print(f"Database populated successfully from {args.dataset_path}")

if __name__ == "__main__":
    main()
