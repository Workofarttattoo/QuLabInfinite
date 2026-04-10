import argparse
import json
from pathlib import Path
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent))
from ingest.schemas import RecordChem, RecordMaterial
from database.models import RecordChemDB, RecordMaterialDB, Base

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

    with open(dataset_path, 'r') as f:
        for line in f:
            record_data = json.loads(line)
            
            try:
                if 'material_id' in record_data:
                    record = RecordMaterial.model_validate(record_data)
                    db_record = RecordMaterialDB(**record.model_dump())
                    session.merge(db_record)
                else:
                    record = RecordChem.model_validate(record_data)
                    db_record = RecordChemDB(**record.model_dump())
                    session.add(db_record)
            except Exception as e:
                print(f"Skipping record due to error: {e}")
    
    session.commit()
    print(f"Database populated successfully from {args.dataset_path}")

if __name__ == "__main__":
    main()

