from __future__ import annotations

import json
import os
from typing import Optional

import h5py
from sqlalchemy import create_engine, text


def get_database_url() -> Optional[str]:
    return os.environ.get("DATABASE_URL")  # e.g., postgresql+psycopg2://user:pass@host:5432/db


def populate_postgres(registry_path: str) -> None:
    db_url = get_database_url()
    if not db_url:
        print("DATABASE_URL not set; skipping PostgreSQL population.")
        return
    engine = create_engine(db_url, future=True)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS materials_registry (
                    id SERIAL PRIMARY KEY,
                    source TEXT NOT NULL,
                    key TEXT NOT NULL,
                    data JSONB NOT NULL
                )
                """
            )
        )
        with open(registry_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                conn.execute(
                    text("INSERT INTO materials_registry (source, key, data) VALUES (:source, :key, :data)"),
                    {"source": rec["source"], "key": rec["key"], "data": json.dumps(rec["data"])},
                )
    print("PostgreSQL population complete.")


def populate_hdf5(registry_path: str, hdf5_path: str = "data/registry.h5") -> None:
    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
    with h5py.File(hdf5_path, "w") as h5:
        ds = h5.create_group("records")
        with open(registry_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                ds.create_dataset(str(i), data=line.encode("utf-8"))
    print(f"HDF5 population complete at {hdf5_path}.")


if __name__ == "__main__":
    registry = os.environ.get("REGISTRY_PATH", "data/registry.jsonl")
    if not os.path.exists(registry):
        raise SystemExit(f"Registry file not found: {registry}. Run the ingestion pipeline first.")
    populate_postgres(registry)
    populate_hdf5(registry)
