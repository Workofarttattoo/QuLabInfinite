from __future__ import annotations
import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values


def connect():
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", "postgres"),
        dbname=os.getenv("PGDATABASE", "materials"),
    )


def ensure_table(cur) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS canonical_properties (
            material_id TEXT NOT NULL,
            formula TEXT,
            property_name TEXT NOT NULL,
            value DOUBLE PRECISION NOT NULL,
            units TEXT NOT NULL,
            temperature_k DOUBLE PRECISION,
            source TEXT NOT NULL,
            source_ref TEXT
        );
        """
    )


def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def upsert_dataframe(cur, df: pd.DataFrame) -> None:
    cols = [
        "material_id", "formula", "property_name", "value", "units",
        "temperature_k", "source", "source_ref",
    ]
    values = [tuple(map(lambda x: None if pd.isna(x) else x, row)) for row in df[cols].itertuples(index=False, name=None)]
    execute_values(
        cur,
        f"INSERT INTO canonical_properties ({', '.join(cols)}) VALUES %s",
        values,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python scripts/populate_postgres.py <parquet_path>")
        raise SystemExit(2)

    parquet_path = sys.argv[1]
    df = load_parquet(parquet_path)
    with connect() as conn:
        with conn.cursor() as cur:
            ensure_table(cur)
            upsert_dataframe(cur, df)
        conn.commit()
    print(f"Loaded {len(df)} rows into PostgreSQL")
