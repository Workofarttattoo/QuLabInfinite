from __future__ import annotations
from pathlib import Path
import pandas as pd
import h5py


def dataframe_to_hdf5(df: pd.DataFrame, out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        # Store as simple datasets per column; extend to structured datasets as needed
        for col in df.columns:
            # Convert object columns to utf-8 bytes
            if df[col].dtype == object:
                data = df[col].fillna("").astype(str).str.encode("utf-8").to_numpy()
                dt = h5py.string_dtype(encoding="utf-8")
                f.create_dataset(col, data=data, dtype=dt)
            else:
                f.create_dataset(col, data=df[col].to_numpy())


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python scripts/populate_hdf5.py <parquet_path> <hdf5_path>")
        raise SystemExit(2)

    parquet_path, hdf5_path = sys.argv[1], sys.argv[2]
    df = pd.read_parquet(parquet_path)
    dataframe_to_hdf5(df, hdf5_path)
    print(f"Wrote HDF5 file: {hdf5_path}")
