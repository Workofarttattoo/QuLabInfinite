from __future__ import annotations
import pandas as pd


class ResultsValidator:
    def validate_numeric_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add simple validity flags; extend with physics-based constraints.

        Columns added:
          - valid_value: True if value is finite and within coarse bounds
        """
        out = df.copy()
        out["valid_value"] = pd.to_numeric(out["value"], errors="coerce").between(-1e12, 1e12)
        return out

    def check_required_columns(self, df: pd.DataFrame) -> list[str]:
        required = {"material_id", "property_name", "value", "units", "source"}
        missing = [c for c in required if c not in df.columns]
        return missing
