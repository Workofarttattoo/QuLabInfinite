"""
Hail Data Preprocessor — NEXRAD Level II, Mesonet, and SPC report ingestion.

Handles:
  - NEXRAD radar sweeps (reflectivity, velocity, dual-pol variables)
  - Mesonet surface observations (CSV)
  - NOAA SPC severe-weather reports (CSV)
  - Derived meteorological features (CAPE, shear, VIL, echo-top)
  - Spatial merging via nearest-neighbour join

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

# Physically-meaningful clipping bounds
_CLIP_BOUNDS: dict[str, tuple[float, float]] = {
    "reflectivity_max": (0.0, 80.0),
    "reflectivity_mean": (0.0, 80.0),
    "differential_reflectivity": (-2.0, 8.0),
    "correlation_coefficient": (0.8, 1.0),
    "specific_differential_phase": (0.0, 10.0),
    "cape": (0.0, 6000.0),
    "shear_0_6km": (0.0, 100.0),
    "vil": (0.0, 80.0),
    "echo_top_km": (0.0, 20.0),
    "storm_relative_helicity": (-200.0, 600.0),
    "freezing_level_m": (0.0, 6000.0),
}


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load YAML configuration."""
    import yaml

    with open(config_path) as fh:
        return yaml.safe_load(fh)


class HailDataPreprocessor:
    """Ingests, merges, and cleans radar + surface data for hail prediction."""

    def __init__(self, config: dict[str, Any] | None = None, config_path: str | None = None):
        if config is None:
            if config_path is None:
                config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
            config = load_config(config_path)
        self.config = config
        self.hail_threshold: float = config["data"]["hail_threshold"]
        self.target_column: str = config["data"]["target_column"]
        self.feature_columns: list[str] = config["data"]["feature_columns"]

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------

    @staticmethod
    def load_nexrad_csv(file_path: str | Path) -> pd.DataFrame:
        """Load pre-extracted NEXRAD features (CSV).

        Expected columns at minimum:
            latitude, longitude, time, reflectivity_max, reflectivity_mean,
            reflectivity_std, velocity_max, velocity_mean, spectrum_width_mean
        Dual-pol columns (optional but recommended):
            differential_reflectivity, correlation_coefficient,
            specific_differential_phase
        """
        df = pd.read_csv(file_path, parse_dates=["time"])
        logger.info("Loaded NEXRAD CSV: %d rows from %s", len(df), file_path)
        return df

    @staticmethod
    def load_mesonet_csv(file_path: str | Path) -> pd.DataFrame:
        """Load Mesonet surface observations (CSV).

        Expected columns:
            station_id, latitude, longitude, time, temperature_c,
            dewpoint_c, wind_speed_ms, wind_direction_deg
        """
        df = pd.read_csv(file_path, parse_dates=["time"])
        required = {"station_id", "latitude", "longitude", "time"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Mesonet CSV missing columns: {missing}")
        logger.info("Loaded Mesonet CSV: %d rows from %s", len(df), file_path)
        return df

    @staticmethod
    def load_spc_reports(file_path: str | Path, hail_threshold: float = 1.0) -> pd.DataFrame:
        """Load NOAA SPC hail reports (CSV).

        Expected columns:
            latitude, longitude, time, hail_size_inches
        """
        df = pd.read_csv(file_path, parse_dates=["time"])
        df = df[df["hail_size_inches"] >= hail_threshold].copy()
        logger.info(
            "Loaded SPC reports: %d events >= %.1f in from %s",
            len(df), hail_threshold, file_path,
        )
        return df

    # ------------------------------------------------------------------
    # Spatial merging
    # ------------------------------------------------------------------

    @staticmethod
    def spatial_nearest_join(
        left: pd.DataFrame,
        right: pd.DataFrame,
        left_lat: str = "latitude",
        left_lon: str = "longitude",
        right_lat: str = "latitude",
        right_lon: str = "longitude",
        max_dist_deg: float = 0.1,
    ) -> pd.DataFrame:
        """Nearest-neighbour spatial join (haversine-approximate)."""
        if right.empty or left.empty:
            return left

        right_coords = right[[right_lat, right_lon]].values
        left_coords = left[[left_lat, left_lon]].values

        nn = NearestNeighbors(n_neighbors=1, metric="haversine")
        nn.fit(np.deg2rad(right_coords))
        distances, indices = nn.kneighbors(np.deg2rad(left_coords))

        mask = distances.flatten() <= np.deg2rad(max_dist_deg)
        matched = right.iloc[indices.flatten()[mask]].reset_index(drop=True)
        return left.loc[mask].reset_index(drop=True).join(
            matched.drop(columns=[right_lat, right_lon, "time"], errors="ignore"),
            rsuffix="_right",
        )

    def merge_with_spc(self, radar_df: pd.DataFrame, spc_df: pd.DataFrame) -> pd.DataFrame:
        """Label radar rows with hail occurrence from SPC reports."""
        df = radar_df.copy()
        df["hail_occurred"] = 0
        df["hail_size_inches"] = 0.0

        if spc_df.empty:
            return df

        spc_coords = spc_df[["latitude", "longitude"]].values
        radar_coords = df[["latitude", "longitude"]].values

        nn = NearestNeighbors(n_neighbors=1, metric="haversine")
        nn.fit(np.deg2rad(spc_coords))
        distances, indices = nn.kneighbors(np.deg2rad(radar_coords))

        close_mask = distances.flatten() <= np.deg2rad(0.05)
        df.loc[close_mask, "hail_occurred"] = 1
        df.loc[close_mask, "hail_size_inches"] = (
            spc_df.iloc[indices.flatten()[close_mask]]["hail_size_inches"].values
        )

        logger.info(
            "Labelled %d / %d radar rows as hail events",
            close_mask.sum(), len(df),
        )
        return df

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    @staticmethod
    def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        """Derive meteorological and temporal features."""
        out = df.copy()

        if "time" in out.columns:
            ts = pd.to_datetime(out["time"])
            out["hour"] = ts.dt.hour
            out["month"] = ts.dt.month
            out["day_of_year"] = ts.dt.dayofyear
        else:
            for col in ("hour", "month", "day_of_year"):
                if col not in out.columns:
                    out[col] = 0

        # Reflectivity statistics (fallback if only single column)
        if "reflectivity" in out.columns and "reflectivity_max" not in out.columns:
            out["reflectivity_max"] = out["reflectivity"]
            out["reflectivity_mean"] = out["reflectivity"]
            out["reflectivity_std"] = 0.0

        # Fill dual-pol defaults when not available
        dual_pol_defaults = {
            "differential_reflectivity": 0.0,
            "correlation_coefficient": 0.98,
            "specific_differential_phase": 0.0,
        }
        for col, default in dual_pol_defaults.items():
            if col not in out.columns:
                out[col] = default

        # Environment defaults
        env_defaults = {
            "cape": 1500.0,
            "shear_0_6km": 30.0,
            "temp_500mb": -12.0,
            "freezing_level_m": 3500.0,
            "vil": 25.0,
            "echo_top_km": 10.0,
            "storm_relative_helicity": 150.0,
        }
        for col, default in env_defaults.items():
            if col not in out.columns:
                out[col] = default

        return out

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values and clip outliers."""
        out = df.copy()

        critical = ["latitude", "longitude"]
        for col in critical:
            if col in out.columns:
                out = out.dropna(subset=[col])

        for col, (lo, hi) in _CLIP_BOUNDS.items():
            if col in out.columns:
                out[col] = out[col].clip(lo, hi)

        numeric_cols = out.select_dtypes(include=[np.number]).columns
        out[numeric_cols] = out[numeric_cols].fillna(out[numeric_cols].median())

        return out

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def preprocess(
        self,
        nexrad_path: str | Path,
        spc_path: str | Path | None = None,
        mesonet_path: str | Path | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run the full preprocessing pipeline and return (train, test) splits."""
        radar_df = self.load_nexrad_csv(nexrad_path)

        if spc_path:
            spc_df = self.load_spc_reports(spc_path, self.hail_threshold)
            radar_df = self.merge_with_spc(radar_df, spc_df)
        else:
            if "hail_occurred" not in radar_df.columns:
                radar_df["hail_occurred"] = 0
            if "hail_size_inches" not in radar_df.columns:
                radar_df["hail_size_inches"] = 0.0

        radar_df = self.add_derived_features(radar_df)
        radar_df = self.clean_data(radar_df)

        stratify = radar_df[self.target_column] if self.target_column in radar_df.columns else None
        train_df, test_df = train_test_split(
            radar_df,
            test_size=self.config["data"]["test_size"],
            random_state=self.config["data"]["random_state"],
            stratify=stratify,
        )

        logger.info("Split: train=%d, test=%d", len(train_df), len(test_df))
        return train_df, test_df

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess a single DataFrame (no split)."""
        df = self.add_derived_features(df)
        df = self.clean_data(df)
        return df

    def generate_synthetic_data(self, n_samples: int = 2000, hail_fraction: float = 0.15) -> pd.DataFrame:
        """Generate synthetic training data for testing or bootstrapping.

        Creates physically-plausible radar + environment features with
        a controllable hail-event fraction.
        """
        rng = np.random.default_rng(self.config["data"]["random_state"])
        n_hail = int(n_samples * hail_fraction)
        n_nohail = n_samples - n_hail

        def _block(n: int, hail: bool) -> pd.DataFrame:
            ref_mean = 55.0 if hail else 35.0
            ref_std_mean = 8.0 if hail else 4.0
            zdr_mean = 3.0 if hail else 1.0
            rho_mean = 0.92 if hail else 0.98
            kdp_mean = 2.0 if hail else 0.5
            cape_mean = 3000.0 if hail else 1200.0
            shear_mean = 50.0 if hail else 25.0
            vil_mean = 45.0 if hail else 20.0
            et_mean = 13.0 if hail else 8.0
            srh_mean = 250.0 if hail else 100.0

            return pd.DataFrame({
                "latitude": rng.uniform(30.0, 45.0, n),
                "longitude": rng.uniform(-105.0, -85.0, n),
                "time": pd.date_range("2024-04-01", periods=n, freq="15min"),
                "reflectivity_max": rng.normal(ref_mean + 5, 5, n),
                "reflectivity_mean": rng.normal(ref_mean, 5, n),
                "reflectivity_std": rng.normal(ref_std_mean, 2, n).clip(0),
                "differential_reflectivity": rng.normal(zdr_mean, 1.0, n),
                "correlation_coefficient": rng.normal(rho_mean, 0.02, n).clip(0.8, 1.0),
                "specific_differential_phase": rng.normal(kdp_mean, 0.5, n).clip(0),
                "velocity_max": rng.normal(25 if hail else 15, 5, n),
                "velocity_mean": rng.normal(12 if hail else 8, 3, n),
                "spectrum_width_mean": rng.normal(8 if hail else 4, 2, n).clip(0),
                "cape": rng.normal(cape_mean, 500, n).clip(0),
                "shear_0_6km": rng.normal(shear_mean, 10, n).clip(0),
                "temp_500mb": rng.normal(-15 if hail else -8, 3, n),
                "freezing_level_m": rng.normal(3200 if hail else 4000, 300, n).clip(0),
                "vil": rng.normal(vil_mean, 10, n).clip(0),
                "echo_top_km": rng.normal(et_mean, 2, n).clip(0),
                "storm_relative_helicity": rng.normal(srh_mean, 50, n),
                "hail_occurred": int(hail),
                "hail_size_inches": rng.normal(2.0, 0.5, n).clip(1.0) if hail else np.zeros(n),
            })

        hail_df = _block(n_hail, hail=True)
        nohail_df = _block(n_nohail, hail=False)
        combined = pd.concat([hail_df, nohail_df], ignore_index=True)
        return combined.sample(frac=1.0, random_state=self.config["data"]["random_state"]).reset_index(drop=True)
