"""
Calibration system for FJH reactor — store predicted vs measured values.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MeasurementRecord:
    """Single physical measurement for calibration."""

    measurement_type: str  # V_t, I_t, pressure, Raman, XRD, SEM, etc.
    predicted_value: float | None
    measured_value: float | None
    unit: str = ""
    timestamp: float = field(default_factory=time.time)
    notes: str = ""

    @property
    def prediction_error(self) -> float | None:
        if self.predicted_value is not None and self.measured_value is not None:
            return self.measured_value - self.predicted_value
        return None


@dataclass
class PhysicalRunRecord:
    """Physical experiment run for calibration."""

    run_id: str
    experiment_id: str
    measured_initial_voltage_V: float | None = None
    measured_V_t: list[float] | None = None
    measured_I_t: list[float] | None = None
    measured_pressure_Pa: float | None = None
    measured_sample_mass_g: float | None = None
    measured_resistance_ohm: float | None = None
    observed_damage: str | None = None
    raman_results: dict | None = None
    xrd_results: dict | None = None
    sem_tem_results: dict | None = None
    haadf_stem_results: dict | None = None
    xps_results: dict | None = None
    xanes_exafs_results: dict | None = None
    icp_ms_results: dict | None = None
    measurements: list[MeasurementRecord] = field(default_factory=list)


class CalibrationDatabase:
    """SQLite-backed calibration storage."""

    def __init__(self, db_path: str = "data/fjh_calibration.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS calibration_runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    data_json TEXT,
                    created_at REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS measurements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    measurement_type TEXT,
                    predicted REAL,
                    measured REAL,
                    error REAL,
                    unit TEXT,
                    created_at REAL,
                    FOREIGN KEY (run_id) REFERENCES calibration_runs(run_id)
                )
            """)

    def store_physical_run(self, record: PhysicalRunRecord) -> str:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO calibration_runs (run_id, experiment_id, data_json, created_at) VALUES (?, ?, ?, ?)",
                (record.run_id, record.experiment_id, json.dumps(asdict(record), default=str), time.time()),
            )
            for m in record.measurements:
                conn.execute(
                    "INSERT INTO measurements (run_id, measurement_type, predicted, measured, error, unit, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (record.run_id, m.measurement_type, m.predicted_value, m.measured_value, m.prediction_error, m.unit, time.time()),
                )
        return record.run_id

    def get_calibration_report(self, run_id: str) -> dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT data_json FROM calibration_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if not row:
                return {"error": f"Run {run_id} not found"}
            data = json.loads(row[0])
            measurements = conn.execute(
                "SELECT measurement_type, predicted, measured, error, unit FROM measurements WHERE run_id = ?",
                (run_id,),
            ).fetchall()
        return {
            "run_id": run_id,
            "data": data,
            "measurements": [
                {"type": m[0], "predicted": m[1], "measured": m[2], "error": m[3], "unit": m[4]}
                for m in measurements
            ],
            "calibration_cycle": "simulation -> experiment -> characterization -> model correction",
        }
