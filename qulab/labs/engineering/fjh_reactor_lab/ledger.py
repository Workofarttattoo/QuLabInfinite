"""
Experiment ledger — immutable, reproducible experiment records.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SOFTWARE_VERSION = "1.0.0"
MODEL_VERSION = "fjh_twin_v1"


@dataclass
class ExperimentLedgerEntry:
    """Immutable experiment record."""

    experiment_id: str
    timestamp: float
    configuration_hash: str
    software_version: str
    model_version: str
    input_parameters: dict[str, Any]
    assumptions: list[str]
    unknown_parameters: list[str]
    simulation_results: dict[str, Any] | None = None
    characterization_results: dict[str, Any] | None = None
    notes: str = ""
    experiment_type: str = "virtual"  # virtual or physical

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentLedger:
    """Append-only experiment ledger."""

    def __init__(self, db_path: str = "data/fjh_experiment_ledger.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    configuration_hash TEXT,
                    software_version TEXT,
                    model_version TEXT,
                    experiment_type TEXT,
                    entry_json TEXT NOT NULL,
                    created_at REAL
                )
            """)

    def record(
        self,
        configuration_hash: str,
        input_parameters: dict[str, Any],
        assumptions: list[str],
        unknown_parameters: list[str],
        simulation_results: dict[str, Any] | None = None,
        characterization_results: dict[str, Any] | None = None,
        notes: str = "",
        experiment_type: str = "virtual",
        experiment_id: str | None = None,
    ) -> ExperimentLedgerEntry:
        """Append experiment record — never overwrites."""
        eid = experiment_id or str(uuid.uuid4())
        entry = ExperimentLedgerEntry(
            experiment_id=eid,
            timestamp=time.time(),
            configuration_hash=configuration_hash,
            software_version=SOFTWARE_VERSION,
            model_version=MODEL_VERSION,
            input_parameters=input_parameters,
            assumptions=assumptions,
            unknown_parameters=unknown_parameters,
            simulation_results=simulation_results,
            characterization_results=characterization_results,
            notes=notes,
            experiment_type=experiment_type,
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO experiments (experiment_id, timestamp, configuration_hash, software_version, model_version, experiment_type, entry_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (eid, entry.timestamp, configuration_hash, SOFTWARE_VERSION, MODEL_VERSION, experiment_type, json.dumps(entry.to_dict(), default=str), time.time()),
            )
        return entry

    def get(self, experiment_id: str) -> ExperimentLedgerEntry | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT entry_json FROM experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
        if row:
            data = json.loads(row[0])
            return ExperimentLedgerEntry(**data)
        return None

    def list_experiments(self, limit: int = 50) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT experiment_id, timestamp, configuration_hash, experiment_type FROM experiments ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"experiment_id": r[0], "timestamp": r[1], "configuration_hash": r[2], "experiment_type": r[3]}
            for r in rows
        ]

    def compare(self, experiment_id_a: str, experiment_id_b: str) -> dict[str, Any]:
        """Compare two experiments for reproducibility analysis."""
        a = self.get(experiment_id_a)
        b = self.get(experiment_id_b)
        if not a or not b:
            return {"error": "One or both experiments not found"}
        diffs = {}
        for key in set(a.input_parameters) | set(b.input_parameters):
            va, vb = a.input_parameters.get(key), b.input_parameters.get(key)
            if va != vb:
                diffs[key] = {"A": va, "B": vb}
        return {
            "experiment_a": experiment_id_a,
            "experiment_b": experiment_id_b,
            "configuration_match": a.configuration_hash == b.configuration_hash,
            "parameter_differences": diffs,
            "model_version_a": a.model_version,
            "model_version_b": b.model_version,
        }
