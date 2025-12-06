"""
Unified Results Database
Central storage for all experiment results with efficient querying
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import logging
import sqlite3
from pathlib import Path

try:
    import psycopg2
    from psycopg2.extras import execute_values
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ResultsQuery:
    """Query object for searching results"""
    lab_names: Optional[List[str]] = None
    status: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    parameter_filters: Optional[Dict] = None
    limit: int = 100
    offset: int = 0


class UnifiedResultsDatabase:
    """
    Central database for all experiment results.

    Features:
    - Multi-lab result storage
    - Efficient querying with filters
    - Cross-lab discovery
    - Result lineage tracking
    - Vector embeddings for semantic search
    """

    def __init__(
        self,
        db_type: str = "sqlite",
        db_path: str = "./qulab_results.db",
        db_connection_string: Optional[str] = None
    ):
        """
        Initialize results database.

        Args:
            db_type: "sqlite" or "postgresql"
            db_path: Path for SQLite database
            db_connection_string: PostgreSQL connection string
        """
        self.db_type = db_type
        self.db_path = db_path
        self.db_connection_string = db_connection_string

        if db_type == "sqlite":
            self._init_sqlite()
        elif db_type == "postgresql" and PSYCOPG2_AVAILABLE:
            self._init_postgresql()
        else:
            logger.warning(f"Database type {db_type} not available, falling back to SQLite")
            self.db_type = "sqlite"
            self._init_sqlite()

        logger.info(f"✓ Initialized {db_type} results database")

    def _init_sqlite(self):
        """Initialize SQLite database"""
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Create tables
        cursor.executescript("""
        CREATE TABLE IF NOT EXISTS experiment_results (
            task_id TEXT PRIMARY KEY,
            lab_name TEXT NOT NULL,
            parameters TEXT NOT NULL,
            result TEXT,
            status TEXT NOT NULL,
            runtime_seconds REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            error_message TEXT,
            retries_used INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_lab_name ON experiment_results(lab_name);
        CREATE INDEX IF NOT EXISTS idx_status ON experiment_results(status);
        CREATE INDEX IF NOT EXISTS idx_created_at ON experiment_results(created_at);

        CREATE TABLE IF NOT EXISTS result_lineage (
            dependent_task_id TEXT,
            dependency_task_id TEXT,
            PRIMARY KEY (dependent_task_id, dependency_task_id),
            FOREIGN KEY (dependent_task_id) REFERENCES experiment_results(task_id),
            FOREIGN KEY (dependency_task_id) REFERENCES experiment_results(task_id)
        );

        CREATE TABLE IF NOT EXISTS result_embeddings (
            task_id TEXT PRIMARY KEY,
            embedding BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (task_id) REFERENCES experiment_results(task_id)
        );

        CREATE TABLE IF NOT EXISTS result_annotations (
            annotation_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            annotation_type TEXT,
            annotation_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by TEXT,
            FOREIGN KEY (task_id) REFERENCES experiment_results(task_id)
        );
        """)

        conn.commit()
        conn.close()

        logger.info(f"SQLite database initialized at {self.db_path}")

    def _init_postgresql(self):
        """Initialize PostgreSQL database"""
        try:
            conn = psycopg2.connect(self.db_connection_string)
            cursor = conn.cursor()

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiment_results (
                task_id VARCHAR(255) PRIMARY KEY,
                lab_name VARCHAR(255) NOT NULL,
                parameters JSONB NOT NULL,
                result JSONB,
                status VARCHAR(50) NOT NULL,
                runtime_seconds FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT,
                retries_used INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_lab_name ON experiment_results(lab_name);
            CREATE INDEX IF NOT EXISTS idx_status ON experiment_results(status);
            CREATE INDEX IF NOT EXISTS idx_created_at ON experiment_results(created_at);

            CREATE TABLE IF NOT EXISTS result_lineage (
                dependent_task_id VARCHAR(255),
                dependency_task_id VARCHAR(255),
                PRIMARY KEY (dependent_task_id, dependency_task_id)
            );

            CREATE TABLE IF NOT EXISTS result_embeddings (
                task_id VARCHAR(255) PRIMARY KEY,
                embedding FLOAT8[],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS result_annotations (
                annotation_id VARCHAR(255) PRIMARY KEY,
                task_id VARCHAR(255) NOT NULL,
                annotation_type VARCHAR(255),
                annotation_data JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by VARCHAR(255)
            );
            """)

            conn.commit()
            cursor.close()
            conn.close()

            logger.info("PostgreSQL database initialized")

        except Exception as e:
            logger.error(f"PostgreSQL initialization failed: {e}")
            raise

    def store_result(
        self,
        task_id: str,
        lab_name: str,
        parameters: Dict,
        result: Dict,
        status: str,
        runtime_seconds: float,
        error_message: Optional[str] = None
    ) -> None:
        """
        Store an experiment result.

        Args:
            task_id: Unique task identifier
            lab_name: Name of the lab
            parameters: Input parameters
            result: Result data
            status: Completion status
            runtime_seconds: Execution time
            error_message: Error message if failed
        """
        if self.db_type == "sqlite":
            self._store_result_sqlite(
                task_id, lab_name, parameters, result,
                status, runtime_seconds, error_message
            )
        else:
            self._store_result_postgresql(
                task_id, lab_name, parameters, result,
                status, runtime_seconds, error_message
            )

    def _store_result_sqlite(
        self,
        task_id: str,
        lab_name: str,
        parameters: Dict,
        result: Dict,
        status: str,
        runtime_seconds: float,
        error_message: Optional[str] = None
    ) -> None:
        """Store result in SQLite"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
        INSERT OR REPLACE INTO experiment_results
        (task_id, lab_name, parameters, result, status, runtime_seconds, error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            task_id,
            lab_name,
            json.dumps(parameters),
            json.dumps(result),
            status,
            runtime_seconds,
            error_message
        ))

        conn.commit()
        conn.close()

    def _store_result_postgresql(
        self,
        task_id: str,
        lab_name: str,
        parameters: Dict,
        result: Dict,
        status: str,
        runtime_seconds: float,
        error_message: Optional[str] = None
    ) -> None:
        """Store result in PostgreSQL"""
        conn = psycopg2.connect(self.db_connection_string)
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO experiment_results
        (task_id, lab_name, parameters, result, status, runtime_seconds, error_message)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (task_id) DO UPDATE SET
            result = EXCLUDED.result,
            status = EXCLUDED.status,
            runtime_seconds = EXCLUDED.runtime_seconds,
            error_message = EXCLUDED.error_message
        """, (
            task_id,
            lab_name,
            json.dumps(parameters),
            json.dumps(result),
            status,
            runtime_seconds,
            error_message
        ))

        conn.commit()
        cursor.close()
        conn.close()

    def query_results(self, query: ResultsQuery) -> List[Dict]:
        """
        Query results with filters.

        Args:
            query: ResultsQuery object with filters

        Returns:
            List of matching results
        """
        if self.db_type == "sqlite":
            return self._query_results_sqlite(query)
        else:
            return self._query_results_postgresql(query)

    def _query_results_sqlite(self, query: ResultsQuery) -> List[Dict]:
        """Query results from SQLite"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Build query
        where_clauses = []
        params = []

        if query.lab_names:
            placeholders = ",".join(["?" for _ in query.lab_names])
            where_clauses.append(f"lab_name IN ({placeholders})")
            params.extend(query.lab_names)

        if query.status:
            where_clauses.append("status = ?")
            params.append(query.status)

        if query.date_from:
            where_clauses.append("created_at >= ?")
            params.append(query.date_from.isoformat())

        if query.date_to:
            where_clauses.append("created_at <= ?")
            params.append(query.date_to.isoformat())

        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

        sql = f"""
        SELECT task_id, lab_name, parameters, result, status,
               runtime_seconds, created_at, error_message
        FROM experiment_results
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
        """

        params.extend([query.limit, query.offset])

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def _query_results_postgresql(self, query: ResultsQuery) -> List[Dict]:
        """Query results from PostgreSQL"""
        conn = psycopg2.connect(self.db_connection_string)
        cursor = conn.cursor()

        # Build query
        where_clauses = []
        params = []

        if query.lab_names:
            placeholders = ",".join(["%s"] * len(query.lab_names))
            where_clauses.append(f"lab_name IN ({placeholders})")
            params.extend(query.lab_names)

        if query.status:
            where_clauses.append("status = %s")
            params.append(query.status)

        if query.date_from:
            where_clauses.append("created_at >= %s")
            params.append(query.date_from)

        if query.date_to:
            where_clauses.append("created_at <= %s")
            params.append(query.date_to)

        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

        sql = f"""
        SELECT task_id, lab_name, parameters, result, status,
               runtime_seconds, created_at, error_message
        FROM experiment_results
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT %s OFFSET %s
        """

        params.extend([query.limit, query.offset])

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            results.append({
                "task_id": row[0],
                "lab_name": row[1],
                "parameters": row[2],
                "result": row[3],
                "status": row[4],
                "runtime_seconds": row[5],
                "created_at": row[6].isoformat(),
                "error_message": row[7]
            })

        return results

    def get_result(self, task_id: str) -> Optional[Dict]:
        """
        Get a specific result by task ID.

        Args:
            task_id: Task identifier

        Returns:
            Result dictionary or None
        """
        query = ResultsQuery(limit=1)

        if self.db_type == "sqlite":
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
            SELECT task_id, lab_name, parameters, result, status,
                   runtime_seconds, created_at, error_message
            FROM experiment_results
            WHERE task_id = ?
            """, (task_id,))

            row = cursor.fetchone()
            conn.close()

            return dict(row) if row else None

        else:
            conn = psycopg2.connect(self.db_connection_string)
            cursor = conn.cursor()

            cursor.execute("""
            SELECT task_id, lab_name, parameters, result, status,
                   runtime_seconds, created_at, error_message
            FROM experiment_results
            WHERE task_id = %s
            """, (task_id,))

            row = cursor.fetchone()
            conn.close()

            if row:
                return {
                    "task_id": row[0],
                    "lab_name": row[1],
                    "parameters": row[2],
                    "result": row[3],
                    "status": row[4],
                    "runtime_seconds": row[5],
                    "created_at": row[6].isoformat(),
                    "error_message": row[7]
                }

            return None

    def get_stats(self) -> Dict:
        """Get database statistics"""
        if self.db_type == "sqlite":
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM experiment_results")
            total_count = cursor.fetchone()[0]

            cursor.execute("""
            SELECT lab_name, COUNT(*) as count
            FROM experiment_results
            GROUP BY lab_name
            """)

            by_lab = {row[0]: row[1] for row in cursor.fetchall()}

            cursor.execute("""
            SELECT status, COUNT(*) as count
            FROM experiment_results
            GROUP BY status
            """)

            by_status = {row[0]: row[1] for row in cursor.fetchall()}

            conn.close()

        else:
            conn = psycopg2.connect(self.db_connection_string)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM experiment_results")
            total_count = cursor.fetchone()[0]

            cursor.execute("""
            SELECT lab_name, COUNT(*) as count
            FROM experiment_results
            GROUP BY lab_name
            """)

            by_lab = {row[0]: row[1] for row in cursor.fetchall()}

            cursor.execute("""
            SELECT status, COUNT(*) as count
            FROM experiment_results
            GROUP BY status
            """)

            by_status = {row[0]: row[1] for row in cursor.fetchall()}

            cursor.close()
            conn.close()

        return {
            "total_results": total_count,
            "by_lab": by_lab,
            "by_status": by_status,
            "timestamp": datetime.now().isoformat()
        }
