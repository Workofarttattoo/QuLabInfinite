"""
Unified Lab Orchestrator
Central coordinator for all 30+ QuLabInfinite labs
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import asyncio
import uuid
import json
import logging
from pathlib import Path

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import psycopg2
    from psycopg2.extras import execute_values
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Experiment lifecycle states"""
    QUEUED = "queued"
    WAITING_DEPENDENCIES = "waiting_dependencies"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class PriorityLevel(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class ExperimentTask:
    """Represents a single experiment to execute"""
    lab_name: str  # e.g., "cancer_optimizer", "materials_lab"
    parameters: Dict[str, Any]
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: PriorityLevel = PriorityLevel.NORMAL
    scheduled_time: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    max_runtime_seconds: int = 3600
    retry_count: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            "task_id": self.task_id,
            "lab_name": self.lab_name,
            "parameters": self.parameters,
            "priority": self.priority.value,
            "scheduled_time": self.scheduled_time.isoformat() if self.scheduled_time else None,
            "dependencies": self.dependencies,
            "max_runtime_seconds": self.max_runtime_seconds,
            "retry_count": self.retry_count,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class ExperimentResult:
    """Result from an experiment execution"""
    task_id: str
    lab_name: str
    status: ExperimentStatus
    result_data: Optional[Dict] = None
    error_message: Optional[str] = None
    runtime_seconds: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retries_used: int = 0


class UnifiedLabOrchestrator:
    """
    Central coordinator for all QuLabInfinite labs.

    Features:
    - Task queueing with priority levels
    - Dependency tracking and validation
    - Resource allocation and management
    - Result caching and retrieval
    - Multi-lab experiment coordination
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        db_connection_string: Optional[str] = None,
        use_redis: bool = True,
        storage_dir: str = "./orchestrator_storage"
    ):
        """
        Initialize the orchestrator.

        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            db_connection_string: PostgreSQL connection string
            use_redis: Whether to use Redis for queuing
            storage_dir: Directory for file-based storage
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Redis if available
        self.redis_client = None
        if REDIS_AVAILABLE and use_redis:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    decode_responses=True
                )
                self.redis_client.ping()
                logger.info("✓ Connected to Redis")
            except Exception as e:
                logger.warning(f"Redis unavailable: {e}. Using file-based queuing.")
                self.redis_client = None

        # Initialize PostgreSQL if available
        self.db_connection_string = db_connection_string
        self.psycopg2_available = PSYCOPG2_AVAILABLE and db_connection_string is not None
        if self.psycopg2_available:
            try:
                self._initialize_database()
                logger.info("✓ Connected to PostgreSQL")
            except Exception as e:
                logger.warning(f"PostgreSQL unavailable: {e}. Using file-based storage.")
                self.psycopg2_available = False

        # In-memory structures
        self.task_queue: asyncio.PriorityQueue = None
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_cache: Dict[str, ExperimentResult] = {}
        self.lab_registry: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self.stats = {
            "total_tasks_processed": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "total_compute_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }

        logger.info("✓ Unified Lab Orchestrator initialized")

    def _initialize_database(self):
        """Initialize PostgreSQL database schema"""
        if not self.psycopg2_available:
            return

        try:
            conn = psycopg2.connect(self.db_connection_string)
            cur = conn.cursor()

            # Create tables
            create_tables_sql = """
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
                retries_used INT DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_lab_name ON experiment_results(lab_name);
            CREATE INDEX IF NOT EXISTS idx_status ON experiment_results(status);
            CREATE INDEX IF NOT EXISTS idx_created_at ON experiment_results(created_at);

            CREATE TABLE IF NOT EXISTS task_dependencies (
                dependent_task_id VARCHAR(255),
                dependency_task_id VARCHAR(255),
                PRIMARY KEY (dependent_task_id, dependency_task_id),
                FOREIGN KEY (dependent_task_id) REFERENCES experiment_results(task_id),
                FOREIGN KEY (dependency_task_id) REFERENCES experiment_results(task_id)
            );
            """

            cur.execute(create_tables_sql)
            conn.commit()
            cur.close()
            conn.close()

        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise

    def register_lab(self, lab_name: str, lab_interface: Any) -> None:
        """
        Register a lab for orchestration.

        Args:
            lab_name: Name of the lab
            lab_interface: Lab interface object with run_experiment() method
        """
        self.lab_registry[lab_name] = {
            "interface": lab_interface,
            "registered_at": datetime.now(),
            "total_experiments": 0,
            "successful_experiments": 0,
            "failed_experiments": 0,
            "avg_runtime": 0.0
        }
        logger.info(f"✓ Registered lab: {lab_name}")

    async def submit_experiment(self, task: ExperimentTask) -> str:
        """
        Submit an experiment to the queue.

        Args:
            task: ExperimentTask to execute

        Returns:
            task_id: Unique identifier for the task
        """
        # Validate lab exists
        if task.lab_name not in self.lab_registry:
            raise ValueError(f"Lab '{task.lab_name}' not registered")

        # Store task metadata in cache/database
        await self._store_task(task)

        # Add to queue (use Redis if available, else in-memory)
        if self.redis_client:
            queue_entry = {
                "task_id": task.task_id,
                "priority": task.priority.value,
                "created_at": datetime.now().isoformat()
            }
            self.redis_client.zadd(
                "experiment_queue",
                {json.dumps(queue_entry): task.priority.value}
            )
            self.redis_client.hset(
                f"task:{task.task_id}",
                mapping=task.to_dict()
            )
        else:
            # File-based fallback
            queue_file = self.storage_dir / "queue.jsonl"
            with open(queue_file, "a") as f:
                f.write(json.dumps(task.to_dict()) + "\n")

        logger.info(f"✓ Task {task.task_id} queued for {task.lab_name}")
        return task.task_id

    async def _store_task(self, task: ExperimentTask) -> None:
        """Store task in database"""
        if self.psycopg2_available:
            try:
                conn = psycopg2.connect(self.db_connection_string)
                cur = conn.cursor()

                # We'll update this once the task completes
                # For now, just store initial metadata
                cur.close()
                conn.close()
            except Exception as e:
                logger.warning(f"Error storing task metadata: {e}")

    async def process_queue(self, max_concurrent: int = 10) -> None:
        """
        Main loop to process queued tasks.

        Args:
            max_concurrent: Maximum concurrent tasks to run
        """
        if self.task_queue is None:
            self.task_queue = asyncio.PriorityQueue()

        logger.info(f"Starting queue processor (max concurrent: {max_concurrent})")

        while True:
            try:
                # Check running tasks
                running_count = len(self.running_tasks)

                if running_count < max_concurrent:
                    # Get next task from queue
                    task = await self._get_next_task()

                    if task:
                        # Check dependencies
                        if not await self._check_dependencies(task):
                            # Re-queue if dependencies not met
                            await asyncio.sleep(5)
                            continue

                        # Run task
                        task_coroutine = self._run_task(task)
                        task_handle = asyncio.create_task(task_coroutine)
                        self.running_tasks[task.task_id] = task_handle

                        # Clean up completed tasks
                        completed = [
                            tid for tid, t in self.running_tasks.items()
                            if t.done()
                        ]
                        for tid in completed:
                            del self.running_tasks[tid]

                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                await asyncio.sleep(5)

    async def _get_next_task(self) -> Optional[ExperimentTask]:
        """Get next task from queue"""
        # Simplified - in production would handle Redis/file-based queues
        if not self.task_queue.empty():
            try:
                priority, task_id, task = self.task_queue.get_nowait()
                return task
            except asyncio.QueueEmpty:
                return None
        return None

    async def _check_dependencies(self, task: ExperimentTask) -> bool:
        """Check if all task dependencies are complete"""
        if not task.dependencies:
            return True

        for dep_id in task.dependencies:
            result = await self.get_task_result(dep_id)
            if not result or result.status != ExperimentStatus.COMPLETED:
                return False

        return True

    async def _run_task(self, task: ExperimentTask) -> ExperimentResult:
        """
        Execute a single task.

        Args:
            task: ExperimentTask to execute

        Returns:
            ExperimentResult with outcome
        """
        task_id = task.task_id
        lab_name = task.lab_name
        retries_used = 0

        logger.info(f"Starting task {task_id} on {lab_name}")

        while retries_used < task.retry_count:
            try:
                # Update status
                await self._update_task_status(task_id, ExperimentStatus.RUNNING)

                # Get lab interface
                lab_info = self.lab_registry[lab_name]
                lab = lab_info["interface"]

                # Run with timeout
                start_time = datetime.now()

                try:
                    result = await asyncio.wait_for(
                        self._execute_lab_experiment(lab, task.parameters),
                        timeout=task.max_runtime_seconds
                    )
                    end_time = datetime.now()
                    runtime = (end_time - start_time).total_seconds()

                    # Create result
                    experiment_result = ExperimentResult(
                        task_id=task_id,
                        lab_name=lab_name,
                        status=ExperimentStatus.COMPLETED,
                        result_data=result,
                        runtime_seconds=runtime,
                        started_at=start_time,
                        completed_at=end_time,
                        retries_used=retries_used
                    )

                    # Store result
                    await self._store_result(experiment_result)
                    self.task_cache[task_id] = experiment_result

                    # Update stats
                    self.stats["total_tasks_processed"] += 1
                    self.stats["successful_tasks"] += 1
                    self.stats["total_compute_time"] += runtime
                    lab_info["total_experiments"] += 1
                    lab_info["successful_experiments"] += 1

                    logger.info(
                        f"✓ Task {task_id} completed in {runtime:.1f}s"
                    )

                    return experiment_result

                except asyncio.TimeoutError:
                    raise TimeoutError(f"Task exceeded max runtime of {task.max_runtime_seconds}s")

            except Exception as e:
                retries_used += 1
                logger.warning(
                    f"Task {task_id} failed (attempt {retries_used}/{task.retry_count}): {e}"
                )

                if retries_used >= task.retry_count:
                    # Final failure
                    experiment_result = ExperimentResult(
                        task_id=task_id,
                        lab_name=lab_name,
                        status=ExperimentStatus.FAILED,
                        error_message=str(e),
                        retries_used=retries_used
                    )

                    await self._store_result(experiment_result)
                    self.task_cache[task_id] = experiment_result

                    self.stats["total_tasks_processed"] += 1
                    self.stats["failed_tasks"] += 1
                    lab_info["total_experiments"] += 1
                    lab_info["failed_experiments"] += 1

                    logger.error(f"✗ Task {task_id} failed after {retries_used} retries")

                    return experiment_result

                # Wait before retry
                await asyncio.sleep(5 * retries_used)

        return ExperimentResult(
            task_id=task_id,
            lab_name=lab_name,
            status=ExperimentStatus.FAILED,
            error_message="Max retries exceeded"
        )

    async def _execute_lab_experiment(self, lab: Any, parameters: Dict) -> Dict:
        """Execute lab experiment (handle sync or async)"""
        if asyncio.iscoroutinefunction(lab.run_experiment):
            return await lab.run_experiment(parameters)
        else:
            # Run sync function in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lab.run_experiment,
                parameters
            )

    async def _update_task_status(self, task_id: str, status: ExperimentStatus) -> None:
        """Update task status in cache/database"""
        if self.redis_client:
            self.redis_client.hset(f"task:{task_id}", "status", status.value)

    async def _store_result(self, result: ExperimentResult) -> None:
        """Store experiment result in database"""
        if self.psycopg2_available:
            try:
                conn = psycopg2.connect(self.db_connection_string)
                cur = conn.cursor()

                query = """
                INSERT INTO experiment_results
                (task_id, lab_name, parameters, result, status, runtime_seconds,
                 started_at, completed_at, error_message, retries_used)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (task_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    result = EXCLUDED.result,
                    runtime_seconds = EXCLUDED.runtime_seconds,
                    completed_at = EXCLUDED.completed_at,
                    error_message = EXCLUDED.error_message
                """

                cur.execute(query, (
                    result.task_id,
                    result.lab_name,
                    json.dumps({}),  # parameters placeholder
                    json.dumps(result.result_data or {}),
                    result.status.value,
                    result.runtime_seconds,
                    result.started_at,
                    result.completed_at,
                    result.error_message,
                    result.retries_used
                ))

                conn.commit()
                cur.close()
                conn.close()

            except Exception as e:
                logger.warning(f"Error storing result: {e}")

    async def get_task_result(self, task_id: str) -> Optional[ExperimentResult]:
        """
        Retrieve result for a completed task.

        Args:
            task_id: Task identifier

        Returns:
            ExperimentResult if found, None otherwise
        """
        # Check cache first
        if task_id in self.task_cache:
            self.stats["cache_hits"] += 1
            return self.task_cache[task_id]

        self.stats["cache_misses"] += 1

        # Query from database if available
        if self.psycopg2_available:
            try:
                conn = psycopg2.connect(self.db_connection_string)
                cur = conn.cursor()

                query = """
                SELECT task_id, lab_name, status, result, error_message,
                       runtime_seconds, started_at, completed_at, retries_used
                FROM experiment_results
                WHERE task_id = %s
                """

                cur.execute(query, (task_id,))
                row = cur.fetchone()

                if row:
                    return ExperimentResult(
                        task_id=row[0],
                        lab_name=row[1],
                        status=ExperimentStatus(row[2]),
                        result_data=row[3],
                        error_message=row[4],
                        runtime_seconds=row[5],
                        started_at=row[6],
                        completed_at=row[7],
                        retries_used=row[8]
                    )

                cur.close()
                conn.close()

            except Exception as e:
                logger.warning(f"Error retrieving result from database: {e}")

        return None

    def get_lab_status(self, lab_name: Optional[str] = None) -> Dict:
        """
        Get status of labs.

        Args:
            lab_name: Specific lab name or None for all

        Returns:
            Status dictionary for lab(s)
        """
        if lab_name:
            if lab_name not in self.lab_registry:
                return {}

            info = self.lab_registry[lab_name]
            return {
                "lab_name": lab_name,
                "registered_at": info["registered_at"].isoformat(),
                "total_experiments": info["total_experiments"],
                "successful_experiments": info["successful_experiments"],
                "failed_experiments": info["failed_experiments"],
                "success_rate": (
                    info["successful_experiments"] / max(info["total_experiments"], 1)
                    * 100
                )
            }
        else:
            return {
                lab: self.get_lab_status(lab)
                for lab in self.lab_registry.keys()
            }

    def get_stats(self) -> Dict:
        """Get orchestrator statistics"""
        return {
            **self.stats,
            "running_tasks": len(self.running_tasks),
            "cache_size": len(self.task_cache),
            "registered_labs": len(self.lab_registry),
            "timestamp": datetime.now().isoformat()
        }
