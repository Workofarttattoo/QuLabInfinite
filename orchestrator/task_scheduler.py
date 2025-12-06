"""
Task Scheduler - Schedule experiments at specific times with cron-like functionality
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScheduledExperiment:
    """A scheduled experiment"""
    experiment_name: str
    lab_name: str
    parameters: Dict
    schedule: str  # Cron-like: "0 * * * *" (hourly), "0 0 * * *" (daily), etc.
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    callback: Optional[Callable] = None


class TaskScheduler:
    """
    Schedule recurring experiments with cron-like syntax.

    Supports:
    - Hourly runs: "0 * * * *"
    - Daily runs: "0 0 * * *"
    - Weekly runs: "0 0 * * 0" (Sundays)
    - Custom intervals via callback functions
    """

    def __init__(self):
        """Initialize task scheduler"""
        self.scheduled_tasks: Dict[str, ScheduledExperiment] = {}
        self.running = False

    def schedule_experiment(
        self,
        experiment_name: str,
        lab_name: str,
        parameters: Dict,
        schedule: str,
        callback: Optional[Callable] = None
    ) -> str:
        """
        Schedule a recurring experiment.

        Args:
            experiment_name: Name of the experiment
            lab_name: Lab to run experiment in
            parameters: Parameters for the experiment
            schedule: Cron-like schedule string
            callback: Optional callback function after execution

        Returns:
            experiment_name
        """
        scheduled = ScheduledExperiment(
            experiment_name=experiment_name,
            lab_name=lab_name,
            parameters=parameters,
            schedule=schedule,
            callback=callback,
            next_run=self._calculate_next_run(schedule)
        )

        self.scheduled_tasks[experiment_name] = scheduled

        logger.info(
            f"✓ Scheduled {experiment_name} on {lab_name} "
            f"with schedule '{schedule}'"
        )

        return experiment_name

    def _calculate_next_run(self, schedule: str) -> datetime:
        """
        Calculate next run time from cron schedule.

        Simple implementation - in production use croniter library.
        """
        parts = schedule.split()

        if len(parts) != 5:
            raise ValueError(f"Invalid cron format: {schedule}")

        minute, hour, day, month, weekday = parts

        # Simple cases
        now = datetime.now()

        if schedule == "0 * * * *":  # Every hour
            return now + timedelta(hours=1)
        elif schedule == "0 0 * * *":  # Every day
            return (now + timedelta(days=1)).replace(hour=0, minute=0)
        elif schedule == "0 0 * * 0":  # Weekly (Sunday)
            days_ahead = 6 - now.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return (now + timedelta(days=days_ahead)).replace(hour=0, minute=0)
        else:
            # Default: run in 1 hour
            return now + timedelta(hours=1)

    async def start_scheduler(self, orchestrator) -> None:
        """
        Start the scheduler loop.

        Args:
            orchestrator: UnifiedLabOrchestrator instance
        """
        self.running = True
        logger.info("✓ Task Scheduler started")

        while self.running:
            try:
                now = datetime.now()

                for exp_name, scheduled in self.scheduled_tasks.items():
                    if not scheduled.enabled:
                        continue

                    if scheduled.next_run and now >= scheduled.next_run:
                        logger.info(
                            f"Triggering scheduled experiment: {exp_name}"
                        )

                        # Submit experiment to orchestrator
                        from .unified_orchestrator import ExperimentTask

                        task = ExperimentTask(
                            lab_name=scheduled.lab_name,
                            parameters=scheduled.parameters,
                            metadata={"scheduled_experiment": exp_name}
                        )

                        task_id = await orchestrator.submit_experiment(task)

                        # Update last run and calculate next run
                        scheduled.last_run = now
                        scheduled.next_run = self._calculate_next_run(
                            scheduled.schedule
                        )

                        # Call callback if provided
                        if scheduled.callback:
                            if asyncio.iscoroutinefunction(scheduled.callback):
                                await scheduled.callback(task_id)
                            else:
                                scheduled.callback(task_id)

                        logger.info(
                            f"Scheduled task {exp_name} submitted as {task_id}, "
                            f"next run at {scheduled.next_run}"
                        )

                # Sleep before next check
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(60)

    def stop_scheduler(self) -> None:
        """Stop the scheduler"""
        self.running = False
        logger.info("Task Scheduler stopped")

    def get_schedule_status(self) -> Dict:
        """Get status of all scheduled tasks"""
        return {
            exp_name: {
                "lab_name": scheduled.lab_name,
                "schedule": scheduled.schedule,
                "enabled": scheduled.enabled,
                "last_run": scheduled.last_run.isoformat() if scheduled.last_run else None,
                "next_run": scheduled.next_run.isoformat() if scheduled.next_run else None
            }
            for exp_name, scheduled in self.scheduled_tasks.items()
        }

    def disable_schedule(self, experiment_name: str) -> None:
        """Disable a scheduled experiment"""
        if experiment_name in self.scheduled_tasks:
            self.scheduled_tasks[experiment_name].enabled = False
            logger.info(f"Disabled schedule for {experiment_name}")

    def enable_schedule(self, experiment_name: str) -> None:
        """Enable a scheduled experiment"""
        if experiment_name in self.scheduled_tasks:
            self.scheduled_tasks[experiment_name].enabled = True
            logger.info(f"Enabled schedule for {experiment_name}")
