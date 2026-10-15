"""Temporal workflow and activity for discover and propose jobs, on their own task queue.

Registered only by the jobs worker (``worker.py``), never by the sync worker's ``wiring.py``.
The workflow is a thin shell: one activity that runs the job. Retries are off: the job row holds
the state, a lease fences a second attempt, and the activity heartbeats so a hung build is
noticed. A crashed attempt is retried by Temporal's activity timeout and resumes from the row.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from uuid import UUID

from temporalio import activity, workflow
from temporalio.common import RetryPolicy

ACTIVITY_NAME = "run_taxonomy_job_activity"
WORKFLOW_NAME = "TaxonomyJobWorkflow"
WORKFLOW_ID_PREFIX = "taxonomy-job-"

with workflow.unsafe.imports_passed_through():
    pass


@workflow.defn(name=WORKFLOW_NAME)
class TaxonomyJobWorkflow:
    """Runs one job's activity; the workflow id is the job id, so a replayed start is a no-op."""

    @workflow.run
    async def run(self, job_id: str) -> str:
        """Run the job and return its final state."""
        state: str = await workflow.execute_activity(
            ACTIVITY_NAME,
            job_id,
            start_to_close_timeout=timedelta(hours=6),
            heartbeat_timeout=timedelta(minutes=10),
            retry_policy=RetryPolicy(maximum_attempts=3, initial_interval=timedelta(seconds=30)),
        )
        return state


class TaxonomyJobActivity:
    """The activity: ``runner.run_job`` with a heartbeat while it waits."""

    def __init__(self, runner: Any) -> None:
        """``runner`` is a ``TaxonomyJobRunner``."""
        self._runner = runner

    @activity.defn(name=ACTIVITY_NAME)
    async def run(self, job_id: str) -> str:
        """Run the job; returns the final state (``busy`` when another attempt holds the lease)."""
        import asyncio

        async def beat() -> None:
            while True:
                activity.heartbeat()
                await asyncio.sleep(30)

        task = asyncio.create_task(beat())
        try:
            state: str = await self._runner.run_job(UUID(job_id))
            return state
        finally:
            task.cancel()


async def start_job_workflow(client: Any, task_queue: str, job_id: UUID) -> None:
    """Start the job's workflow; an id that is already running is the same job (a replay)."""
    from temporalio.common import WorkflowIDReusePolicy
    from temporalio.exceptions import WorkflowAlreadyStartedError

    try:
        await client.start_workflow(
            WORKFLOW_NAME,
            str(job_id),
            id=f"{WORKFLOW_ID_PREFIX}{job_id}",
            task_queue=task_queue,
            id_reuse_policy=WorkflowIDReusePolicy.REJECT_DUPLICATE,
        )
    except WorkflowAlreadyStartedError:
        return
