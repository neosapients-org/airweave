"""Temporal workflow and activity for discover and propose jobs, on their own task queue.

Registered only by the jobs worker (``worker.py``), never by the sync worker's ``wiring.py``.
The workflow is a thin shell: one activity that runs the job. The job row holds the state, a
lease fences a second attempt, and the activity heartbeats so a hung build is noticed. A crashed
attempt is retried by Temporal's activity timeout and resumes from the row. The workflow may end
while the row is still active (the retries ran out, or the lease stayed held for every wait); the
jobs worker's reconciler (``housekeeping.py``) then starts it again, and fails a job that stays
unattended for longer than ``TAXONOMY_JOBS_MAX_AGE_HOURS``.
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


BUSY = "busy"
# A lease is held for TAXONOMY_JOBS_LEASE_SECONDS (300 by default). A dead worker's lease
# expires after that, so the workflow waits a little longer and asks again, instead of ending
# while the job row still says RUNNING (it did once: the first attempt died holding the lease and
# the second saw "busy").
BUSY_WAIT = timedelta(seconds=330)
BUSY_ATTEMPTS = 6


async def run_until_not_busy(run: Any, sleep: Any, attempts: int = BUSY_ATTEMPTS) -> str:
    """Call ``run`` until it returns something other than ``busy``, waiting between calls.

    Returns the last answer, which is ``busy`` only when another live worker kept the lease for the
    whole of ``attempts`` waits.
    """
    state: str = BUSY
    for attempt in range(attempts):
        state = await run()
        if state != BUSY:
            return state
        if attempt + 1 < attempts:
            await sleep(BUSY_WAIT)
    return state


@workflow.defn(name=WORKFLOW_NAME)
class TaxonomyJobWorkflow:
    """Runs one job's activity; the workflow id is the job id, so a replayed start is a no-op."""

    @workflow.run
    async def run(self, job_id: str) -> str:
        """Run the job and return its final state."""

        async def attempt() -> str:
            state: str = await workflow.execute_activity(
                ACTIVITY_NAME,
                job_id,
                start_to_close_timeout=timedelta(hours=6),
                heartbeat_timeout=timedelta(minutes=10),
                retry_policy=RetryPolicy(
                    maximum_attempts=3, initial_interval=timedelta(seconds=30)
                ),
            )
            return state

        return await run_until_not_busy(attempt, workflow.sleep)


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
    """Start the job's workflow, or do nothing when one is already running for it.

    An id that is running is the same job (``USE_EXISTING``); one whose workflow has ended while
    the row is still active (the retries ran out, or ``busy`` ended it) is started again, which is
    how the reconciler recovers it.
    """
    from temporalio.common import WorkflowIDConflictPolicy, WorkflowIDReusePolicy

    await client.start_workflow(
        WORKFLOW_NAME,
        str(job_id),
        id=f"{WORKFLOW_ID_PREFIX}{job_id}",
        task_queue=task_queue,
        id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE,
        id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
    )
